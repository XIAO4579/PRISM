"""
MoE VL Discriminator Warmup Training Script (Optimized)

基于 train_moe_vl_warmup.py 优化:
1. 动态 padding: 按 batch 内最长序列 left-pad，而非固定 MAX_LENGTH
2. DeepSpeed ZeRO-2: 优化器状态+梯度分片，节省显存、加速通信
3. 移除 find_unused_parameters 开销 (DeepSpeed 不需要)
4. 进度条显示当前 batch 的动态 seq_len，方便观察加速效果
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
import os
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

from value_head import Qwen3VLMoeValueHead, get_last_token_scores


# --- 配置 ---
# 所有路径从环境变量读取，没设置时用占位符 (会显式失败)，保持脚本可发布。
# 通常由 scripts/train/moe_warmup/train_moe_warmup.sh 注入。
MOE_MODEL_PATH = os.environ.get("MOE_MODEL_PATH", "/path/to/Qwen3-VL-2B-MoE-4x")
PROCESSOR_PATH = os.environ.get("PROCESSOR_PATH", MOE_MODEL_PATH)
DATA_PATH = os.environ.get("DATA_PATH", "/path/to/warmup_pairwise.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/path/to/Qwen3-VL-2B-4X-Moe-warmup")
IMAGE_BASE_PATH = os.environ.get("IMAGE_BASE_PATH", "")

SYSTEM_PROMPT = """You are a helpful scientific assistant.
Answer the user's question based on the image provided.

Your output must strictly follow this XML format:

<caption>
- [Briefly describe the image style, key objects, and text using bullet points]
</caption>
<think>
[Reason step-by-step to solve the problem]
</think>
<answer>
[Final answer ONLY]
</answer>
"""

MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 8192))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-5))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 1))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.1))
SEED = int(os.environ.get("SEED", 42))
GRADIENT_CHECKPOINTING = os.environ.get("GRADIENT_CHECKPOINTING", "1") not in ("0", "false", "False")


def load_image(image_source, max_size=560):
    """加载图像，支持本地路径和 URL，并限制最大尺寸"""
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_source).convert('RGB')

    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


class PairwiseVLDataset(Dataset):
    """
    Pairwise VL 判别器训练数据集 (动态 padding 版本)

    与原版的区别: __getitem__ 不做 padding，返回变长 tensor，
    padding 统一在 collate_fn 中按 batch 内最长序列进行。
    """
    def __init__(self, data_path, processor, max_length=4096, image_base_path=None, max_pixels=560*560):
        self.samples = []
        self.processor = processor
        self.max_length = max_length
        self.image_base_path = image_base_path

        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = max_pixels
            if not hasattr(processor.image_processor, "min_pixels"):
                processor.image_processor.min_pixels = 28 * 28
            print(f"image_processor: max_pixels={processor.image_processor.max_pixels}, "
                  f"min_pixels={processor.image_processor.min_pixels}")

        if not os.path.exists(data_path):
            print(f"Warning: Data path {data_path} does not exist. Creating dummy data.")
            self._create_dummy_data(data_path)

        mismatch_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    prompt = item.get('prompt', item.get('question', ''))
                    image = item.get('image', None)

                    teacher_caption = item.get('teacher_caption', '')
                    teacher_cot = item.get('teacher_cot', '')
                    student_caption = item.get('student_caption', '')
                    student_cot = item.get('student_cot', '')

                    if not teacher_caption and not teacher_cot:
                        teacher_response = item.get('teacher_response', '')
                        student_response = item.get('student_response', '')
                        if teacher_response and student_response:
                            self.samples.append({
                                'prompt': prompt,
                                'image': image,
                                'teacher_res': teacher_response,
                                'student_res': student_response,
                                'type': 'response',
                            })
                        continue

                    has_caption = bool(prompt and teacher_caption and student_caption)
                    has_cot = bool(prompt and teacher_cot and student_cot)

                    if not (has_caption and has_cot):
                        mismatch_count += 1
                        if mismatch_count <= 5:
                            print(f"Warning: Line {line_idx} incomplete, skipped: "
                                  f"caption=({bool(teacher_caption)},{bool(student_caption)}), "
                                  f"cot=({bool(teacher_cot)},{bool(student_cot)})")
                        continue

                    self.samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_caption,
                        'student_res': student_caption,
                        'type': 'caption',
                    })

                    self.samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_cot,
                        'student_res': student_cot,
                        'type': 'cot',
                    })

        if mismatch_count > 0:
            print(f"Warning: Skipped {mismatch_count} incomplete lines")

        type_counts = {}
        for s in self.samples:
            t = s.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Loaded {len(self.samples)} comparison pairs: {type_counts}")

    def _create_dummy_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            for i in range(100):
                f.write(json.dumps({
                    "prompt": f"Question {i}: What is {i} + {i}?",
                    "teacher_caption": f"The answer is {i + i}.",
                    "student_caption": f"The answer is {i + i + 1}.",
                    "teacher_cot": f"Let me think... {i} + {i} = {i + i}.",
                    "student_cot": f"I guess... {i} + {i} = {i + i + 1}."
                }) + "\n")
        print(f"Created dummy data at {path}")

    def __len__(self):
        return len(self.samples)

    def _build_messages(self, item, role='teacher'):
        prompt = item.get('prompt', '')
        response = item.get(f'{role}_res', '')
        image_source = item.get('image', None)

        if image_source and self.image_base_path and not image_source.startswith(('http://', 'https://', '/')):
            image_source = os.path.join(self.image_base_path, image_source)

        user_content = []
        if image_source:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        return messages, image_source

    def __getitem__(self, idx):
        try:
            return self._get_item_impl(idx)
        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            return self._get_item_impl((idx + 1) % len(self.samples))

    def _get_item_impl(self, idx):
        item = self.samples[idx]

        teacher_messages, image_source = self._build_messages(item, 'teacher')
        student_messages, _ = self._build_messages(item, 'student')

        image = None
        if image_source:
            try:
                image = load_image(image_source)
            except Exception as e:
                print(f"Warning: Failed to load image {image_source}: {e}")
                teacher_messages[0]["content"] = [c for c in teacher_messages[0]["content"] if c.get("type") != "image"]
                student_messages[0]["content"] = [c for c in student_messages[0]["content"] if c.get("type") != "image"]

        teacher_text = self.processor.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False
        )
        teacher_inputs = self.processor(
            text=[teacher_text],
            images=[image] if image else None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            max_pixels=560 * 560,
        )

        student_text = self.processor.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False
        )
        student_inputs = self.processor(
            text=[student_text],
            images=[image] if image else None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            max_pixels=560 * 560,
        )

        result = {
            "teacher_input_ids": teacher_inputs['input_ids'].squeeze(0),
            "teacher_attention_mask": teacher_inputs['attention_mask'].squeeze(0),
            "student_input_ids": student_inputs['input_ids'].squeeze(0),
            "student_attention_mask": student_inputs['attention_mask'].squeeze(0),
        }

        if 'pixel_values' in teacher_inputs:
            result["teacher_pixel_values"] = teacher_inputs['pixel_values'].squeeze(0)
            result["student_pixel_values"] = student_inputs['pixel_values'].squeeze(0)
        if 'image_grid_thw' in teacher_inputs:
            thw = teacher_inputs['image_grid_thw']
            result["teacher_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw
            thw = student_inputs['image_grid_thw']
            result["student_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw

        return result


def _left_pad_1d(tensor, target_len, pad_value):
    """Left-pad a 1D tensor to target_len."""
    pad_size = target_len - tensor.size(0)
    if pad_size <= 0:
        return tensor[:target_len]
    return F.pad(tensor, (pad_size, 0), value=pad_value)


def make_collate_fn(pad_token_id):
    """
    创建动态 padding 的 collate 函数。

    所有 teacher 和 student 序列统一 left-pad 到 batch 内的最大长度，
    保证拼接后可以做一次 forward。
    """
    def collate_fn(batch):
        max_len = 0
        for b in batch:
            max_len = max(max_len,
                         b['teacher_input_ids'].size(0),
                         b['student_input_ids'].size(0))

        result = {}

        result['teacher_input_ids'] = torch.stack([
            _left_pad_1d(b['teacher_input_ids'], max_len, pad_token_id) for b in batch
        ])
        result['teacher_attention_mask'] = torch.stack([
            _left_pad_1d(b['teacher_attention_mask'], max_len, 0) for b in batch
        ])
        result['student_input_ids'] = torch.stack([
            _left_pad_1d(b['student_input_ids'], max_len, pad_token_id) for b in batch
        ])
        result['student_attention_mask'] = torch.stack([
            _left_pad_1d(b['student_attention_mask'], max_len, 0) for b in batch
        ])

        teacher_pixels = [b['teacher_pixel_values'] for b in batch if 'teacher_pixel_values' in b]
        student_pixels = [b['student_pixel_values'] for b in batch if 'student_pixel_values' in b]

        if teacher_pixels:
            result['teacher_pixel_values'] = torch.cat(teacher_pixels, dim=0)
            result['student_pixel_values'] = torch.cat(student_pixels, dim=0)

        teacher_thws = []
        student_thws = []
        for b in batch:
            if 'teacher_image_grid_thw' in b:
                t_thw = b['teacher_image_grid_thw']
                s_thw = b['student_image_grid_thw']
                if t_thw.dim() == 1:
                    t_thw = t_thw.unsqueeze(0)
                if s_thw.dim() == 1:
                    s_thw = s_thw.unsqueeze(0)
                teacher_thws.append(t_thw)
                student_thws.append(s_thw)

        if teacher_thws:
            result['teacher_image_grid_thw'] = torch.cat(teacher_thws, dim=0)
            result['student_image_grid_thw'] = torch.cat(student_thws, dim=0)

        return result

    return collate_fn


def train():
    set_seed(SEED)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
    )

    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    pad_token_id = processor.tokenizer.pad_token_id

    accelerator.print(f"Loading VL MoE model with Value Head from {MOE_MODEL_PATH}...")
    with accelerator.main_process_first():
        model = Qwen3VLMoeValueHead(
            MOE_MODEL_PATH,
            num_labels=1,
            gradient_checkpointing=GRADIENT_CHECKPOINTING
        )

    accelerator.print(f"Model type: {type(model).__name__}")
    text_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    accelerator.print(f"Model config: num_labels={model.num_labels}, "
                      f"num_experts={text_config.num_experts}")

    dataset = PairwiseVLDataset(
        DATA_PATH,
        processor,
        max_length=MAX_LENGTH,
        image_base_path=IMAGE_BASE_PATH
    )
    collate_fn = make_collate_fn(pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    is_deepspeed = accelerator.distributed_type.value == "DEEPSPEED"

    accelerator.print("=" * 60)
    accelerator.print("Starting VL MoE Value Head Warmup Training (Optimized)")
    accelerator.print(f"  Model: {MOE_MODEL_PATH}")
    accelerator.print(f"  Output: {OUTPUT_DIR}")
    accelerator.print(f"  Num samples: {len(dataset)}")
    accelerator.print(f"  Batch size: {BATCH_SIZE}")
    accelerator.print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    accelerator.print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes}")
    accelerator.print(f"  Num epochs: {NUM_EPOCHS}")
    accelerator.print(f"  Learning rate: {LEARNING_RATE}")
    accelerator.print(f"  Max length: {MAX_LENGTH}")
    accelerator.print(f"  Dynamic padding: ON")
    accelerator.print(f"  Backend: {'DeepSpeed ZeRO-2' if is_deepspeed else 'DDP'}")
    accelerator.print("=" * 60)

    model.train()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        total_accuracy = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                batch_size = batch['teacher_input_ids'].shape[0]

                combined_input_ids = torch.cat([
                    batch['teacher_input_ids'],
                    batch['student_input_ids']
                ], dim=0)

                combined_attention_mask = torch.cat([
                    batch['teacher_attention_mask'],
                    batch['student_attention_mask']
                ], dim=0)

                combined_kwargs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask,
                }

                if 'teacher_pixel_values' in batch:
                    combined_kwargs['pixel_values'] = torch.cat([
                        batch['teacher_pixel_values'],
                        batch['student_pixel_values']
                    ], dim=0)

                if 'teacher_image_grid_thw' in batch:
                    combined_kwargs['image_grid_thw'] = torch.cat([
                        batch['teacher_image_grid_thw'],
                        batch['student_image_grid_thw']
                    ], dim=0)

                combined_outputs = model(**combined_kwargs, output_router_logits=True)

                combined_logits = combined_outputs['logits']
                teacher_logits = combined_logits[:batch_size]
                student_logits = combined_logits[batch_size:]

                teacher_scores = get_last_token_scores(
                    teacher_logits, batch['teacher_attention_mask']
                )
                student_scores = get_last_token_scores(
                    student_logits, batch['student_attention_mask']
                )

                diff = teacher_scores - student_scores
                pairwise_loss = -F.logsigmoid(diff).mean()

                aux_loss = combined_outputs['aux_loss']

                loss = pairwise_loss
                if aux_loss is not None:
                    unwrapped = accelerator.unwrap_model(model)
                    tc = unwrapped.config.text_config if hasattr(unwrapped.config, 'text_config') else unwrapped.config
                    router_aux_loss_coef = getattr(tc, 'router_aux_loss_coef', 0.01)
                    loss = loss + router_aux_loss_coef * aux_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    global_step += 1

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                accuracy = (diff > 0).float().mean().item()
                total_accuracy += accuracy

                cur_seq_len = combined_input_ids.shape[1]
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "pair": f"{pairwise_loss.item():.4f}",
                    "acc": f"{accuracy:.2%}",
                    "diff": f"{diff.mean().item():.2f}",
                    "seq": cur_seq_len,
                }
                if aux_loss is not None:
                    postfix["aux"] = f"{aux_loss.item():.4f}"
                pbar.set_postfix(postfix)

        n_steps = len(dataloader)
        accelerator.print(f"\nEpoch {epoch+1} Summary:")
        accelerator.print(f"  Avg Loss: {total_loss / n_steps:.4f}")
        accelerator.print(f"  Avg Accuracy: {total_accuracy / n_steps:.2%}")

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"\nSaving model to {OUTPUT_DIR}...")

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)

        accelerator.print(f"\nModel saved to {OUTPUT_DIR}")
        accelerator.print("=" * 60)
        accelerator.print("Saved files:")
        accelerator.print("  - model-*.safetensors (base model + v_head merged)")
        accelerator.print("  - value_head.pt (trl/verl compatible)")
        accelerator.print("")
        accelerator.print("Load with Qwen3VLMoeValueHead:")
        accelerator.print("  from value_head import Qwen3VLMoeValueHead")
        accelerator.print(f"  model = Qwen3VLMoeValueHead.from_pretrained('{OUTPUT_DIR}')")
        accelerator.print("=" * 60)


if __name__ == "__main__":
    train()
