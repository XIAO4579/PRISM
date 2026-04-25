"""
Dense VL Discriminator Warmup Training Script

使用 Qwen3-VL-4B dense 模型进行 pairwise warmup 训练，
训练完成后可以作为 verl 中的 critic model (value head)

与 MoE 版本的区别：
  - 直接使用 dense backbone，无需 sparse upcycling
  - 没有 aux_loss (无 router 负载均衡)
  - 不需要 find_unused_parameters=True
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
import subprocess
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

# 导入公共模块
from value_head import Qwen3VLDenseValueHead, get_last_token_scores


# --- 配置 ---
DENSE_MODEL_PATH = "/data/user/swang886/gad_project/models/Qwen3-VL-4B-Instruct"
DATA_PATH = "/data/user/swang886/gad_project/datasets/data_process/data_pipeline/data_process_7.6K/dataset_process/api_output_qwen3_full_sft_warmup_dataset/qwen3_vl_moe_warmup_pairwise_120k.jsonl"
OUTPUT_DIR = "/data/user/swang886/gad_project/models/Qwen3-VL-4B-dense-warmup"

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

MAX_LENGTH = 4096
BATCH_SIZE = 14
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
SEED = 42
GRADIENT_CHECKPOINTING = True
# 仅用于快速 debug：限制读取的数据条数（0 表示不限制）
WARMUP_MAX_RECORDS = int(os.environ.get("WARMUP_MAX_RECORDS", "0"))


def load_image(image_source, max_size=560):
    """
    加载图像，支持本地路径和 URL，并限制最大尺寸

    Args:
        image_source: 图像路径或 URL
        max_size: 最大边长，超过会按比例缩放

    Returns:
        PIL.Image 对象
    """
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_source).convert('RGB')

    # 强制限制图片最大尺寸，避免产生过多 image token
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    return image


class PairwiseVLDataset(Dataset):
    """
    Pairwise VL 判别器训练数据集

    数据格式 (JSONL):
    {
        "image": "path/to/image.jpg" 或 "http://...",  # 可选
        "prompt": "问题内容",
        "teacher_caption": "Teacher 的 caption",
        "student_caption": "Student 的 caption",
        "teacher_cot": "Teacher 的 CoT",
        "student_cot": "Student 的 CoT"
    }

    会分别生成 Caption 对比 和 CoT 对比两种样本
    """
    def __init__(
        self,
        data_path,
        processor,
        max_length=4096,
        image_base_path=None,
        max_pixels=560 * 560,
        max_records=None,
    ):
        self.samples = []
        self.processor = processor
        self.max_length = max_length
        self.image_base_path = image_base_path
        self.max_records = max_records

        # 设定图像处理参数
        if hasattr(processor, "image_processor"):
            processor.image_processor.max_pixels = max_pixels
            if not hasattr(processor.image_processor, "min_pixels"):
                processor.image_processor.min_pixels = 28 * 28
            print(f"Image processor: max_pixels={processor.image_processor.max_pixels}, "
                  f"min_pixels={processor.image_processor.min_pixels}")

        if not os.path.exists(data_path):
            print(f"Warning: Data path {data_path} does not exist. Creating dummy data.")
            self._create_dummy_data(data_path)

        mismatch_count = 0
        used_records = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if self.max_records is not None and used_records >= self.max_records:
                    break
                if line.strip():
                    item = json.loads(line)
                    used_records += 1
                    prompt = item.get('prompt', item.get('question', ''))
                    image = item.get('image', None)

                    # 提取 Teacher/Student 的 Caption 和 CoT
                    teacher_caption = item.get('teacher_caption', '')
                    teacher_cot = item.get('teacher_cot', '')
                    student_caption = item.get('student_caption', '')
                    student_cot = item.get('student_cot', '')

                    # 兼容旧格式：teacher_response / student_response
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

                    # 检查 caption 和 cot 是否都存在
                    has_caption = bool(prompt and teacher_caption and student_caption)
                    has_cot = bool(prompt and teacher_cot and student_cot)

                    # 只有 caption 和 cot 都完整时才使用这条数据
                    if not (has_caption and has_cot):
                        mismatch_count += 1
                        if mismatch_count <= 5:
                            print(f"Warning: Line {line_idx} incomplete, skipped: "
                                  f"caption=({bool(teacher_caption)},{bool(student_caption)}), "
                                  f"cot=({bool(teacher_cot)},{bool(student_cot)})")
                        continue

                    # Caption 对比
                    self.samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_caption,
                        'student_res': student_caption,
                        'type': 'caption',
                    })

                    # CoT 对比
                    self.samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_cot,
                        'student_res': student_cot,
                        'type': 'cot',
                    })

        if mismatch_count > 0:
            print(f"Warning: Skipped {mismatch_count} incomplete lines")
        if self.max_records is not None:
            print(f"Debug mode: only used first {used_records} records from dataset")

        # 统计各类型样本数量
        type_counts = {}
        for s in self.samples:
            t = s.get('type', 'unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"Loaded {len(self.samples)} comparison pairs: {type_counts}")

    def _create_dummy_data(self, path):
        """创建测试用的 dummy 数据"""
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
        """
        构建对话格式的消息

        Args:
            item: 样本数据
            role: 'teacher' 或 'student'
        """
        prompt = item.get('prompt', '')
        response = item.get(f'{role}_res', '')
        image_source = item.get('image', None)

        # 如果是相对路径，拼接基础路径
        if image_source and self.image_base_path and not image_source.startswith(('http://', 'https://', '/')):
            image_source = os.path.join(self.image_base_path, image_source)

        # 构建 user content
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

        # 构建 Teacher 和 Student 的消息
        teacher_messages, image_source = self._build_messages(item, 'teacher')
        student_messages, _ = self._build_messages(item, 'student')

        # 加载图像（如果有）
        image = None
        if image_source:
            try:
                image = load_image(image_source)
            except Exception as e:
                print(f"Warning: Failed to load image {image_source}: {e}")
                teacher_messages[1]["content"] = [
                    c for c in teacher_messages[1]["content"] if c.get("type") != "image"
                ]
                student_messages[1]["content"] = [
                    c for c in student_messages[1]["content"] if c.get("type") != "image"
                ]

        # 处理 Teacher 输入
        teacher_text = self.processor.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False
        )
        teacher_inputs = self.processor(
            text=[teacher_text],
            images=[image] if image else None,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )

        # 处理 Student 输入
        student_text = self.processor.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False
        )
        student_inputs = self.processor(
            text=[student_text],
            images=[image] if image else None,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )

        result = {
            "teacher_input_ids": teacher_inputs['input_ids'].squeeze(0),
            "teacher_attention_mask": teacher_inputs['attention_mask'].squeeze(0),
            "student_input_ids": student_inputs['input_ids'].squeeze(0),
            "student_attention_mask": student_inputs['attention_mask'].squeeze(0),
        }

        # 添加视觉输入（如果有）
        if 'pixel_values' in teacher_inputs:
            result["teacher_pixel_values"] = teacher_inputs['pixel_values'].squeeze(0)
            result["student_pixel_values"] = student_inputs['pixel_values'].squeeze(0)
        if 'image_grid_thw' in teacher_inputs:
            thw = teacher_inputs['image_grid_thw']
            result["teacher_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw
            thw = student_inputs['image_grid_thw']
            result["student_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw

        return result


def collate_fn(batch):
    """
    自定义 collate 函数，处理可变长度的视觉输入
    """
    result = {}

    # 基础文本输入
    result['teacher_input_ids'] = torch.stack([b['teacher_input_ids'] for b in batch])
    result['teacher_attention_mask'] = torch.stack([b['teacher_attention_mask'] for b in batch])
    result['student_input_ids'] = torch.stack([b['student_input_ids'] for b in batch])
    result['student_attention_mask'] = torch.stack([b['student_attention_mask'] for b in batch])

    # 视觉输入
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


def train():
    set_seed(SEED)

    from accelerate import DeepSpeedPlugin

    ds_config_path = os.path.join(os.path.dirname(__file__), "ds_z2_config.json")
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=ds_config_path,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_clipping=1.0,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin,
    )

    # 加载 processor
    processor = AutoProcessor.from_pretrained(DENSE_MODEL_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 加载模型 - 使用自定义的 Qwen3VLDenseValueHead
    accelerator.print(f"Loading Dense VL model with Value Head from {DENSE_MODEL_PATH}...")
    with accelerator.main_process_first():
        model = Qwen3VLDenseValueHead(
            DENSE_MODEL_PATH,
            num_labels=1,
            gradient_checkpointing=GRADIENT_CHECKPOINTING
        )

    accelerator.print(f"Model type: {type(model).__name__}")
    text_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    accelerator.print(f"Model config: num_labels={model.num_labels}, "
                      f"hidden_size={text_config.hidden_size}")

    # 数据集
    dataset = PairwiseVLDataset(
        DATA_PATH,
        processor,
        max_length=MAX_LENGTH,
        max_records=WARMUP_MAX_RECORDS if WARMUP_MAX_RECORDS > 0 else None,
    )
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

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # 学习率调度器
    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Prepare for distributed training
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    accelerator.print("=" * 60)
    accelerator.print("Starting Dense VL Value Head Warmup Training")
    accelerator.print(f"  Model: {DENSE_MODEL_PATH}")
    accelerator.print(f"  Output: {OUTPUT_DIR}")
    accelerator.print(f"  Num samples: {len(dataset)}")
    accelerator.print(f"  Batch size: {BATCH_SIZE}")
    accelerator.print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    accelerator.print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes}")
    accelerator.print(f"  Num epochs: {NUM_EPOCHS}")
    accelerator.print(f"  Learning rate: {LEARNING_RATE}")
    accelerator.print(f"  Max length: {MAX_LENGTH}")
    accelerator.print("=" * 60)

    model.train()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        total_accuracy = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # 合并 teacher/student 为一次 forward，减少 kernel launch 开销
                bsz = batch['teacher_input_ids'].shape[0]

                merged_kwargs = {
                    'input_ids': torch.cat([batch['teacher_input_ids'], batch['student_input_ids']], dim=0),
                    'attention_mask': torch.cat([batch['teacher_attention_mask'], batch['student_attention_mask']], dim=0),
                }
                if 'teacher_pixel_values' in batch and 'student_pixel_values' in batch:
                    merged_kwargs['pixel_values'] = torch.cat(
                        [batch['teacher_pixel_values'], batch['student_pixel_values']], dim=0
                    )
                if 'teacher_image_grid_thw' in batch and 'student_image_grid_thw' in batch:
                    merged_kwargs['image_grid_thw'] = torch.cat(
                        [batch['teacher_image_grid_thw'], batch['student_image_grid_thw']], dim=0
                    )

                merged_outputs = model(**merged_kwargs)
                merged_scores = get_last_token_scores(
                    merged_outputs['logits'], merged_kwargs['attention_mask']
                )

                teacher_scores = merged_scores[:bsz]
                student_scores = merged_scores[bsz:]

                # Pairwise Loss (Bradley-Terry)
                diff = teacher_scores - student_scores
                loss = -F.logsigmoid(diff).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    global_step += 1

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                accuracy = (diff > 0).float().mean().item()
                total_accuracy += accuracy

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{accuracy:.2%}",
                    "diff": f"{diff.mean().item():.2f}",
                })

                # 每50步打印 GPU 显存使用情况
                if step % 50 == 0 and accelerator.is_main_process:
                    try:
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5
                        )
                        print(f"\n[Step {step}] GPU Memory Usage:")
                        for line in result.stdout.strip().split('\n'):
                            idx, used, total, util = [x.strip() for x in line.split(',')]
                            print(f"  GPU {idx}: {used}/{total} MiB ({util}% util)")
                    except Exception as e:
                        print(f"\n[Step {step}] nvidia-smi failed: {e}")

        # Epoch 统计
        n_steps = len(dataloader)
        accelerator.print(f"\nEpoch {epoch+1} Summary:")
        accelerator.print(f"  Avg Loss: {total_loss / n_steps:.4f}")
        accelerator.print(f"  Avg Accuracy: {total_accuracy / n_steps:.2%}")

    # 保存模型
    accelerator.wait_for_everyone()
    accelerator.print(f"\nSaving model to {OUTPUT_DIR}...")

    unwrapped_model = accelerator.unwrap_model(model)

    # ZeRO-2 下 accelerator.get_state_dict 会自动 gather 所有分片
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        base_state = {}
        v_head_state = {}
        for k, v in state_dict.items():
            if k.startswith("v_head."):
                v_head_state[k.replace("v_head.", "")] = v.to(torch.bfloat16)
            else:
                clean_key = k.removeprefix("base_model.")
                base_state[clean_key] = v.to(torch.bfloat16)

        merged_state = dict(base_state)
        for k, v in v_head_state.items():
            merged_state[f"v_head.{k}"] = v
        unwrapped_model.base_model.save_pretrained(
            OUTPUT_DIR, state_dict=merged_state
        )

        # 单独保存 value_head.pt（兼容 trl/verl）
        v_head_trl_state = {}
        for k, v in v_head_state.items():
            if k == "1.weight":
                v_head_trl_state["v_head.summary.weight"] = v.detach().clone()
            elif k == "1.bias":
                v_head_trl_state["v_head.summary.bias"] = v.detach().clone()
            else:
                v_head_trl_state[f"v_head.{k}"] = v.detach().clone()

        torch.save(v_head_trl_state, os.path.join(OUTPUT_DIR, "value_head.pt"))
        processor.save_pretrained(OUTPUT_DIR)

        accelerator.print(f"\nModel saved to {OUTPUT_DIR}")
        accelerator.print("=" * 60)
        accelerator.print("Saved files:")
        accelerator.print("  - model-*.safetensors (base model + v_head merged)")
        accelerator.print("  - value_head.pt (trl/verl compatible)")
        accelerator.print("")
        accelerator.print("Load with Qwen3VLDenseValueHead:")
        accelerator.print("  from value_head import Qwen3VLDenseValueHead")
        accelerator.print(f"  model = Qwen3VLDenseValueHead.from_pretrained('{OUTPUT_DIR}')")
        accelerator.print("")
        accelerator.print("Load value_head for verl/trl:")
        accelerator.print("  import torch")
        accelerator.print(f"  v_head_state = torch.load('{OUTPUT_DIR}/value_head.pt')")
        accelerator.print("  # Keys: v_head.summary.weight")
        accelerator.print("=" * 60)


if __name__ == "__main__":
    train()
