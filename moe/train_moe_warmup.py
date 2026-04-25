"""
MoE Text Discriminator Warmup Training Script (Optimized)

对齐 train_moe_vl_warmup_fast.py 的训练流程（无图像）:
1. 动态 padding: 按 batch 内最长序列 left-pad，而非固定 MAX_LENGTH
2. DeepSpeed ZeRO-2: 优化器状态+梯度分片，节省显存、加速通信
3. Chat template: 使用 apply_chat_template 格式化输入
4. Gradient checkpointing: 节省显存
5. 数据解析: 支持 caption/cot/response 三种格式
6. 进度条显示当前 batch 的动态 seq_len
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from accelerate import Accelerator
from accelerate.utils import set_seed
import json
import os
from tqdm import tqdm


# --- 配置 ---
MOE_MODEL_PATH = "/data/user/swang886/gad_project/models/Qwen3-1.7B-MoE-4x"
TOKENIZER_PATH = "/data/user/swang886/gad_project/models/Qwen3-1.7B-MoE-4x"
DATA_PATH = "/data/user/swang886/gad_project/datasets/data_process/data_pipeline/data_process_7.6K/dataset_process/api_output_qwen3_full_sft_warmup_dataset/qwen3_vl_moe_warmup_pairwise_120k.jsonl"
OUTPUT_DIR = "/data/user/swang886/gad_project/models/Qwen3-1.7B-4X-Moe-warmup"

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

MAX_LENGTH = 8192
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
SEED = 42
GRADIENT_CHECKPOINTING = True
MAX_SAMPLES = 100  # 设为 None 则使用全部数据


def load_moe_for_token_classification(model_path: str, num_labels: int = 1,
                                      gradient_checkpointing: bool = False):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    config.num_labels = num_labels
    config.output_router_logits = True

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        ignore_mismatched_sizes=True,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    return model


def get_last_token_scores(logits, attention_mask):
    batch_size = logits.size(0)
    seq_len = logits.size(1)

    if attention_mask[0, 0] == 0:
        seq_lengths = torch.full((batch_size,), seq_len - 1,
                                 dtype=torch.long, device=logits.device)
    else:
        seq_lengths = attention_mask.sum(dim=1) - 1

    batch_indices = torch.arange(batch_size, device=logits.device)
    scores = logits[batch_indices, seq_lengths]
    return scores


class PairwiseTextDataset(Dataset):
    """
    Pairwise 判别器训练数据集（动态 padding 版本）

    __getitem__ 不做 padding，返回变长 tensor，
    padding 统一在 collate_fn 中按 batch 内最长序列进行。
    """
    def __init__(self, data_path, tokenizer, max_length=4096):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not os.path.exists(data_path):
            print(f"Warning: Data path {data_path} does not exist. Creating dummy data.")
            self._create_dummy_data(data_path)

        mismatch_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get('prompt', item.get('question', ''))

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
                    'teacher_res': teacher_caption,
                    'student_res': student_caption,
                    'type': 'caption',
                })

                self.samples.append({
                    'prompt': prompt,
                    'teacher_res': teacher_cot,
                    'student_res': student_cot,
                    'type': 'cot',
                })

        if mismatch_count > 0:
            print(f"Warning: Skipped {mismatch_count} incomplete lines")

        if MAX_SAMPLES is not None and len(self.samples) > MAX_SAMPLES:
            self.samples = self.samples[:MAX_SAMPLES]
            print(f"Truncated to {MAX_SAMPLES} samples for quick test")

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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return messages

    def __getitem__(self, idx):
        try:
            return self._get_item_impl(idx)
        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            return self._get_item_impl((idx + 1) % len(self.samples))

    def _get_item_impl(self, idx):
        item = self.samples[idx]

        teacher_messages = self._build_messages(item, 'teacher')
        student_messages = self._build_messages(item, 'student')

        teacher_text = self.tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False
        )
        teacher_enc = self.tokenizer(
            teacher_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        student_text = self.tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=False
        )
        student_enc = self.tokenizer(
            student_text,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        return {
            "teacher_input_ids": teacher_enc['input_ids'].squeeze(0),
            "teacher_attention_mask": teacher_enc['attention_mask'].squeeze(0),
            "student_input_ids": student_enc['input_ids'].squeeze(0),
            "student_attention_mask": student_enc['attention_mask'].squeeze(0),
        }


def _left_pad_1d(tensor, target_len, pad_value):
    pad_size = target_len - tensor.size(0)
    if pad_size <= 0:
        return tensor[:target_len]
    return F.pad(tensor, (pad_size, 0), value=pad_value)


def make_collate_fn(pad_token_id):
    """
    动态 padding 的 collate 函数。
    所有 teacher 和 student 序列统一 left-pad 到 batch 内的最大长度。
    """
    def collate_fn(batch):
        max_len = 0
        for b in batch:
            max_len = max(max_len,
                         b['teacher_input_ids'].size(0),
                         b['student_input_ids'].size(0))

        result = {
            'teacher_input_ids': torch.stack([
                _left_pad_1d(b['teacher_input_ids'], max_len, pad_token_id) for b in batch
            ]),
            'teacher_attention_mask': torch.stack([
                _left_pad_1d(b['teacher_attention_mask'], max_len, 0) for b in batch
            ]),
            'student_input_ids': torch.stack([
                _left_pad_1d(b['student_input_ids'], max_len, pad_token_id) for b in batch
            ]),
            'student_attention_mask': torch.stack([
                _left_pad_1d(b['student_attention_mask'], max_len, 0) for b in batch
            ]),
        }
        return result

    return collate_fn


def train():
    set_seed(SEED)

    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    accelerator.print(f"Loading MoE model as TokenClassification from {MOE_MODEL_PATH}...")
    with accelerator.main_process_first():
        model = load_moe_for_token_classification(
            MOE_MODEL_PATH,
            num_labels=1,
            gradient_checkpointing=GRADIENT_CHECKPOINTING,
        )

    accelerator.print(f"Model type: {type(model).__name__}")
    accelerator.print(f"Model config: num_labels={model.config.num_labels}, "
                      f"num_experts={model.config.num_experts}")

    dataset = PairwiseTextDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)
    collate_fn = make_collate_fn(pad_token_id)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    num_training_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    is_deepspeed = accelerator.distributed_type.value == "DEEPSPEED"

    accelerator.print("=" * 60)
    accelerator.print("Starting Text MoE Value Head Warmup Training (Optimized)")
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
    accelerator.print(f"  Gradient checkpointing: {GRADIENT_CHECKPOINTING}")
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

                unwrapped = accelerator.unwrap_model(model)

                base_outputs = unwrapped.model(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                    output_router_logits=True,
                )

                sequence_output = unwrapped.dropout(base_outputs.last_hidden_state)
                logits = unwrapped.score(sequence_output)

                teacher_logits = logits[:batch_size]
                student_logits = logits[batch_size:]

                teacher_scores = get_last_token_scores(
                    teacher_logits, batch['teacher_attention_mask']
                )
                student_scores = get_last_token_scores(
                    student_logits, batch['student_attention_mask']
                )

                diff = teacher_scores - student_scores
                pairwise_loss = -F.logsigmoid(diff).mean()

                aux_loss = None
                if hasattr(base_outputs, 'router_logits') and base_outputs.router_logits is not None:
                    aux_loss = load_balancing_loss_func(
                        base_outputs.router_logits,
                        unwrapped.config.num_experts,
                        unwrapped.config.num_experts_per_tok,
                        combined_attention_mask,
                    )

                loss = pairwise_loss
                if aux_loss is not None:
                    router_aux_loss_coef = getattr(unwrapped.config, 'router_aux_loss_coef', 0.01)
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        accelerator.print(f"\nModel saved to {OUTPUT_DIR}")
        accelerator.print("=" * 60)
        accelerator.print("You can now load it in verl with:")
        accelerator.print(f"  critic.model.path: {OUTPUT_DIR}")
        accelerator.print("")
        accelerator.print("Or directly in Python:")
        accelerator.print("  from transformers import AutoModelForTokenClassification")
        accelerator.print(f"  model = AutoModelForTokenClassification.from_pretrained('{OUTPUT_DIR}')")
        accelerator.print("=" * 60)


if __name__ == "__main__":
    train()
