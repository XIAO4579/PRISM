"""
Qwen2-VL MoE Discriminator Warmup Training Script

使用保存的 Qwen2-VL-MoE 模型进行 pairwise warmup 训练，
训练完成后可以作为 verl 中的 critic model (value head)

关键：使用原生 transformers，自己实现 Value Head 包装类
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
import json
import os
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

# 导入公共模块 - 使用 Qwen2VLMoeValueHead
from value_head import Qwen2VLMoeValueHead, get_last_token_scores


# --- 配置 ---
MOE_MODEL_PATH = "/path/to/models/Qwen2-VL-2B-MoE-4x"  # Qwen2-VL MoE 模型
PROCESSOR_PATH = "/path/to/models/Qwen2-VL-2B-MoE-4x"
DATA_PATH = "/path/to/datasets/teacher_student_merged_fixed.jsonl"
OUTPUT_DIR = "/path/to/models/Qwen2-VL-2B-MoE-4x-warmup-distill"
IMAGE_BASE_PATH = ""  # 图片基础路径，用于拼接相对路径

MAX_LENGTH = 4096
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.1
SEED = 42
GRADIENT_CHECKPOINTING = True  # 启用 gradient checkpointing 节省显存


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
        print(f"Resized image to {new_w}x{new_h}")
    
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
    def __init__(self, data_path, processor, max_length=4096, image_base_path=None, max_pixels=560*560):
        self.samples = []
        self.processor = processor
        self.max_length = max_length
        self.image_base_path = image_base_path  # 图片基础路径

        # --- ✅ 修正部分开始 ---
        # 针对 Qwen2-VL，直接修改属性，不要去动 'size' 字典
        if hasattr(processor, "image_processor"):
            # 1. 强制设定最大像素数 (限制显存关键)
            processor.image_processor.max_pixels = max_pixels
            
            # 2. 设定最小像素数 (防止图片太小导致 patch 切分报错)
            # Qwen2-VL 默认通常是 3136 (56*56) 或 256*28*28，根据你的模型需求设定
            # 如果不确定，可以维持原样，或者设为一个安全值，如 28*28
            if not hasattr(processor.image_processor, "min_pixels"):
                processor.image_processor.min_pixels = 28 * 28
            
            print(f"Force updated image_processor: max_pixels={processor.image_processor.max_pixels}, min_pixels={processor.image_processor.min_pixels}")
        # --- ✅ 修正部分结束 ---
        
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
                # 纯文本样本，包含 caption 和 cot
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
            item: 样本数据，包含 prompt, image, teacher_res, student_res
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
        
        # 如果有图像，添加图像
        if image_source:
            user_content.append({"type": "image"})
        
        # 添加文本
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        
        return messages, image_source
    
    def __getitem__(self, idx):
        # 使用 try-except 处理可能的错误，出错时尝试下一个样本
        try:
            return self._get_item_impl(idx)
        except Exception as e:
            print(f"Warning: Failed to process sample {idx}: {e}")
            # 尝试下一个样本（循环避免越界）
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
                # 如果图像加载失败，移除图像引用
                teacher_messages[0]["content"] = [c for c in teacher_messages[0]["content"] if c.get("type") != "image"]
                student_messages[0]["content"] = [c for c in student_messages[0]["content"] if c.get("type") != "image"]
        
        # 处理 Teacher 输入
        teacher_text = self.processor.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=False
        )
        # 限制图片大小，避免 image token 过多导致截断错误
        # max_pixels 限制图片最大像素数，减少 image token 数量
        # 降低到 512*28*28 ≈ 400k 像素，图片 token 约 1000-1500
        teacher_inputs = self.processor(
            text=[teacher_text],
            images=[image] if image else None,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
            max_pixels=560 * 560,  # 限制最大像素数，约 313600 像素
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
            max_pixels=560 * 560,  # 限制最大像素数，约 313600 像素
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
            # 注意：image_grid_thw 不能 squeeze，需要保持 (num_images, 3) 的形状
            # squeeze(0) 只移除 batch 维度，保持 (num_images, 3)
            thw = teacher_inputs['image_grid_thw']
            result["teacher_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw
            thw = student_inputs['image_grid_thw']
            result["student_image_grid_thw"] = thw.squeeze(0) if thw.dim() > 2 else thw
        
        return result


def collate_fn(batch):
    """
    自定义 collate 函数，处理可变长度的视觉输入
    
    支持混合有图片和无图片的样本（虽然 batch_size=1 时不需要）
    """
    result = {}
    
    # 基础文本输入
    result['teacher_input_ids'] = torch.stack([b['teacher_input_ids'] for b in batch])
    result['teacher_attention_mask'] = torch.stack([b['teacher_attention_mask'] for b in batch])
    result['student_input_ids'] = torch.stack([b['student_input_ids'] for b in batch])
    result['student_attention_mask'] = torch.stack([b['student_attention_mask'] for b in batch])
    
    # 视觉输入（只处理有图片的样本）
    # 收集所有有 pixel_values 的样本
    teacher_pixels = [b['teacher_pixel_values'] for b in batch if 'teacher_pixel_values' in b]
    student_pixels = [b['student_pixel_values'] for b in batch if 'student_pixel_values' in b]
    
    if teacher_pixels:
        result['teacher_pixel_values'] = torch.cat(teacher_pixels, dim=0)
        result['student_pixel_values'] = torch.cat(student_pixels, dim=0)
    
    # 收集所有有 image_grid_thw 的样本
    teacher_thws = []
    student_thws = []
    for b in batch:
        if 'teacher_image_grid_thw' in b:
            t_thw = b['teacher_image_grid_thw']
            s_thw = b['student_image_grid_thw']
            # 确保是 2D
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
    
    # DDP 配置，MoE 需要 find_unused_parameters
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs]
    )
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(PROCESSOR_PATH, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 加载模型 - 使用自定义的 Qwen2VLMoeValueHead
    accelerator.print(f"Loading Qwen2-VL MoE model with Value Head from {MOE_MODEL_PATH}...")
    with accelerator.main_process_first():
        model = Qwen2VLMoeValueHead(
            MOE_MODEL_PATH, 
            num_labels=1,
            gradient_checkpointing=GRADIENT_CHECKPOINTING
        )
    
    accelerator.print(f"Model type: {type(model).__name__}")
    text_config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    accelerator.print(f"Model config: num_labels={model.num_labels}, "
                      f"num_experts={text_config.num_experts}")
    
    # 数据集
    dataset = PairwiseVLDataset(
        DATA_PATH, 
        processor, 
        max_length=MAX_LENGTH,
        image_base_path=IMAGE_BASE_PATH
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,          # 多进程加载数据
        pin_memory=True,        # 加速 CPU->GPU 传输
        prefetch_factor=2,      # 预取数据
        persistent_workers=True # 保持 worker 进程
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
    accelerator.print("Starting Qwen2-VL MoE Value Head Warmup Training")
    accelerator.print(f"  Model: {MOE_MODEL_PATH}")
    accelerator.print(f"  Output: {OUTPUT_DIR}")
    accelerator.print(f"  Num samples: {len(dataset)}")
    accelerator.print(f"  Batch size: {BATCH_SIZE}")
    accelerator.print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    accelerator.print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes}")
    accelerator.print(f"  Num epochs: {NUM_EPOCHS}")
    accelerator.print(f"  Learning rate: {LEARNING_RATE}")
    accelerator.print(f"  Max length: {MAX_LENGTH}")
    accelerator.print("  [优化] 合并 Teacher/Student forward，单次前向传播")
    accelerator.print("=" * 60)
    
    model.train()
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        total_accuracy = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                # ============================================================
                # 合并 Teacher 和 Student 输入，一次 forward 完成
                # ============================================================
                batch_size = batch['teacher_input_ids'].shape[0]
                
                # 合并 input_ids 和 attention_mask（沿 batch 维度拼接）
                combined_input_ids = torch.cat([
                    batch['teacher_input_ids'], 
                    batch['student_input_ids']
                ], dim=0)  # [2*B, seq_len]
                
                combined_attention_mask = torch.cat([
                    batch['teacher_attention_mask'], 
                    batch['student_attention_mask']
                ], dim=0)  # [2*B, seq_len]
                
                # 准备合并后的输入
                combined_kwargs = {
                    'input_ids': combined_input_ids,
                    'attention_mask': combined_attention_mask,
                }
                
                # 合并视觉输入（如果有）
                # 注意：teacher 和 student 使用同一张图片，所以 pixel_values 相同
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
                
                # ============================================================
                # 单次 Forward（原来需要两次，现在只需要一次）
                # ============================================================
                unwrapped = accelerator.unwrap_model(model)
                combined_outputs = unwrapped(**combined_kwargs, output_router_logits=True)
                
                # 拆分 logits：前半部分是 teacher，后半部分是 student
                combined_logits = combined_outputs['logits']  # [2*B, seq_len, 1]
                teacher_logits = combined_logits[:batch_size]
                student_logits = combined_logits[batch_size:]
                
                # 分别获取 last token scores
                teacher_scores = get_last_token_scores(
                    teacher_logits, batch['teacher_attention_mask']
                )
                student_scores = get_last_token_scores(
                    student_logits, batch['student_attention_mask']
                )

                # Pairwise Loss (Bradley-Terry)
                # Loss = -log(sigmoid(r_teacher - r_student))
                diff = teacher_scores - student_scores
                pairwise_loss = -F.logsigmoid(diff).mean()
                
                # MoE Auxiliary Loss (负载均衡损失)
                aux_loss = combined_outputs['aux_loss']
                
                # 总损失 = pairwise loss + aux loss
                loss = pairwise_loss
                if aux_loss is not None:
                    text_config = unwrapped.config.text_config if hasattr(unwrapped.config, 'text_config') else unwrapped.config
                    router_aux_loss_coef = getattr(text_config, 'router_aux_loss_coef', 0.01)
                    loss = loss + router_aux_loss_coef * aux_loss
                
                # Backward
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    global_step += 1
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 统计
                total_loss += loss.item()
                
                # 计算准确率：teacher_score > student_score 的比例
                accuracy = (diff > 0).float().mean().item()
                total_accuracy += accuracy
                
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "pair": f"{pairwise_loss.item():.4f}",
                    "acc": f"{accuracy:.2%}",
                    "diff": f"{diff.mean().item():.2f}",
                }
                if aux_loss is not None:
                    postfix["aux"] = f"{aux_loss.item():.4f}"
                pbar.set_postfix(postfix)
        
        # Epoch 统计
        n_steps = len(dataloader)
        accelerator.print(f"\nEpoch {epoch+1} Summary:")
        accelerator.print(f"  Avg Loss: {total_loss / n_steps:.4f}")
        accelerator.print(f"  Avg Accuracy: {total_accuracy / n_steps:.2%}")
    
    # 保存模型
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        accelerator.print(f"\nSaving model to {OUTPUT_DIR}...")
        
        unwrapped_model = accelerator.unwrap_model(model)
        
        # 保存模型（base model + value head）
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        
        accelerator.print(f"\n✅ Model saved to {OUTPUT_DIR}")
        accelerator.print("\n" + "=" * 60)
        accelerator.print("Saved files:")
        accelerator.print("  - model-*.safetensors (base model + v_head merged)")
        accelerator.print("  - value_head.pt (trl/verl compatible)")
        accelerator.print("")
        accelerator.print("Load with Qwen2VLMoeValueHead:")
        accelerator.print("  from value_head import Qwen2VLMoeValueHead")
        accelerator.print(f"  model = Qwen2VLMoeValueHead.from_pretrained('{OUTPUT_DIR}')")
        accelerator.print("")
        accelerator.print("Load value_head for verl/trl:")
        accelerator.print("  import torch")
        accelerator.print(f"  v_head_state = torch.load('{OUTPUT_DIR}/value_head.pt')")
        accelerator.print("  # Keys: v_head.summary.weight")
        accelerator.print("=" * 60)


if __name__ == "__main__":
    train()
