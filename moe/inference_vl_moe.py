"""
Qwen3-VL MoE Value Head 推理脚本

测试训练好的模型在随机样本上的准确率
准确率定义：teacher 分数 > student 分数 的比例
"""

import torch
from transformers import AutoProcessor
import json
import os
import random
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import argparse

# 导入公共模块
from value_head import Qwen3VLMoeValueHead, get_last_token_scores


# --- 配置 ---
MODEL_PATH = "/path/to/models/Qwen3-VL-2B-4X-Moe-warmup"  # 训练好的模型
DATA_PATH = "/path/to/datasets/warmup_pairwise.jsonl"
IMAGE_BASE_PATH = ""  # 图片基础路径
NUM_SAMPLES = 800  # 随机采样数量
MAX_LENGTH = 8192
SEED = 42

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


def load_image(image_source):
    """加载图像"""
    if image_source.startswith(('http://', 'https://')):
        response = requests.get(image_source, timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_source).convert('RGB')
    return image


def process_sample(item, processor, image_base_path, max_length):
    """
    处理单个样本，返回 teacher 和 student 的输入
    
    Args:
        item: 样本数据，包含 prompt, image, teacher_res, student_res
    """
    prompt = item.get('prompt', '')
    teacher_res = item.get('teacher_res', '')
    student_res = item.get('student_res', '')
    image_source = item.get('image', None)
    
    def build_messages(response, include_image=True):
        if image_source and image_base_path and not image_source.startswith(('http://', 'https://', '/')):
            full_image_path = os.path.join(image_base_path, image_source)
        else:
            full_image_path = image_source
        
        user_content = []
        if include_image and full_image_path:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt})
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response}
        ]
        return messages, full_image_path if (include_image and full_image_path) else None
    
    # TODO: 临时禁用图片加载进行调试
    DISABLE_IMAGE = False  # 设为 False 恢复图片加载
    
    # Teacher
    teacher_messages, img_path = build_messages(teacher_res, include_image=not DISABLE_IMAGE)
    student_messages, _ = build_messages(student_res, include_image=not DISABLE_IMAGE)
    image_source = img_path  # 更新为完整路径
    
    # 加载图像
    image = None
    if image_source:
        try:
            image = load_image(image_source)
        except Exception as e:
            print(f"Warning: Failed to load image {image_source}: {e}")
            # 重新构建消息，不包含图片（使用实际的 teacher_res 和 student_res）
            teacher_messages, _ = build_messages(teacher_res, include_image=False)
            student_messages, _ = build_messages(student_res, include_image=False)
    
    # 处理 Teacher
    teacher_text = processor.apply_chat_template(
        teacher_messages, tokenize=False, add_generation_prompt=False
    )
    teacher_inputs = processor(
        text=[teacher_text],
        images=[image] if image else None,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt',
        max_pixels=1280 * 28 * 28,
    )
    
    # 处理 Student
    student_text = processor.apply_chat_template(
        student_messages, tokenize=False, add_generation_prompt=False
    )
    student_inputs = processor(
        text=[student_text],
        images=[image] if image else None,
        padding='max_length',
        max_length=max_length,
        truncation=True,
        return_tensors='pt',
        max_pixels=1280 * 28 * 28,
    )
    
    return teacher_inputs, student_inputs


def main():
    parser = argparse.ArgumentParser(description='Qwen3-VL MoE Value Head Inference')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Model path')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Data path')
    parser.add_argument('--num_samples', type=int, default=NUM_SAMPLES, help='Number of samples to test')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"\n📦 Loading model from {args.model_path}...")
    model = Qwen3VLMoeValueHead.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    print("✅ Model loaded")
    
    # 加载处理器
    print(f"\n📦 Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    processor.tokenizer.padding_side = "left"  # 与训练保持一致
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print("✅ Processor loaded")
    
    # 加载数据 - 分开处理 caption 和 cot
    print(f"\n📦 Loading data from {args.data_path}...")
    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
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
                        samples.append({
                            'prompt': prompt,
                            'image': image,
                            'teacher_res': teacher_response,
                            'student_res': student_response,
                            'type': 'response',
                        })
                    continue
                
                # Caption 对比
                if prompt and teacher_caption and student_caption:
                    samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_caption,
                        'student_res': student_caption,
                        'type': 'caption',
                    })
                
                # CoT 对比
                if prompt and teacher_cot and student_cot:
                    samples.append({
                        'prompt': prompt,
                        'image': image,
                        'teacher_res': teacher_cot,
                        'student_res': student_cot,
                        'type': 'cot',
                    })
    
    # 按类型分组样本
    samples_by_type = {}
    for s in samples:
        t = s.get('type', 'unknown')
        if t not in samples_by_type:
            samples_by_type[t] = []
        samples_by_type[t].append(s)
    
    type_counts = {t: len(v) for t, v in samples_by_type.items()}
    print(f"✅ Loaded {len(samples)} comparison pairs: {type_counts}")
    
    # 分类型均衡采样：每个类型采样相同数量
    num_types = len(samples_by_type)
    samples_per_type = args.num_samples // num_types if num_types > 0 else args.num_samples
    
    balanced_samples = []
    for t, type_samples in samples_by_type.items():
        # 按顺序取前 N 个（调试用）
        # TODO: 改回随机采样时使用 random.sample
        selected = type_samples[:samples_per_type]
        # selected = random.sample(type_samples, min(samples_per_type, len(type_samples)))
        balanced_samples.extend(selected)
        print(f"  📊 {t}: selected {len(selected)}/{len(type_samples)} samples")
    
    samples = balanced_samples
    print(f"📊 Total balanced samples: {len(samples)} ({samples_per_type} per type)")
    
    # 推理
    print(f"\n🔍 Running inference...")
    correct = 0
    total = 0
    results = []
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(samples, desc="Inference")):
            try:
                teacher_inputs, student_inputs = process_sample(
                    item, processor, IMAGE_BASE_PATH, MAX_LENGTH
                )
            except Exception as e:
                print(f"⚠️ Skip sample {i}: {e}")
                continue
            
            # 移动到 GPU
            teacher_batch = {k: v.to(device) for k, v in teacher_inputs.items() if isinstance(v, torch.Tensor)}
            student_batch = {k: v.to(device) for k, v in student_inputs.items() if isinstance(v, torch.Tensor)}
            
            # Teacher forward
            teacher_outputs = model(
                input_ids=teacher_batch.get('input_ids'),
                attention_mask=teacher_batch.get('attention_mask'),
                pixel_values=teacher_batch.get('pixel_values'),
                image_grid_thw=teacher_batch.get('image_grid_thw'),
            )
            debug_mode = (i < 3)  # 前3个样本调试
            if debug_mode:
                print(f"\n[Sample {i}] Teacher:")
            teacher_score = get_last_token_scores(
                teacher_outputs['logits'], teacher_batch['attention_mask'], debug=debug_mode
            ).squeeze(-1).item()
            
            # Student forward
            student_outputs = model(
                input_ids=student_batch.get('input_ids'),
                attention_mask=student_batch.get('attention_mask'),
                pixel_values=student_batch.get('pixel_values'),
                image_grid_thw=student_batch.get('image_grid_thw'),
            )
            if debug_mode:
                print(f"[Sample {i}] Student:")
            student_score = get_last_token_scores(
                student_outputs['logits'], student_batch['attention_mask'], debug=debug_mode
            ).squeeze(-1).item()
            
            # 比较
            is_correct = teacher_score > student_score
            if is_correct:
                correct += 1
            total += 1
            
            diff = teacher_score - student_score
            results.append({
                'idx': i,
                'teacher_score': teacher_score,
                'student_score': student_score,
                'diff': diff,
                'correct': is_correct,
                'type': item.get('type', 'unknown'),
            })
            
            # 打印一些样本
            if i < 10 or i % 20 == 0:
                status = "✅" if is_correct else "❌"
                print(f"  [{i}] Teacher: {teacher_score:+.4f}, Student: {student_score:+.4f}, Diff: {diff:+.4f} {status}")
    
    # 统计结果
    print("\n" + "=" * 60)
    print("📊 Inference Results")
    print("=" * 60)
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"  Total samples: {total}")
    print(f"  Correct (teacher > student): {correct}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # 分类型统计准确率
    type_stats = {}
    for r in results:
        t = r.get('type', 'unknown')
        if t not in type_stats:
            type_stats[t] = {'correct': 0, 'total': 0}
        type_stats[t]['total'] += 1
        if r['correct']:
            type_stats[t]['correct'] += 1
    
    print(f"\n  📈 Accuracy by type:")
    for t, stats in sorted(type_stats.items()):
        t_acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"    {t}: {stats['correct']}/{stats['total']} = {t_acc:.2f}%")
    
    # 分数分布统计
    teacher_scores = [r['teacher_score'] for r in results]
    student_scores = [r['student_score'] for r in results]
    diffs = [r['diff'] for r in results]
    
    print(f"\n  Teacher scores: min={min(teacher_scores):.4f}, max={max(teacher_scores):.4f}, avg={sum(teacher_scores)/len(teacher_scores):.4f}")
    print(f"  Student scores: min={min(student_scores):.4f}, max={max(student_scores):.4f}, avg={sum(student_scores)/len(student_scores):.4f}")
    print(f"  Diff (T-S): min={min(diffs):.4f}, max={max(diffs):.4f}, avg={sum(diffs)/len(diffs):.4f}")
    
    # 按 diff 排序，显示最错误和最正确的样本
    sorted_results = sorted(results, key=lambda x: x['diff'])
    
    print(f"\n  🔻 Top 5 most wrong (student >> teacher):")
    for r in sorted_results[:5]:
        print(f"    [{r['idx']}] Teacher: {r['teacher_score']:+.4f}, Student: {r['student_score']:+.4f}, Diff: {r['diff']:+.4f}")
    
    print(f"\n  🔺 Top 5 most correct (teacher >> student):")
    for r in sorted_results[-5:]:
        print(f"    [{r['idx']}] Teacher: {r['teacher_score']:+.4f}, Student: {r['student_score']:+.4f}, Diff: {r['diff']:+.4f}")
    
    print("\n" + "=" * 60)
    
    return accuracy


if __name__ == "__main__":
    main()
