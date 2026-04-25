"""
验证 v_head 是否正确加载，并测试 train vs eval 模式的差异
"""

import torch
import json
import os
import random
from safetensors.torch import load_file

MODEL_PATH = "/training-data/sudongwang/model/Qwen3-VL-2B-MoE-4x-warmup-gemini-distill"
DATA_PATH = "/training-data/sudongwang/prm_workspace/dataset_process/data_pipeline/teacher_student_merged.jsonl"

# 配置
NUM_SAMPLES = 200
RANDOM_SAMPLE = True  # 设为 False 使用顺序采样
MAX_LENGTH = 8192
SEED = 42


def main():
    random.seed(SEED)
    
    print("=" * 60)
    print("验证 v_head 加载 + train/eval 模式对比")
    print("=" * 60)
    
    # 1. 从 checkpoint 读取 v_head 权重
    index_path = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    v_head_keys = [k for k in weight_map if k.startswith('v_head.')]
    print(f"\n1. Checkpoint 中的 v_head 权重: {v_head_keys}")
    
    # 加载 v_head 权重
    v_head_checkpoint = {}
    for key in v_head_keys:
        shard_file = weight_map[key]
        shard_path = os.path.join(MODEL_PATH, shard_file)
        shard_dict = load_file(shard_path)
        if key in shard_dict:
            v_head_checkpoint[key] = shard_dict[key]
            print(f"   {key}: shape={shard_dict[key].shape}, mean={shard_dict[key].float().mean():.6f}")
    
    # 2. 加载模型并检查 v_head
    print("\n2. 加载模型...")
    from value_head import Qwen3VLMoeValueHead
    model = Qwen3VLMoeValueHead.from_pretrained(MODEL_PATH)
    
    # 检查加载后的 v_head 权重
    print("\n3. 加载后的 v_head 权重:")
    for name, param in model.v_head.named_parameters():
        print(f"   v_head.{name}: shape={param.shape}, mean={param.float().mean():.6f}")
    
    # 4. 比较权重
    print("\n4. 权重比较:")
    loaded_weight = model.v_head[1].weight
    checkpoint_weight = v_head_checkpoint.get('v_head.1.weight')
    
    if checkpoint_weight is not None:
        checkpoint_weight = checkpoint_weight.to(loaded_weight.device)
        diff = (loaded_weight - checkpoint_weight).abs().max().item()
        print(f"   最大差异: {diff}")
        if diff < 1e-6:
            print("   ✅ v_head 权重加载正确!")
        else:
            print("   ❌ v_head 权重不匹配!")
    
    # 5. 测试 train vs eval 模式
    print("\n5. 测试 train vs eval 模式差异...")
    from transformers import AutoProcessor
    from value_head import get_last_token_scores
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=False)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # 加载所有样本
    all_samples = []
    with open(DATA_PATH, 'r') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))
    
    # 采样
    if RANDOM_SAMPLE:
        samples = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))
        print(f"   随机采样 {len(samples)} 个样本")
    else:
        samples = all_samples[:NUM_SAMPLES]
        print(f"   顺序取前 {len(samples)} 个样本")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n   测试 {len(samples)} 个样本...")
    
    train_correct = 0
    eval_correct = 0
    total = 0
    
    for i, item in enumerate(samples):
        prompt = item.get('prompt', item.get('question', ''))
        
        # 适配两种数据格式
        teacher_resp = item.get('teacher_response', '')
        if not teacher_resp:
            teacher_caption = item.get('teacher_caption', '')
            teacher_cot = item.get('teacher_cot', '')
            teacher_resp = f"{teacher_caption}\n{teacher_cot}" if teacher_caption or teacher_cot else ''
        
        student_resp = item.get('student_response', '')
        if not student_resp:
            student_caption = item.get('student_caption', '')
            student_cot = item.get('student_cot', '')
            student_resp = f"{student_caption}\n{student_cot}" if student_caption or student_cot else ''
        
        if not teacher_resp or not student_resp:
            continue
        
        total += 1
        
        # 构建输入（纯文本格式）
        teacher_text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": teacher_resp}],
            tokenize=False, add_generation_prompt=False
        )
        student_text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": student_resp}],
            tokenize=False, add_generation_prompt=False
        )
        
        teacher_inputs = processor(text=[teacher_text], padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
        student_inputs = processor(text=[student_text], padding='max_length', max_length=MAX_LENGTH, truncation=True, return_tensors='pt')
        
        teacher_batch = {k: v.to(device) for k, v in teacher_inputs.items() if isinstance(v, torch.Tensor)}
        student_batch = {k: v.to(device) for k, v in student_inputs.items() if isinstance(v, torch.Tensor)}
        
        # Train 模式
        model.train()
        with torch.no_grad():
            t_out = model(input_ids=teacher_batch['input_ids'], attention_mask=teacher_batch['attention_mask'])
            s_out = model(input_ids=student_batch['input_ids'], attention_mask=student_batch['attention_mask'])
            t_score_train = get_last_token_scores(t_out['logits'], teacher_batch['attention_mask']).item()
            s_score_train = get_last_token_scores(s_out['logits'], student_batch['attention_mask']).item()
        
        # Eval 模式
        model.eval()
        with torch.no_grad():
            t_out = model(input_ids=teacher_batch['input_ids'], attention_mask=teacher_batch['attention_mask'])
            s_out = model(input_ids=student_batch['input_ids'], attention_mask=student_batch['attention_mask'])
            t_score_eval = get_last_token_scores(t_out['logits'], teacher_batch['attention_mask']).item()
            s_score_eval = get_last_token_scores(s_out['logits'], student_batch['attention_mask']).item()
        
        if t_score_train > s_score_train:
            train_correct += 1
        if t_score_eval > s_score_eval:
            eval_correct += 1
        
        # 只打印前10个和每20个样本
        if i < 10 or i % 20 == 0:
            print(f"   [{i}] Train: T={t_score_train:+.4f} S={s_score_train:+.4f} | Eval: T={t_score_eval:+.4f} S={s_score_eval:+.4f}")
    
    print(f"\n   Train 模式准确率: {train_correct}/{total}")
    print(f"   Eval 模式准确率: {eval_correct}/{total}")


if __name__ == "__main__":
    main()
