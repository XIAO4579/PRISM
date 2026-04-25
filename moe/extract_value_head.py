"""
从已保存的 checkpoint 中提取 value_head.safetensors

用于已训练完成但没有单独保存 value_head 的模型
"""

import os
import json
import argparse
from safetensors.torch import load_file, save_file


def extract_value_head(model_path: str):
    """
    从分片 checkpoint 中提取 v_head 权重并保存为 value_head.safetensors
    
    Args:
        model_path: 模型目录路径
    """
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    
    if not os.path.exists(index_path):
        print(f"❌ Index file not found: {index_path}")
        return False
    
    # 读取 index 文件
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    
    # 找到 v_head 相关的权重
    v_head_keys = [k for k in weight_map.keys() if k.startswith('v_head.')]
    
    if not v_head_keys:
        print(f"❌ No v_head weights found in checkpoint")
        return False
    
    print(f"📦 Found v_head weights: {v_head_keys}")
    
    # 找到包含 v_head 的分片文件
    shard_files = set(weight_map[k] for k in v_head_keys)
    
    # 从分片文件中加载 v_head 权重
    v_head_state = {}
    for shard_file in shard_files:
        shard_path = os.path.join(model_path, shard_file)
        print(f"📖 Loading from {shard_file}...")
        shard_dict = load_file(shard_path)
        
        for k, v in shard_dict.items():
            if k.startswith("v_head."):
                v_head_state[k] = v
                print(f"   - {k}: {v.shape}")
    
    # 转换为 trl 格式
    v_head_trl_state = {}
    for k, v in v_head_state.items():
        # v_head.1.weight -> v_head.summary.weight
        if k == "v_head.1.weight":
            v_head_trl_state["v_head.summary.weight"] = v.contiguous()
        elif k == "v_head.1.bias":
            v_head_trl_state["v_head.summary.bias"] = v.contiguous()
        else:
            # 保留原始格式
            v_head_trl_state[k] = v.contiguous()
    
    # 保存
    output_path = os.path.join(model_path, "value_head.safetensors")
    save_file(v_head_trl_state, output_path)
    
    print(f"\n✅ Saved value_head.safetensors to {output_path}")
    print(f"   Keys: {list(v_head_trl_state.keys())}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Extract value_head from checkpoint')
    parser.add_argument('model_path', type=str, help='Path to model directory')
    args = parser.parse_args()
    
    extract_value_head(args.model_path)


if __name__ == "__main__":
    main()
