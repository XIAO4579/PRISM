"""
将 value_head.safetensors 转换为 value_head.pt 格式
"""

import os
import torch
from safetensors.torch import load_file

MODEL_PATH = ""

def convert_safetensors_to_pt(model_path):
    safetensors_path = os.path.join(model_path, "value_head.safetensors")
    pt_path = os.path.join(model_path, "value_head.pt")
    
    if not os.path.exists(safetensors_path):
        print(f"❌ {safetensors_path} 不存在")
        return False
    
    print(f"📦 加载 {safetensors_path}...")
    state_dict = load_file(safetensors_path)
    
    print(f"   Keys: {list(state_dict.keys())}")
    
    print(f"💾 保存到 {pt_path}...")
    torch.save(state_dict, pt_path)
    
    print(f"✅ 转换完成!")
    print(f"   - 原文件: {safetensors_path}")
    print(f"   - 新文件: {pt_path}")
    
    # 可选：删除原文件
    # os.remove(safetensors_path)
    # print(f"🗑️ 已删除原文件")
    
    return True

if __name__ == "__main__":
    convert_safetensors_to_pt(MODEL_PATH)

