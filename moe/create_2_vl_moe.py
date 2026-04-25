"""
Qwen2-VL Dense -> MoE 转换脚本 (Sparse Upcycling)

基于 transformers 官方的 Qwen2VLMoeForConditionalGeneration 实现。
将 Qwen2-VL dense 模型的 MLP 层转换为 MoE 层。

特点：
- 保留 Vision Encoder 不变
- 只对 Language Model 的 MLP 层做 MoE 转换
- 支持自定义 num_experts 和 num_experts_per_tok
- Qwen3 MoE 风格：无 shared expert，更省显存
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl_moe.configuration_qwen2_vl_moe import (
    Qwen2VLMoeConfig,
    Qwen2VLMoeTextConfig,
    Qwen2VLMoeVisionConfig,
)
from transformers.models.qwen2_vl_moe.modeling_qwen2_vl_moe import Qwen2VLMoeForConditionalGeneration


def upcycle_qwen2_vl_to_moe(
    dense_model_path: str,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    noise_std: float = 0.01,
    norm_topk_prob: bool = True,
):
    """
    将 Qwen2-VL dense 模型转换为 MoE 模型 (Sparse Upcycling)

    Args:
        dense_model_path: Qwen2-VL dense 模型路径 (如 Qwen2-VL-2B-Instruct)
        num_experts: 专家数量
        num_experts_per_tok: 每个 token 激活的专家数 (top-k)
        noise_std: 加到专家权重上的噪声标准差
        norm_topk_prob: 是否归一化 top-k 概率

    Returns:
        moe_model: 转换后的 MoE 模型
        moe_config: MoE 模型配置
    """
    print(f"Loading dense model from: {dense_model_path}")

    # 1. 加载 dense 模型和配置
    dense_config = AutoConfig.from_pretrained(dense_model_path, trust_remote_code=True)
    dense_model = AutoModelForImageTextToText.from_pretrained(
        dense_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # 打印 dense 配置用于调试
    text_cfg = dense_config.text_config
    vision_cfg = dense_config.vision_config
    print(f"Dense text config: hidden_size={text_cfg.hidden_size}, "
          f"intermediate_size={text_cfg.intermediate_size}, "
          f"num_hidden_layers={text_cfg.num_hidden_layers}")
    print(f"Dense vision config: hidden_size={vision_cfg.hidden_size}, "
          f"depth={vision_cfg.depth}")
    
    # 2. 创建 MoE 配置
    # 2.1 Vision 配置保持不变 (Qwen2-VL 风格)
    moe_vision_config = Qwen2VLMoeVisionConfig(
        depth=vision_cfg.depth,
        embed_dim=vision_cfg.embed_dim,
        hidden_size=vision_cfg.hidden_size,
        hidden_act=vision_cfg.hidden_act,
        mlp_ratio=vision_cfg.mlp_ratio,
        num_heads=vision_cfg.num_heads,
        in_channels=vision_cfg.in_channels,
        patch_size=vision_cfg.patch_size,
        spatial_merge_size=vision_cfg.spatial_merge_size,
        temporal_patch_size=vision_cfg.temporal_patch_size,
    )
    
    # 2.2 Text 配置 - 转换为 MoE (无 shared expert，更省显存)
    moe_text_config = Qwen2VLMoeTextConfig(
        vocab_size=text_cfg.vocab_size,
        hidden_size=text_cfg.hidden_size,
        intermediate_size=text_cfg.intermediate_size,  # dense MLP 的 intermediate_size
        num_hidden_layers=text_cfg.num_hidden_layers,
        num_attention_heads=text_cfg.num_attention_heads,
        num_key_value_heads=text_cfg.num_key_value_heads,
        hidden_act=text_cfg.hidden_act,
        max_position_embeddings=text_cfg.max_position_embeddings,
        rms_norm_eps=text_cfg.rms_norm_eps,
        rope_theta=text_cfg.rope_theta,
        tie_word_embeddings=text_cfg.tie_word_embeddings,
        use_sliding_window=getattr(text_cfg, 'use_sliding_window', False),
        sliding_window=getattr(text_cfg, 'sliding_window', 4096),
        max_window_layers=getattr(text_cfg, 'max_window_layers', 80),
        attention_dropout=getattr(text_cfg, 'attention_dropout', 0.0),
        
        # MoE 特有参数 (无 shared expert)
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=text_cfg.intermediate_size,  # 每个专家的 intermediate_size
        shared_expert_intermediate_size=text_cfg.intermediate_size,  # 配置保留但不使用
        decoder_sparse_step=1,  # 每层都是 MoE
        norm_topk_prob=norm_topk_prob,
        router_aux_loss_coef=0.01,
        mlp_only_layers=[],  # 没有纯 MLP 层
        
        # 可选参数
        rope_scaling=getattr(text_cfg, 'rope_scaling', None),
    )
    
    # 2.3 组合成完整的 MoE 配置
    moe_config = Qwen2VLMoeConfig(
        text_config=moe_text_config.to_dict(),
        vision_config=moe_vision_config.to_dict(),
        image_token_id=dense_config.image_token_id,
        video_token_id=dense_config.video_token_id,
        vision_start_token_id=dense_config.vision_start_token_id,
        vision_end_token_id=dense_config.vision_end_token_id,
        tie_word_embeddings=dense_config.tie_word_embeddings,
    )
    
    print(f"Creating MoE model with {num_experts} experts, top-{num_experts_per_tok}")
    
    # 3. 创建空的 MoE 模型
    moe_model = Qwen2VLMoeForConditionalGeneration(moe_config)
    moe_model = moe_model.to(torch.bfloat16)
    
    # 4. 复制权重
    print("Copying weights from dense model to MoE model...")
    
    dense_state_dict = dense_model.state_dict()
    moe_state_dict = moe_model.state_dict()
    
    copied_count = 0
    expert_count = 0
    router_count = 0
    skipped_count = 0
    
    with torch.no_grad():
        for name, param in moe_state_dict.items():
            # 处理专家权重 - 融合格式 (Qwen3 MoE style)
            # MoE 使用融合的专家权重格式：
            #   - experts.gate_up_proj: (num_experts, hidden_size, 2*intermediate_size)
            #   - experts.down_proj: (num_experts, intermediate_size, hidden_size)
            
            if ".mlp.experts.gate_up_proj" in name:
                # 构建对应的 dense 权重名
                # MoE: model.language_model.layers.0.mlp.experts.gate_up_proj
                # Dense: model.language_model.layers.0.mlp.gate_proj.weight + up_proj.weight
                base_name = name.replace(".experts.gate_up_proj", "")
                gate_name = base_name + ".gate_proj.weight"
                up_name = base_name + ".up_proj.weight"
                
                if gate_name in dense_state_dict and up_name in dense_state_dict:
                    gate_weight = dense_state_dict[gate_name]  # (intermediate, hidden)
                    up_weight = dense_state_dict[up_name]      # (intermediate, hidden)
                    
                    # 拼接 gate 和 up: (2*intermediate, hidden) -> transpose -> (hidden, 2*intermediate)
                    gate_up = torch.cat([gate_weight, up_weight], dim=0).T  # (hidden, 2*intermediate)
                    
                    # 为每个专家复制并添加噪声
                    for expert_idx in range(num_experts):
                        param[expert_idx].copy_(gate_up)
                        param[expert_idx].add_(torch.randn_like(param[expert_idx]) * noise_std)
                    
                    expert_count += 1
                    print(f"  Copied gate_up_proj for {name}: {param.shape}")
                else:
                    print(f"  Warning: {gate_name} or {up_name} not found in dense model")
                    skipped_count += 1
                    
            elif ".mlp.experts.down_proj" in name:
                # MoE: model.language_model.layers.0.mlp.experts.down_proj
                # Dense: model.language_model.layers.0.mlp.down_proj.weight
                base_name = name.replace(".experts.down_proj", "")
                dense_name = base_name + ".down_proj.weight"
                
                if dense_name in dense_state_dict:
                    down_weight = dense_state_dict[dense_name]  # (hidden, intermediate)
                    down_weight_t = down_weight.T  # (intermediate, hidden)
                    
                    # 为每个专家复制并添加噪声
                    for expert_idx in range(num_experts):
                        param[expert_idx].copy_(down_weight_t)
                        param[expert_idx].add_(torch.randn_like(param[expert_idx]) * noise_std)
                    
                    expert_count += 1
                    print(f"  Copied down_proj for {name}: {param.shape}")
                else:
                    print(f"  Warning: {dense_name} not found in dense model")
                    skipped_count += 1
                    
            # 处理 router (gate) 权重 - 随机初始化
            elif ".mlp.gate.weight" in name:
                # Router 用小值随机初始化
                nn.init.normal_(param, mean=0.0, std=0.02)
                router_count += 1
                
            # 其他权重直接复制 (embeddings, attention, norm, lm_head, vision encoder)
            else:
                if name in dense_state_dict:
                    src_tensor = dense_state_dict[name]
                    if param.shape != src_tensor.shape:
                        print(f"  Shape mismatch for {name}: MoE={param.shape}, dense={src_tensor.shape}")
                        skipped_count += 1
                        continue
                    param.copy_(src_tensor)
                    copied_count += 1
                else:
                    print(f"  Warning: {name} not found in dense model, keeping random init")
                    skipped_count += 1
    
    print(f"\nSparse Upcycling complete! (Fused experts, no shared expert)")
    print(f"  - Copied weights: {copied_count}")
    print(f"  - Expert weights (fused, with noise): {expert_count}")
    print(f"  - Router weights (random init): {router_count}")
    print(f"  - Skipped: {skipped_count}")
    
    # 清理 dense 模型
    del dense_model
    del dense_state_dict
    torch.cuda.empty_cache()
    
    return moe_model, moe_config


def test_text_only_forward(moe_model, processor):
    """
    测试纯文本输入的 forward（不包含图像）
    """
    print("\n=== Testing text-only forward ===")
    
    # 构建对话格式的输入
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # 使用 processor 处理文本
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(moe_model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = moe_model(
            **inputs,
            output_router_logits=True,
        )
    
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Aux loss: {outputs.aux_loss}")
    
    return outputs


def test_image_forward(moe_model, processor, image_path: str = None):
    """
    测试带图像输入的 forward
    
    Args:
        moe_model: MoE 模型
        processor: 处理器
        image_path: 图像路径（如果为 None，则使用随机生成的图像）
    """
    print("\n=== Testing image forward ===")
    
    from PIL import Image
    import numpy as np
    
    # 如果没有提供图像路径，创建一个随机图像
    if image_path is None:
        print("  Using randomly generated image for testing")
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    else:
        image = Image.open(image_path)
    
    # 构建带图像的对话
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see in this image?"}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(moe_model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = moe_model(
            **inputs,
            output_router_logits=True,
        )
    
    print(f"  Logits shape: {outputs.logits.shape}")
    print(f"  Aux loss: {outputs.aux_loss}")
    
    return outputs


def save_moe_model(moe_model, processor, save_path: str):
    """
    保存 MoE 模型和 processor
    
    Args:
        moe_model: MoE 模型
        processor: 处理器
        save_path: 保存路径
    """
    print(f"\n=== Saving MoE model to: {save_path} ===")
    
    # 保存模型权重和配置
    moe_model.save_pretrained(save_path)
    
    # 保存 processor
    processor.save_pretrained(save_path)
    
    print(f"Model saved! You can now load it with:")
    print(f"  model = AutoModelForImageTextToText.from_pretrained('{save_path}')")
    print(f"  processor = AutoProcessor.from_pretrained('{save_path}')")


def verify_saved_model(save_path: str):
    """
    验证保存的模型能否正确加载
    
    Args:
        save_path: 模型保存路径
    """
    print(f"\n=== Verifying saved model from: {save_path} ===")
    
    # 重新加载模型
    loaded_model = AutoModelForImageTextToText.from_pretrained(
        save_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,  # transformers 原生支持
    )
    loaded_processor = AutoProcessor.from_pretrained(save_path)
    
    print(f"  Loaded model type: {type(loaded_model).__name__}")
    print(f"  Model config: num_experts={loaded_model.config.text_config.num_experts}, "
          f"num_experts_per_tok={loaded_model.config.text_config.num_experts_per_tok}")
    
    # 简单测试
    loaded_model.cuda()
    messages = [{"role": "user", "content": "Hello!"}]
    text = loaded_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    test_inputs = loaded_processor(text=[text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        test_outputs = loaded_model(**test_inputs, output_router_logits=True)
    
    print(f"  Test output logits shape: {test_outputs.logits.shape}")
    print(f"  Test aux_loss: {test_outputs.aux_loss}")
    print("\n✅ Model saved and loaded successfully!")
    
    return loaded_model, loaded_processor


# ============ Discriminator (可选) ============
class Qwen2VLMoEDiscriminator(nn.Module):
    """
    基于 Qwen2-VL MoE 的判别器
    可用于 reward model 或其他判别任务
    """
    def __init__(self, moe_model):
        super().__init__()
        self.moe_model = moe_model
        self.score_head = nn.Linear(
            moe_model.config.text_config.hidden_size, 
            1, 
            bias=False, 
            dtype=torch.bfloat16
        )
    
    def forward(self, **inputs):
        """
        Forward pass
        
        Returns:
            scores: (batch_size, 1) 判别分数
            aux_loss: MoE 的辅助 loss
        """
        outputs = self.moe_model(
            **inputs,
            output_hidden_states=True,
            output_router_logits=True,
        )
        
        # 取最后一个 token 的 hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        scores = self.score_head(last_hidden)
        
        return scores, outputs.aux_loss


# ============ 使用示例 ============
if __name__ == "__main__":
    # 配置
    model_path = "/data/home/scwb352/run/test/model/Qwen2-VL-2B-Instruct"
    save_path = "/data/home/scwb352/run/test/model/Qwen2-VL-2B-MoE-4x"
    
    # 1. Sparse Upcycling 创建 MoE 模型
    moe_model, moe_config = upcycle_qwen2_vl_to_moe(
        dense_model_path=model_path,
        num_experts=4,
        num_experts_per_tok=2,
        noise_std=0.01,
    )
    
    moe_model.cuda()
    moe_model.eval()
    
    # 2. 加载 processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 3. 测试纯文本 forward
    test_text_only_forward(moe_model, processor)
    
    # 4. 测试带图像的 forward
    test_image_forward(moe_model, processor)
    
    # 5. 测试 Discriminator（可选）
    print("\n=== Testing Discriminator ===")
    discriminator = Qwen2VLMoEDiscriminator(moe_model).cuda()
    
    messages = [{"role": "user", "content": "Test input"}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    
    scores, aux_loss = discriminator(**inputs)
    print(f"  Scores: {scores}")
    print(f"  Aux loss: {aux_loss}")
    
    # 6. 保存模型
    save_moe_model(moe_model, processor, save_path)
    
    # 7. 验证加载
    del moe_model
    torch.cuda.empty_cache()
    
    verify_saved_model(save_path)
