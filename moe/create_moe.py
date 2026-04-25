import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM


def upcycle_qwen3_to_moe(
    dense_model_path: str,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    noise_std: float = 0.01,
    norm_topk_prob: bool = True,
):
    """
    将 Qwen3 dense 模型转换为 MoE 模型 (Sparse Upcycling)
    
    Args:
        dense_model_path: Qwen3 dense 模型路径 (如 Qwen3-0.6B)
        num_experts: 专家数量
        num_experts_per_tok: 每个 token 激活的专家数 (top-k)
        noise_std: 加到专家权重上的噪声标准差
        norm_topk_prob: 是否归一化 top-k 概率
    """
    print(f"Loading dense model from: {dense_model_path}")
    
    # 1. 加载 dense 模型和配置
    dense_config = AutoConfig.from_pretrained(dense_model_path, trust_remote_code=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        dense_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 打印 dense 配置用于调试
    print(f"Dense config: hidden_size={dense_config.hidden_size}, "
          f"intermediate_size={dense_config.intermediate_size}, "
          f"num_attention_heads={dense_config.num_attention_heads}, "
          f"num_key_value_heads={dense_config.num_key_value_heads}, "
          f"head_dim={getattr(dense_config, 'head_dim', 'N/A')}")
    
    # 2. 创建 MoE 配置 (继承 dense 配置的参数)
    # 获取所有可能需要的参数
    moe_config_kwargs = {
        # 从 dense 模型继承的核心参数
        "vocab_size": dense_config.vocab_size,
        "hidden_size": dense_config.hidden_size,
        "intermediate_size": dense_config.intermediate_size,
        "num_hidden_layers": dense_config.num_hidden_layers,
        "num_attention_heads": dense_config.num_attention_heads,
        "num_key_value_heads": dense_config.num_key_value_heads,
        "hidden_act": dense_config.hidden_act,
        "max_position_embeddings": dense_config.max_position_embeddings,
        "rms_norm_eps": dense_config.rms_norm_eps,
        "rope_theta": dense_config.rope_theta,
        "attention_bias": getattr(dense_config, 'attention_bias', False),
        "tie_word_embeddings": dense_config.tie_word_embeddings,
        
        # MoE 特有参数
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "moe_intermediate_size": dense_config.intermediate_size,
        "decoder_sparse_step": 1,
        "norm_topk_prob": norm_topk_prob,
        "output_router_logits": True,
        "router_aux_loss_coef": 0.01,
        "mlp_only_layers": [],
    }
    
    # 复制其他可能存在的参数
    optional_params = [
        'rope_scaling', 'use_sliding_window', 'sliding_window', 
        'attention_dropout', 'initializer_range'
    ]
    for param in optional_params:
        if hasattr(dense_config, param):
            moe_config_kwargs[param] = getattr(dense_config, param)
    
    moe_config = Qwen3MoeConfig(**moe_config_kwargs)
    
    # 重要：手动设置 head_dim，因为 Qwen3MoeConfig 没有这个参数
    # Qwen3 dense 模型使用 head_dim=128，但 Qwen3MoeConfig 默认计算为 hidden_size/num_attention_heads
    if hasattr(dense_config, 'head_dim'):
        moe_config.head_dim = dense_config.head_dim
        print(f"Setting head_dim={moe_config.head_dim} from dense config")
    
    print(f"Creating MoE model with {num_experts} experts, top-{num_experts_per_tok}")
    
    # 3. 创建空的 MoE 模型
    moe_model = Qwen3MoeForCausalLM(moe_config)
    moe_model = moe_model.to(torch.bfloat16)
    
    # 4. 复制权重
    print("Copying weights from dense model to MoE model...")
    
    dense_state_dict = dense_model.state_dict()
    moe_state_dict = moe_model.state_dict()
    
    with torch.no_grad():
        for name, param in moe_state_dict.items():
            # 处理专家权重 (mlp.experts.X.{gate_proj,up_proj,down_proj})
            if ".mlp.experts." in name:
                # 从 "model.layers.0.mlp.experts.3.gate_proj.weight" 
                # 提取 "model.layers.0.mlp.gate_proj.weight"
                parts = name.split(".")
                expert_idx = int(parts[parts.index("experts") + 1])
                
                # 构建 dense 模型中对应的权重名
                dense_name = name.replace(f".experts.{expert_idx}", "")
                
                if dense_name in dense_state_dict:
                    src_tensor = dense_state_dict[dense_name]
                    if param.shape != src_tensor.shape:
                        print(f"  Shape mismatch for expert {name}: MoE={param.shape}, dense={src_tensor.shape}")
                        continue
                    # 复制 dense 权重 + 加噪声
                    param.copy_(src_tensor)
                    param.add_(torch.randn_like(param) * noise_std)
                else:
                    print(f"  Warning: {dense_name} not found in dense model for expert {name}")
                    
            # 处理 router (gate) 权重 - 随机初始化
            elif ".mlp.gate.weight" in name:
                # Router 保持随机初始化，用小值初始化
                nn.init.normal_(param, mean=0.0, std=0.02)
                
            # 其他权重直接复制 (embeddings, attention, norm, lm_head)
            else:
                # 尝试从 dense 模型找对应权重
                if name in dense_state_dict:
                    src_tensor = dense_state_dict[name]
                    if param.shape != src_tensor.shape:
                        print(f"  Shape mismatch for {name}: MoE={param.shape}, dense={src_tensor.shape}")
                        continue
                    param.copy_(src_tensor)
                else:
                    print(f"  Warning: {name} not found in dense model, keeping random init")
    
    print("Sparse Upcycling complete!")
    
    # 清理 dense 模型
    del dense_model
    del dense_state_dict
    torch.cuda.empty_cache()
    
    return moe_model, moe_config


# ============ 使用示例 ============
if __name__ == "__main__":
    model_path = "/data/user/swang886/gad_project/models/Qwen3-1.7B"
    
    # 1. Sparse Upcycling 创建 MoE 模型
    moe_model, moe_config = upcycle_qwen3_to_moe(
        dense_model_path=model_path,
        num_experts=4,
        num_experts_per_tok=2,
        noise_std=0.01,
        norm_topk_prob=True,
    )
    
    moe_model.cuda()
    moe_model.train()
    
    # 2. 测试 Forward
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    
    # 使用 transformers 标准接口
    outputs = moe_model(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_router_logits=True,  # 输出 router logits 用于 aux loss
    )
    
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Aux loss: {outputs.aux_loss}")
    
    # 3. 如果需要做判别器，可以添加 score_head
    class Qwen3MoEDiscriminator(nn.Module):
        def __init__(self, moe_model):
            super().__init__()
            self.moe_model = moe_model
            self.score_head = nn.Linear(
                moe_model.config.hidden_size, 1, 
                bias=False, dtype=torch.bfloat16
            )
        
        def forward(self, input_ids, attention_mask):
            outputs = self.moe_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_router_logits=True,
            )
            # 取最后一个 token 的 hidden state
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            scores = self.score_head(last_hidden)
            return scores, outputs.aux_loss
    
    discriminator = Qwen3MoEDiscriminator(moe_model).cuda()
    scores, aux_loss = discriminator(inputs.input_ids, inputs.attention_mask)
    print(f"Scores: {scores}")
    print(f"Aux loss: {aux_loss}")
    
    # 4. 保存模型 (可以用 transformers 标准方式保存和加载)
    save_path = "/data/user/swang886/gad_project/models/Qwen3-1.7B-MoE-4x"
    print(f"Saving MoE model to: {save_path}")
    
    # 保存模型权重和配置
    moe_model.save_pretrained(save_path)
    # 保存 tokenizer (复用原始的)
    tokenizer.save_pretrained(save_path)
    
    print(f"Model saved! You can now load it with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{save_path}')")
    
    # 5. 验证加载
    print("\n--- Verifying saved model ---")
    del moe_model
    
    
    # 重新加载
    loaded_model = AutoModelForCausalLM.from_pretrained(
        save_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,  # 不需要 trust_remote_code，因为是 transformers 原生支持
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained(save_path)
    
    print(f"Loaded model type: {type(loaded_model).__name__}")
    print(f"Model config: num_experts={loaded_model.config.num_experts}, "
          f"num_experts_per_tok={loaded_model.config.num_experts_per_tok}")
    
    # 测试推理
    loaded_model.cuda()
    test_inputs = loaded_tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    with torch.no_grad():
        test_outputs = loaded_model(**test_inputs, output_router_logits=True)
    print(f"Test output logits shape: {test_outputs.logits.shape}")
    print(f"Test aux_loss: {test_outputs.aux_loss}")
    print("\n✅ Model saved and loaded successfully!")