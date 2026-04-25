"""
Qwen3-VL Dense Value Head 模型

在 Qwen3-VL dense backbone 上添加 Value Head (分类头)，
用于 pairwise 判别器训练。

与 MoE 版本的区别：
  - 没有 aux_loss (无 router 负载均衡)
  - 没有 output_router_logits
  - backbone 直接使用 dense model，无需 sparse upcycling

兼容 trl/verl 的 v_head 格式。
"""

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText


class Qwen3VLDenseValueHead(nn.Module):
    """
    Qwen3-VL Dense 模型 + Value Head

    使用 Qwen3VLForConditionalGeneration 作为 backbone，
    添加一个 score head 用于输出 value/reward 分数。

    结构：
        Qwen3-VL backbone → last hidden state → v_head(Dropout + Linear) → score

    兼容 trl/verl 的 v_head 格式：Sequential(Dropout, Linear)
    """

    def __init__(
        self,
        model_path: str,
        num_labels: int = 1,
        gradient_checkpointing: bool = False,
        dropout: float = 0.1,
    ):
        """
        Args:
            model_path: 预训练模型路径 (如 Qwen3-VL-8B)
            num_labels: 输出维度（通常为 1）
            gradient_checkpointing: 是否启用 gradient checkpointing 节省显存
            dropout: v_head 的 dropout 比例
        """
        super().__init__()

        # 加载配置
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # 加载 base model (Qwen3VLForConditionalGeneration)
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 启用 gradient checkpointing 以节省显存
        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for base model")

        # 获取 hidden_size
        if hasattr(self.config, 'text_config'):
            hidden_size = self.config.text_config.hidden_size
        else:
            hidden_size = self.config.hidden_size

        # Value head (score head)
        # 匹配 trl 的结构：Sequential(Dropout, Linear)
        self.num_labels = num_labels
        self.v_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels, bias=True, dtype=torch.bfloat16)
        )

        # 初始化 v_head
        self._init_v_head()

    def _init_v_head(self):
        """初始化 v_head 权重"""
        for module in self.v_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model"""
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model"""
        self.base_model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        output_hidden_states: bool = False,
        **kwargs
    ):
        """
        Forward pass

        Args:
            input_ids: 输入 token ids
            attention_mask: 注意力掩码
            pixel_values: 图像像素值
            pixel_values_videos: 视频像素值
            image_grid_thw: 图像网格信息
            video_grid_thw: 视频网格信息
            output_hidden_states: 是否输出所有层的 hidden states

        Returns:
            dict with:
                - logits: (batch_size, seq_len, num_labels) token-level scores
                - hidden_states: 如果 output_hidden_states=True
        """
        # 获取 base model 的输出
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            output_hidden_states=True,
            **kwargs
        )

        # 获取最后一层 hidden states
        sequence_output = outputs[0]

        # 计算 scores
        logits = self.v_head(sequence_output)  # (batch_size, seq_len, num_labels)

        result = {
            'logits': logits,
        }

        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None

        return result

    def save_pretrained(self, save_path: str):
        """
        保存模型

        将 base_model 和 v_head 的权重合并保存，同时单独保存 value_head.pt
        以兼容 trl/verl 的加载方式。

        Args:
            save_path: 保存路径
        """
        os.makedirs(save_path, exist_ok=True)

        # 1. 获取 base model 权重
        state_dict = self.base_model.state_dict()

        # 2. 获取 v_head 权重并添加前缀
        v_head_state = self.v_head.state_dict()
        for k, v in v_head_state.items():
            state_dict[f"v_head.{k}"] = v

        # 3. 保存合并的 checkpoint（使用 base_model 的 save_pretrained 接口）
        self.base_model.save_pretrained(save_path, state_dict=state_dict)

        # 4. 单独保存 value_head.pt（兼容 trl/verl）
        # 转换为 trl 格式：v_head.0 -> dropout (无权重), v_head.1 -> summary
        # 使用 .detach().clone() 确保 tensor 拥有独立 storage
        v_head_trl_state = {}
        for k, v in v_head_state.items():
            if k == "1.weight":
                v_head_trl_state["v_head.summary.weight"] = v.detach().clone()
            elif k == "1.bias":
                v_head_trl_state["v_head.summary.bias"] = v.detach().clone()
            else:
                v_head_trl_state[f"v_head.{k}"] = v.detach().clone()

        value_head_path = os.path.join(save_path, "value_head.pt")
        torch.save(v_head_trl_state, value_head_path)

        print(f"Model saved to {save_path}")
        print(f"  - Merged checkpoint (base + v_head)")
        print(f"  - value_head.pt (trl/verl compatible)")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_labels: int = 1,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        """
        加载预训练的 Value Head 模型

        支持加载合并后的权重（base_model + v_head）

        Args:
            model_path: 模型路径
            num_labels: 输出维度
            gradient_checkpointing: 是否启用 gradient checkpointing

        Returns:
            加载好的模型实例
        """
        model = cls(
            model_path,
            num_labels=num_labels,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs
        )

        # 尝试加载 v_head 权重
        # 由于 save_pretrained 是把 v_head 混在 base_model 里保存的
        # base_model.from_pretrained 会忽略 v_head 权重，需要手动加载

        from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME

        # 支持分片 checkpoint
        weights_path = os.path.join(model_path, SAFE_WEIGHTS_NAME)
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, WEIGHTS_NAME)

        # 如果是分片 checkpoint，尝试找 model.safetensors.index.json
        if not os.path.exists(weights_path):
            index_path = os.path.join(model_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                loaded = _load_v_head_from_sharded_checkpoint(model_path, index_path, model.v_head)
                if loaded:
                    return model

        if os.path.exists(weights_path):
            # 加载 state_dict
            if weights_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(weights_path)
            else:
                state_dict = torch.load(weights_path, map_location="cpu")

            # 提取 v_head 权重
            v_head_state = {}
            for k, v in state_dict.items():
                if k.startswith("v_head."):
                    v_head_state[k.replace("v_head.", "")] = v

            if v_head_state:
                model.v_head.load_state_dict(v_head_state)
                print(f"Loaded v_head from {weights_path}")
            else:
                print(f"No v_head weights found in {weights_path}, using random init")
        else:
            print(f"No checkpoint found at {model_path}, using random init")

        return model


def _load_v_head_from_sharded_checkpoint(model_path: str, index_path: str, v_head: nn.Module) -> bool:
    """
    从分片 checkpoint 中加载 v_head 权重

    Args:
        model_path: 模型路径
        index_path: index.json 路径
        v_head: v_head 模块

    Returns:
        是否成功加载
    """
    import json
    from safetensors.torch import load_file

    try:
        with open(index_path, 'r') as f:
            index = json.load(f)

        weight_map = index.get('weight_map', {})

        # 找到包含 v_head 权重的分片文件
        v_head_files = set()
        for key in weight_map:
            if key.startswith('v_head.'):
                v_head_files.add(weight_map[key])

        if not v_head_files:
            print(f"No v_head weights found in sharded checkpoint")
            return False

        # 从对应的分片文件加载 v_head 权重
        v_head_state = {}
        for shard_file in v_head_files:
            shard_path = os.path.join(model_path, shard_file)
            if os.path.exists(shard_path):
                shard_dict = load_file(shard_path)
                for k, v in shard_dict.items():
                    if k.startswith("v_head."):
                        v_head_state[k.replace("v_head.", "")] = v

        if v_head_state:
            v_head.load_state_dict(v_head_state)
            print(f"Loaded v_head from sharded checkpoint")
            return True

        return False

    except Exception as e:
        print(f"Failed to load v_head from sharded checkpoint: {e}")
        return False


def get_last_token_scores(logits, attention_mask, debug: bool = False):
    """
    从输出中获取最后一个有效 token 的分数

    支持 Left Padding 和 Right Padding:
    - Left Padding: [0, 0, ..., 0, 1, 1, ..., 1] -> 最后有效 token 在序列末尾
    - Right Padding: [1, 1, ..., 1, 0, 0, ..., 0] -> 最后有效 token 在 sum-1 位置

    Args:
        logits: [batch_size, seq_len, num_labels] 模型输出
        attention_mask: [batch_size, seq_len] 注意力掩码
        debug: 是否打印调试信息

    Returns:
        scores: [batch_size, num_labels] 最后一个有效 token 的分数
    """
    batch_size = logits.size(0)
    seq_len = logits.size(1)

    # 检测 padding 类型（检查第一个样本的第一个位置）
    if attention_mask[0, 0] == 0:
        # Left padding: 最后一个有效 token 在序列末尾
        seq_lengths = torch.full((batch_size,), seq_len - 1,
                                  dtype=torch.long, device=logits.device)
        padding_type = "left"
    else:
        # Right padding: 最后一个有效 token 是 attention_mask 最后一个 1 的位置
        seq_lengths = attention_mask.sum(dim=1) - 1
        padding_type = "right"

    batch_indices = torch.arange(batch_size, device=logits.device)

    # 取最后一个有效 token 的 logits
    scores = logits[batch_indices, seq_lengths]  # [batch_size, num_labels]

    if debug:
        print(f"  [DEBUG] padding type: {padding_type}, seq_lengths: {seq_lengths.tolist()}")
        print(f"  [DEBUG] attention_mask sum: {attention_mask.sum()}")
        print(f"  [DEBUG] logits min/max: {logits.min():.4f}/{logits.max():.4f}")
        last_valid = seq_lengths[0].item()
        print(f"  [DEBUG] scores at last valid pos {last_valid}: {logits[0, last_valid, 0]:.4f}")

    return scores
