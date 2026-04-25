# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2.5-VL-MoE model."""

from typing import Optional, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer,
    Qwen2MoePreTrainedModel,
    Qwen2MoeRMSNorm,
    load_balancing_loss_func,
)
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from ..qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VisionTransformerPretrainedModel,
)


logger = logging.get_logger(__name__)


class Qwen2_5_VLMoeTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5_VLMoeTextModel`]. It is used to instantiate a
    Qwen2.5-VL-MoE model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152064):
            Vocabulary size of the Qwen2.5-VL-MoE model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of key_value heads for GQA.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output word embeddings.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size.
        max_window_layers (`int`, *optional*, defaults to 80):
            The number of layers using full attention.
        decoder_sparse_step (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts.
        num_experts (`int`, *optional*, defaults to 60):
            Number of routed experts.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        mlp_only_layers (`List[int]`, *optional*, defaults to `[]`):
            Indicate which layers use dense MLP rather than MoE.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
    """

    model_type = "qwen2_5_vl_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_dropout=0.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        decoder_sparse_step=1,
        moe_intermediate_size=1408,
        num_experts_per_tok=4,
        num_experts=60,
        norm_topk_prob=True,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            if self.rope_scaling["type"] == "mrope":
                self.rope_scaling["type"] = "default"
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2_5_VLMoeVisionConfig(Qwen2_5_VLVisionConfig):
    """Vision config for Qwen2.5-VL-MoE, inherits from Qwen2.5-VL vision config."""
    model_type = "qwen2_5_vl_moe"


class Qwen2_5_VLMoeConfig(Qwen2_5_VLConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2_5_VLMoeModel`]. It is used to instantiate a
    Qwen2.5-VL-MoE model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Qwen2_5_VLMoeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Qwen2_5_VLMoeVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151655):
            The image token index.
        video_token_id (`int`, *optional*, defaults to 151656):
            The video token index.
        vision_start_token_id (`int`, *optional*, defaults to 151652):
            The start token index for vision input.
        vision_end_token_id (`int`, *optional*, defaults to 151653):
            The end token index for vision input.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the word embeddings.
    """

    model_type = "qwen2_5_vl_moe"
    sub_configs = {"vision_config": Qwen2_5_VLMoeVisionConfig, "text_config": Qwen2_5_VLMoeTextConfig}


class Qwen2_5_VLMoeTextRMSNorm(Qwen2MoeRMSNorm):
    """RMSNorm for Qwen2.5-VL-MoE text model."""
    pass


class Qwen2_5_VLMoeTextExperts(nn.Module):
    """
    MoE Experts layer for Qwen2.5-VL-MoE.
    
    Uses fused gate_up_proj format: (num_experts, hidden_size, 2 * expert_dim)
    """
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for MoE experts.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor: Output hidden states
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        
        if self.training:
            next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx[0]])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx]
                gate, up = gate_up.chunk(2, dim=-1)
                gated_output = up * self.act_fn(gate)
                out = gated_output @ self.down_proj[expert_idx]
                weighted_output = out[0] * routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen2_5_VLMoeTextSparseMoeBlock(nn.Module):
    """
    Sparse MoE block for Qwen2.5-VL-MoE.
    
    Contains a router (gate) and the experts.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen2_5_VLMoeTextExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out, router_logits


class Qwen2_5_VLMoeTextDecoderLayer(Qwen2MoeDecoderLayer):
    """Decoder layer for Qwen2.5-VL-MoE text model."""
    pass


class Qwen2_5_VLMoePreTrainedModel(Qwen2MoePreTrainedModel):
    """PreTrainedModel for Qwen2.5-VL-MoE."""
    config_class = None  # Will be set in modeling file
    _no_split_modules = ["Qwen2_5_VLMoeTextDecoderLayer", "Qwen2_5_VLMoeVisionBlock"]

    def _init_weights(self, module):
        """Initialize the weights."""
        PreTrainedModel._init_weights(self, module)
        if hasattr(self.config, "initializer_range"):
            std = self.config.initializer_range
        else:
            std = getattr(self.config.get_text_config(), "initializer_range", 0.02)
        if isinstance(module, Qwen2_5_VLMoeTextExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)


class Qwen2_5_VLMoeVisionModel(Qwen2_5_VisionTransformerPretrainedModel):
    """Vision model for Qwen2.5-VL-MoE, uses the same architecture as Qwen2.5-VL."""
    pass


class Qwen2_5_VLMoeCausalLMOutputWithPast(Qwen2_5_VLCausalLMOutputWithPast):
    """Output class with auxiliary loss for MoE."""
    aux_loss: Optional[torch.FloatTensor] = None


class Qwen2_5_VLMoeModel(Qwen2_5_VLModel):
    """Base model for Qwen2.5-VL-MoE."""
    pass


class Qwen2_5_VLMoeForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL-MoE model for conditional generation.
    
    Combines the Qwen2.5-VL vision encoder with a MoE-based text decoder.
    """
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Forward pass for Qwen2.5-VL-MoE.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            past_key_values (`Cache`, *optional*):
                Pre-computed hidden-states (key and values in the self-attention blocks).
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss.
            pixel_values (`torch.Tensor`, *optional*):
                Pixel values for image inputs.
            pixel_values_videos (`torch.FloatTensor`, *optional*):
                Pixel values for video inputs.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval for each grid along the temporal dimension.
            cache_position (`torch.LongTensor`, *optional*):
                Cache position for incremental decoding.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Number of logits to keep for memory efficiency.

        Returns:
            `Qwen2_5_VLMoeCausalLMOutputWithPast`: Output with loss, logits, and auxiliary loss.
        """

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if hasattr(outputs, 'router_logits') else None,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None and aux_loss is not None and aux_loss != 0:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen2_5_VLMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, 'hidden_states', None),
            attentions=getattr(outputs, 'attentions', None),
            rope_deltas=getattr(outputs, 'rope_deltas', None),
        )


class Qwen2_5_VLMoeForTokenClassification(Qwen2_5_VLMoePreTrainedModel):
    """
    Qwen2.5-VL-MoE model for Token Classification tasks (e.g., value head for RL).
    
    Can be used as a critic model in reinforcement learning scenarios.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = getattr(config, 'num_labels', 1)
        
        # Use the full VLM model as backbone
        self.model = Qwen2_5_VLMoeModel(config)
        
        # Classification head (value head)
        text_config = config.text_config if hasattr(config, 'text_config') else config
        hidden_size = text_config.hidden_size
        
        classifier_dropout = getattr(config, 'classifier_dropout', 
                                     getattr(config, 'hidden_dropout', 0.0))
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(hidden_size, self.num_labels, bias=False)
        
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        Forward pass for token classification.

        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the token classification loss.
        
        Returns:
            dict: Dictionary containing loss, logits, hidden_states, and attentions.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        
        sequence_output = outputs[0]  # last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)
        
        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": getattr(outputs, 'hidden_states', None),
            "attentions": getattr(outputs, 'attentions', None),
        }


__all__ = [
    "Qwen2_5_VLMoeConfig",
    "Qwen2_5_VLMoeTextConfig",
    "Qwen2_5_VLMoeVisionConfig",
    "Qwen2_5_VLMoeVisionModel",
    "Qwen2_5_VLMoeForConditionalGeneration",
    "Qwen2_5_VLMoeModel",
    "Qwen2_5_VLMoePreTrainedModel",
    "Qwen2_5_VLMoeForTokenClassification",
]

