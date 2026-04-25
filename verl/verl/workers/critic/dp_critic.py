# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement a multiprocess PPOCritic
"""

import logging
import math
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

        # Detect if critic model supports multi-modal inputs (e.g., pixel_values).
        # Text-only models like Qwen3MoeForTokenClassification do not accept these kwargs.
        _module = critic_module
        while hasattr(_module, 'module'):
            _module = _module.module
        _config = getattr(_module, 'config', None)
        self._critic_supports_mm = (
            _config is not None
            and hasattr(_config, 'vision_config')
            and _config.vision_config is not None
        )
        if not self._critic_supports_mm:
            print("Critic is a text-only model; multi-modal inputs will be skipped.")
        
        # whether to use gad (teacher/discriminator mode, whole response scoring)
        self.use_gad = self.config.get("use_gad", False)
        if self.use_gad:
            print("Critic using gad mode (teacher/discriminator, whole response)")
        
        # whether to use mm_gad (caption/cot separate scoring for multimodal)
        self.use_mm_gad = self.config.get("use_mm_gad", False)
        if self.use_mm_gad:
            print("Critic using mm_gad mode (caption/cot separate scoring)")
        
        self.score_clip = self.config.get("score_clip", None)
        if self.score_clip is not None:
            print(f"Critic score clipping enabled: [-{self.score_clip}, {self.score_clip}]")

        # Optional score-level length correction for mm_gad rewards/losses.
        self.use_score_length_correction = self.config.get("use_score_length_correction", False)
        base_beta = self.config.get("score_length_correction_beta", 0.0)
        self.score_length_correction_beta = float(base_beta if base_beta is not None else 0.0)
        cap_beta = self.config.get("caption_length_correction_beta", self.score_length_correction_beta)
        cot_beta = self.config.get("cot_length_correction_beta", self.score_length_correction_beta)
        self.caption_length_correction_beta = float(cap_beta if cap_beta is not None else self.score_length_correction_beta)
        self.cot_length_correction_beta = float(cot_beta if cot_beta is not None else self.score_length_correction_beta)
        if self.use_score_length_correction:
            print(
                "Critic score length correction enabled: "
                f"caption_beta={self.caption_length_correction_beta}, cot_beta={self.cot_length_correction_beta}"
            )

        # Optional regularization to keep discriminator score magnitude bounded.
        reg_coef = self.config.get("score_reg_coef", 0.0)
        self.score_reg_coef = float(reg_coef if reg_coef is not None else 0.0)
        if self.score_reg_coef > 0:
            print(f"Critic score regularization enabled: coef={self.score_reg_coef}")
        
        # Lazy-load tokenizer for text-based format validation
        self._tokenizer = None
        self._tokenizer_path = None
        if self.use_mm_gad:
            # Get tokenizer path from model config - try multiple possible locations
            model_config = self.config.get("model", {})
            if hasattr(model_config, 'get'):
                self._tokenizer_path = model_config.get("tokenizer_path") or model_config.get("path")
            elif hasattr(model_config, 'tokenizer_path'):
                self._tokenizer_path = model_config.tokenizer_path
            elif hasattr(model_config, 'path'):
                self._tokenizer_path = model_config.path
            
            # Fallback: try config.model_config
            if not self._tokenizer_path:
                model_config = self.config.get("model_config", {})
                if hasattr(model_config, 'get'):
                    self._tokenizer_path = model_config.get("tokenizer_path") or model_config.get("path")
                elif hasattr(model_config, 'tokenizer_path'):
                    self._tokenizer_path = model_config.tokenizer_path
                elif hasattr(model_config, 'path'):
                    self._tokenizer_path = model_config.path
            
            if self._tokenizer_path:
                print(f"Critic will use tokenizer from: {self._tokenizer_path}")
            else:
                print("WARNING: Could not find tokenizer_path in critic config! Format validation will fail.")

    @property
    def tokenizer(self):
        """Lazy-load tokenizer for format validation."""
        if self._tokenizer is None and self._tokenizer_path:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)
            print(f"Loaded tokenizer from {self._tokenizer_path}")
        return self._tokenizer

    @staticmethod
    def _extract_response_lengths(full_attention_mask: torch.Tensor, response_tokens: torch.Tensor) -> torch.Tensor:
        """Extract valid response lengths from full attention mask."""
        response_window = response_tokens.size(-1)
        return full_attention_mask[:, -response_window:].sum(dim=-1).to(dtype=torch.float32).clamp_min(1.0)

    def _apply_score_length_correction(self, scores: torch.Tensor, lengths: torch.Tensor, beta: float) -> torch.Tensor:
        """Apply score debiasing term: score - beta * log(1 + length)."""
        if not self.use_score_length_correction:
            return scores
        lengths = lengths.to(device=scores.device, dtype=scores.dtype).clamp_min(1.0)
        return scores - beta * torch.log1p(lengths)

    def _forward_micro_batch(self, micro_batch, compute_teacher=False):
        # determine response length based on teacher or student
        if compute_teacher:
            response_length = micro_batch["teacher_response"].size(-1)
        else:
            response_length = micro_batch["responses"].size(-1)
        
        multi_modal_inputs = {}
        if self._critic_supports_mm and "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            # select input_ids, attention_mask, position_ids based on teacher or student
            if compute_teacher:
                input_ids = micro_batch["teacher_input_ids"]
                attention_mask = micro_batch["teacher_attention_mask"]
                position_ids = micro_batch["teacher_position_ids"]
            else:
                input_ids = micro_batch["input_ids"]
                attention_mask = micro_batch["attention_mask"]
                position_ids = micro_batch["position_ids"]
            
            batch, seqlen = input_ids.shape
            if position_ids.dim() == 3:  # qwen2vl mrope
                # position_ids = position_ids.transpose(0, 1)
                # For text-only critic model, use only text position (first dimension)
                position_ids = position_ids[:, 0, :]  # (batch, 4, seq) -> (batch, seq)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead or custom value head models
                    if isinstance(output, dict):
                        # Custom value head model (e.g., Qwen3VLMoeValueHead) returns dict
                        # logits shape: (1, total_nnz, num_labels), squeeze to (total_nnz, num_labels)
                        values_rmpad = output['logits'].squeeze(0)
                    else:
                        # trl's AutoModelForCausalLMWithValueHead returns (loss, logits, value)
                        values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outputs_and_unpad(
                        values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                
                if self.use_gad or self.use_mm_gad:
                    # for gad/mm_gad mode: get last token value for discriminator
                    values = values[:, -response_length:]
                    response_mask = attention_mask[:, -response_length:]
                    response_lengths = response_mask.sum(dim=1).long()
                    last_token_indices = response_lengths - 1
                    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                    last_token_mask[batch_indices, last_token_indices] = True
                    values = values * last_token_mask.type_as(values)
                else:
                    values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead or custom value head models
                    if isinstance(output, dict):
                        # Custom value head model (e.g., Qwen3VLMoeValueHead) returns dict
                        # logits shape: (batch, seq_len, num_labels), squeeze last dim if num_labels=1
                        values = output['logits'].squeeze(-1)
                    else:
                        # trl's AutoModelForCausalLMWithValueHead returns (loss, logits, value)
                        values = output[2]
                else:
                    values = output.logits
                
                if self.use_gad or self.use_mm_gad:
                    # for gad/mm_gad mode: get last token value for discriminator
                    values = values[:, -response_length:].squeeze(-1)
                    response_mask = attention_mask[:, -response_length:]
                    response_lengths = response_mask.sum(dim=1).long()
                    last_token_indices = response_lengths - 1
                    last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                    batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                    last_token_mask[batch_indices, last_token_indices] = True
                    values = values * last_token_mask.type_as(values)
                else:
                    values = values[:, -response_length - 1 : -1].squeeze(-1)
            
            if not torch.isfinite(values).all():
                nan_count = torch.isnan(values).sum().item()
                inf_count = torch.isinf(values).sum().item()
                print(f"WARN: critic forward output has {nan_count} NaN, {inf_count} Inf values, replacing with zeros")
                values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            if self.score_clip is not None:
                values = values.clamp(-self.score_clip, self.score_clip)

            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto):
        """Compute values for the given data.
        
        Returns:
            For mm_gad mode: dict with 'values', 'caption_scores', 'cot_scores', 'format_scores'
            For standard mode: torch.Tensor of values
        """
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        # select keys for student only (compute_values always computes student values)
        select_keys = (
            ["responses", "input_ids", "response_mask", "attention_mask", "position_ids"]
            if "response_mask" in data.batch
            else ["responses", "input_ids", "attention_mask", "position_ids"]
        )
        
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # For mm_gad mode: compute caption and cot scores separately (consistent with training)
        if self.use_mm_gad:
            result = self._compute_values_mm_gad(data)
            values = result["values"]
            
            if "response_mask" in data.batch:
                response_mask = data.batch["response_mask"]
                response_mask = response_mask.to(values.device)
                result["values"] = values * response_mask  # Only action tokens have values
            
            return result
        else:
            # Standard mode: compute values for full response
            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                micro_batches = data.split(micro_batch_size)

            values_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                with torch.no_grad():
                    values = self._forward_micro_batch(model_inputs, compute_teacher=False)
                values_lst.append(values)
            values = torch.concat(values_lst, dim=0)

            if use_dynamic_bsz:
                values = restore_dynamic_batch(values, batch_idx_list)

            if "response_mask" in data.batch:
                response_mask = data.batch["response_mask"]
                response_mask = response_mask.to(values.device)
                values = values * response_mask  # Only action tokens have values
            return values
    
    def _compute_values_mm_gad(self, data: DataProto) -> dict:
        """Compute values for mm_gad mode by separately scoring caption and cot.
        
        This ensures consistency between training (where we train on separate caption/cot)
        and inference (where we compute rewards for Actor).
        
        Returns:
            dict with:
                - values: (batch_size, response_length) tensor with combined scores at last token
                - caption_scores: (batch_size,) tensor with caption scores for each sample
                - cot_scores: (batch_size,) tensor with cot scores for each sample
                - format_scores: (batch_size,) binary format-valid scores (1 valid, 0 invalid)
        """
        import re
        
        def validate_and_extract_text(response_text: str):
            """Validate format and extract caption/cot boundaries using text regex."""
            pattern = r"^\s*(<caption>.*?</caption>)\s*(<think>.*?</think>\s*<answer>.*?</answer>)\s*$"
            match = re.match(pattern, response_text, re.DOTALL)
            if match:
                caption_text = match.group(1)
                cot_text = match.group(2)
                return (True, caption_text, cot_text)
            else:
                return (False, None, None)
        
        def find_text_boundary_in_tokens(response_ids, tokenizer, target_text):
            """Find the token index where target_text ends in response."""
            for end_idx in range(1, len(response_ids) + 1):
                decoded = tokenizer.decode(response_ids[:end_idx], skip_special_tokens=True)
                if target_text in decoded:
                    return end_idx
            return -1
        
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        input_ids = data.batch["input_ids"]
        position_ids = data.batch["position_ids"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
        batch_size = responses.size(0)
        device = responses.device
        prompt_length = input_ids.size(1) - response_length
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        multi_modal_inputs = data.non_tensor_batch.get("multi_modal_inputs", None)
        
        # Initialize output values tensor
        values = torch.zeros(batch_size, response_length, device=device)
        # Initialize separate score tensors for GRPO multi-reward advantage
        caption_scores = torch.zeros(batch_size, device=device)
        cot_scores = torch.zeros(batch_size, device=device)
        format_scores = torch.zeros(batch_size, device=device)
        
        # Check format and find split points for each sample
        valid_mask = []
        caption_end_indices = []
        cot_start_indices = []
        
        tokenizer = self.tokenizer
        
        for i in range(batch_size):
            response_i = responses[i]
            response_mask_i = response_mask[i]
            valid_length = response_mask_i.sum().item()
            response_ids_i = response_i[:int(valid_length)].tolist()
            
            if tokenizer is not None:
                response_text = tokenizer.decode(response_ids_i, skip_special_tokens=True)
                is_valid, caption_text, cot_text = validate_and_extract_text(response_text)
                
                if is_valid:
                    valid_mask.append(True)
                    cap_end_idx = find_text_boundary_in_tokens(response_ids_i, tokenizer, caption_text)
                    caption_end_indices.append(cap_end_idx)
                    cot_start_indices.append(cap_end_idx)
                else:
                    valid_mask.append(False)
                    caption_end_indices.append(-1)
                    cot_start_indices.append(-1)
            else:
                valid_mask.append(False)
                caption_end_indices.append(-1)
                cot_start_indices.append(-1)
        
        valid_mask = torch.tensor(valid_mask, device=device)

        # Compute caption and cot scores for all samples.
        # For invalid-format samples, use fallback split boundaries so every rank
        # executes the same number of forwards, then force their scores to zero.
        with torch.no_grad():
            for i in range(batch_size):
                is_valid = bool(valid_mask[i].item())
                format_scores[i] = 1.0 if is_valid else 0.0
                cap_end = caption_end_indices[i]
                cot_start = cot_start_indices[i]
                valid_length = int(response_mask[i].sum().item())

                # Fallback split for invalid or malformed boundaries:
                # keep compute path consistent, and later zero-out rewards.
                if (not is_valid) or cap_end <= 0 or cot_start < 0:
                    fallback_valid_length = max(1, valid_length)
                    cap_end = max(1, fallback_valid_length // 2)
                    cot_start = cap_end

                cap_end = min(cap_end, response_length)
                cot_start = min(cot_start, response_length - 1)
                
                # Extract prompt
                prompt_ids = input_ids[i, :prompt_length]
                prompt_mask = attention_mask[i, :prompt_length]
                if position_ids.dim() == 3:
                    prompt_pos = position_ids[i, 0, :prompt_length]
                else:
                    prompt_pos = position_ids[i, :prompt_length]
                
                # Build student caption batch: prompt + caption
                student_caption_ids = responses[i, :cap_end]
                student_caption_full_ids = torch.cat([prompt_ids, student_caption_ids], dim=-1)
                student_caption_mask = torch.cat([
                    prompt_mask, 
                    torch.ones(cap_end, device=device, dtype=prompt_mask.dtype)
                ], dim=-1)
                cap_delta = torch.arange(1, cap_end + 1, device=device)
                student_caption_pos = torch.cat([prompt_pos, prompt_pos[-1:] + cap_delta], dim=-1)
                
                student_cap_batch = {
                    "input_ids": student_caption_full_ids.unsqueeze(0),
                    "attention_mask": student_caption_mask.unsqueeze(0),
                    "position_ids": student_caption_pos.unsqueeze(0),
                    "responses": student_caption_ids.unsqueeze(0),
                }
                if has_multi_modal_inputs:
                    student_cap_batch["multi_modal_inputs"] = [multi_modal_inputs[i]]
                
                # Compute caption score
                cap_vpreds = self._forward_micro_batch(student_cap_batch, compute_teacher=False)
                caption_score = cap_vpreds.sum().item()
                cap_len = student_caption_ids.size(0)
                
                # Build student cot batch: prompt + cot (from cot_start to end)
                cot_end = max(valid_length, cot_start + 1)
                cot_end = min(cot_end, response_length)
                student_cot_ids = responses[i, cot_start:cot_end]
                if student_cot_ids.numel() == 0:
                    student_cot_ids = responses[i, cot_start : cot_start + 1]
                cot_len = student_cot_ids.size(0)
                student_cot_full_ids = torch.cat([prompt_ids, student_cot_ids], dim=-1)
                student_cot_mask = torch.cat([
                    prompt_mask,
                    torch.ones(cot_len, device=device, dtype=prompt_mask.dtype)
                ], dim=-1)
                cot_delta = torch.arange(1, cot_len + 1, device=device)
                student_cot_pos = torch.cat([prompt_pos, prompt_pos[-1:] + cot_delta], dim=-1)
                
                student_cot_batch = {
                    "input_ids": student_cot_full_ids.unsqueeze(0),
                    "attention_mask": student_cot_mask.unsqueeze(0),
                    "position_ids": student_cot_pos.unsqueeze(0),
                    "responses": student_cot_ids.unsqueeze(0),
                }
                if has_multi_modal_inputs:
                    student_cot_batch["multi_modal_inputs"] = [multi_modal_inputs[i]]
                
                # Compute cot score
                cot_vpreds = self._forward_micro_batch(student_cot_batch, compute_teacher=False)
                cot_score = cot_vpreds.sum().item()

                if self.use_score_length_correction:
                    caption_score -= self.caption_length_correction_beta * math.log1p(cap_len)
                    cot_score -= self.cot_length_correction_beta * math.log1p(cot_len)

                if self.score_clip is not None:
                    c = self.score_clip
                    caption_score = max(-c, min(c, caption_score))
                    cot_score = max(-c, min(c, cot_score))

                # No penalty for invalid samples: keep them neutral.
                if not is_valid:
                    caption_score = 0.0
                    cot_score = 0.0

                caption_scores[i] = caption_score
                cot_scores[i] = cot_score
                
                total_score = caption_score + cot_score
                
                # Put the combined score at the last valid token position
                if valid_length > 0:
                    last_valid_idx = valid_length - 1
                    values[i, last_valid_idx] = total_score
        
        return {
            "values": values,
            # Unsqueeze to 2D for DataProto chunk compatibility
            "caption_scores": caption_scores.unsqueeze(-1),
            "cot_scores": cot_scores.unsqueeze(-1),
            "format_scores": format_scores.unsqueeze(-1),
        }
    
    def _apply_format_validation_mask(self, data: DataProto, values: torch.Tensor) -> torch.Tensor:
        """Apply format validation for mm_gad mode.
        
        Validates that responses follow the format:
        <caption>...</caption><think>...</think><answer>...</answer>
        
        Uses text-based regex matching instead of token IDs for robustness.
        Invalid samples get values set to 0.
        """
        import re
        
        def validate_strict_format_text(response_text: str) -> bool:
            """Validate format using regex on decoded text.
            
            Expected format: <caption>...</caption><think>...</think><answer>...</answer>
            Allows whitespace between tags.
            """
            # Pattern: starts with <caption>, ends with </answer>, proper nesting
            pattern = r"^\s*<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
            return bool(re.match(pattern, response_text, re.DOTALL))
        
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
        batch_size = responses.size(0)
        device = values.device
        
        # Validate each sample using text matching
        valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        if self.tokenizer is not None:
            for i in range(batch_size):
                # Get valid tokens (non-padding)
                mask_i = response_mask[i]
                valid_length = mask_i.sum().item()
                response_ids_i = responses[i, :int(valid_length)]
                
                # Decode to text
                response_text = self.tokenizer.decode(response_ids_i, skip_special_tokens=True)
                
                # Validate format
                is_valid = validate_strict_format_text(response_text)
                valid_mask[i] = is_valid
        else:
            # If no tokenizer, assume all valid (fallback)
            logger.warning("No tokenizer available for format validation, assuming all valid")
        
        # Keep invalid samples neutral (0 reward), no format penalty.
        # values shape: (batch_size, response_length)
        FORMAT_PENALTY = 0.0
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            values[invalid_mask] = FORMAT_PENALTY
        
        return values

    def _preprocess_student_inputs(self, data):
        """Pre-construct batched student caption/cot inputs for mm_gad training.

        Parses each response to find caption/cot boundary, then pads all samples
        to uniform lengths so they can be forward-passed as a batch with gradients.
        """
        import re

        def validate_and_extract_text(response_text: str):
            pattern = r"^\s*(<caption>.*?</caption>)\s*(<think>.*?</think>\s*<answer>.*?</answer>)\s*$"
            match = re.match(pattern, response_text, re.DOTALL)
            if match:
                return (True, match.group(1), match.group(2))
            return (False, None, None)

        def find_text_boundary_in_tokens(response_ids, tokenizer, target_text):
            for end_idx in range(1, len(response_ids) + 1):
                decoded = tokenizer.decode(response_ids[:end_idx], skip_special_tokens=True)
                if target_text in decoded:
                    return end_idx
            return -1

        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        input_ids = data.batch["input_ids"]
        position_ids = data.batch["position_ids"]

        batch_size = responses.size(0)
        response_length = responses.size(1)
        prompt_length = input_ids.size(1) - response_length
        device = responses.device
        response_mask = attention_mask[:, -response_length:]

        tokenizer = self.tokenizer

        valid_mask_list = []
        student_caption_list = []
        student_cot_list = []

        for i in range(batch_size):
            valid_length = int(response_mask[i].sum().item())
            response_ids_i = responses[i, :valid_length].tolist()

            is_valid = False
            cap_end = max(1, valid_length // 2)

            if tokenizer is not None:
                response_text = tokenizer.decode(response_ids_i, skip_special_tokens=True)
                fmt_valid, caption_text, _ = validate_and_extract_text(response_text)
                if fmt_valid:
                    boundary = find_text_boundary_in_tokens(response_ids_i, tokenizer, caption_text)
                    if boundary > 0:
                        is_valid = True
                        cap_end = boundary

            valid_mask_list.append(is_valid)
            student_caption_list.append(responses[i, :cap_end].detach().clone())
            cot_tokens = responses[i, cap_end:valid_length].detach().clone()
            if cot_tokens.numel() == 0:
                cot_tokens = responses[i, cap_end:cap_end + 1].detach().clone()
            student_cot_list.append(cot_tokens)

        max_cap_len = max(t.size(0) for t in student_caption_list)
        max_cot_len = max(t.size(0) for t in student_cot_list)

        student_caption_padded = torch.zeros(batch_size, max_cap_len, device=device, dtype=responses.dtype)
        student_caption_resp_mask = torch.zeros(batch_size, max_cap_len, device=device, dtype=attention_mask.dtype)
        student_cot_padded = torch.zeros(batch_size, max_cot_len, device=device, dtype=responses.dtype)
        student_cot_resp_mask = torch.zeros(batch_size, max_cot_len, device=device, dtype=attention_mask.dtype)

        for i in range(batch_size):
            cap_len = student_caption_list[i].size(0)
            student_caption_padded[i, :cap_len] = student_caption_list[i]
            student_caption_resp_mask[i, :cap_len] = 1
            cot_len = student_cot_list[i].size(0)
            student_cot_padded[i, :cot_len] = student_cot_list[i]
            student_cot_resp_mask[i, :cot_len] = 1

        prompt_ids = input_ids[:, :prompt_length]
        prompt_mask = attention_mask[:, :prompt_length]
        if position_ids.dim() == 3:
            prompt_pos = position_ids[:, 0, :prompt_length]
        else:
            prompt_pos = position_ids[:, :prompt_length]

        student_caption_input_ids = torch.cat([prompt_ids, student_caption_padded], dim=-1)
        student_caption_attention_mask = torch.cat([prompt_mask, student_caption_resp_mask], dim=-1)
        cap_delta = torch.arange(1, max_cap_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        student_caption_position_ids = torch.cat([prompt_pos, prompt_pos[:, -1:] + cap_delta], dim=-1)

        student_cot_input_ids = torch.cat([prompt_ids, student_cot_padded], dim=-1)
        student_cot_attention_mask = torch.cat([prompt_mask, student_cot_resp_mask], dim=-1)
        cot_delta = torch.arange(1, max_cot_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        student_cot_position_ids = torch.cat([prompt_pos, prompt_pos[:, -1:] + cot_delta], dim=-1)

        format_valid_mask = torch.tensor(valid_mask_list, device=device, dtype=torch.bool)

        data.batch["student_caption"] = student_caption_padded
        data.batch["student_caption_input_ids"] = student_caption_input_ids
        data.batch["student_caption_attention_mask"] = student_caption_attention_mask
        data.batch["student_caption_position_ids"] = student_caption_position_ids
        data.batch["student_cot"] = student_cot_padded
        data.batch["student_cot_input_ids"] = student_cot_input_ids
        data.batch["student_cot_attention_mask"] = student_cot_attention_mask
        data.batch["student_cot_position_ids"] = student_cot_position_ids
        data.batch["format_valid_mask"] = format_valid_mask

        return data

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        # breakpoint()
        self.critic_module.train()
        metrics = {}

        if self.use_mm_gad:
            # mm_gad mode: use discriminator loss with teacher/student (caption + cot separate scoring)
            select_keys = [
                "input_ids", "responses", "attention_mask", "position_ids",
                "teacher_input_ids", "teacher_response", "teacher_attention_mask", "teacher_position_ids",
                # teacher caption and cot inputs for mm_gad mode
                "teacher_caption", "teacher_caption_input_ids", "teacher_caption_attention_mask", "teacher_caption_position_ids",
                "teacher_cot", "teacher_cot_input_ids", "teacher_cot_attention_mask", "teacher_cot_position_ids",
            ]
        else:
            # standard PPO mode
            select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if self.use_mm_gad:
            data = self._preprocess_student_inputs(data)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.critic_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    
                    if self.use_gad:
                        # gad mode: whole response discriminator loss (original use_mm_gad logic)
                        from verl.utils.torch_functional import masked_sum
                        
                        # for student
                        responses = model_inputs["responses"]
                        attention_mask = model_inputs["attention_mask"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]
                        
                        # for teacher
                        teacher_response = model_inputs["teacher_response"]
                        teacher_attention_mask = model_inputs["teacher_attention_mask"]
                        teacher_response_length = teacher_response.size(1)
                        teacher_response_mask = teacher_attention_mask[:, -teacher_response_length:]
                        
                        student_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=False)
                        teacher_vpreds = self._forward_micro_batch(model_inputs, compute_teacher=True)
                        
                        # compute discriminator accuracy
                        d_acc = (teacher_vpreds.sum(dim=-1) > student_vpreds.sum(dim=-1)).float().mean().detach().item()
                        
                        d_loss = core_algos.compute_discriminator_loss(
                            student_vpreds=student_vpreds,
                            teacher_vpreds=teacher_vpreds,
                            response_mask=response_mask,
                            teacher_response_mask=teacher_response_mask,
                        )
                        
                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            loss = d_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = d_loss * loss_scale_factor
                        
                        loss.backward()
                        
                        micro_batch_metrics.update(
                            {
                                "critic/d_loss": d_loss.detach().item() * loss_scale_factor,
                                "critic/d_acc": d_acc,
                                "critic/student_value_mean": masked_sum(student_vpreds, response_mask, axis=-1).mean().detach().item(),
                                "critic/teacher_value_mean": masked_sum(teacher_vpreds, teacher_response_mask, axis=-1).mean().detach().item(),
                            }
                        )
                    elif self.use_mm_gad:
                        import torch.nn.functional as F

                        responses = model_inputs["responses"]
                        attention_mask = model_inputs["attention_mask"]
                        response_length = responses.size(1)
                        response_mask = attention_mask[:, -response_length:]
                        batch_size = responses.size(0)
                        device = responses.device

                        valid_mask = model_inputs["format_valid_mask"]

                        teacher_caption_input_ids = model_inputs.get("teacher_caption_input_ids")
                        teacher_caption_attention_mask = model_inputs.get("teacher_caption_attention_mask")
                        teacher_caption_position_ids = model_inputs.get("teacher_caption_position_ids")
                        teacher_cot_input_ids = model_inputs.get("teacher_cot_input_ids")
                        teacher_cot_attention_mask = model_inputs.get("teacher_cot_attention_mask")
                        teacher_cot_position_ids = model_inputs.get("teacher_cot_position_ids")
                        teacher_caption = model_inputs.get("teacher_caption")
                        teacher_cot = model_inputs.get("teacher_cot")

                        required_teacher_inputs = {
                            "teacher_caption_input_ids": teacher_caption_input_ids,
                            "teacher_caption_attention_mask": teacher_caption_attention_mask,
                            "teacher_caption_position_ids": teacher_caption_position_ids,
                            "teacher_cot_input_ids": teacher_cot_input_ids,
                            "teacher_cot_attention_mask": teacher_cot_attention_mask,
                            "teacher_cot_position_ids": teacher_cot_position_ids,
                            "teacher_caption": teacher_caption,
                            "teacher_cot": teacher_cot,
                        }
                        missing_teacher_inputs = [k for k, v in required_teacher_inputs.items() if v is None]
                        if missing_teacher_inputs:
                            raise RuntimeError(
                                f"mm_gad mode requires teacher caption/cot inputs, missing: {missing_teacher_inputs}"
                            )

                        # Student caption forward (WITH gradients)
                        student_caption_batch = {
                            "input_ids": model_inputs["student_caption_input_ids"],
                            "attention_mask": model_inputs["student_caption_attention_mask"],
                            "position_ids": model_inputs["student_caption_position_ids"],
                            "responses": model_inputs["student_caption"],
                        }
                        if "multi_modal_inputs" in model_inputs:
                            student_caption_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        student_caption_vpreds = self._forward_micro_batch(student_caption_batch, compute_teacher=False)
                        student_caption_scores = student_caption_vpreds.sum(dim=-1)

                        # Student cot forward (WITH gradients)
                        student_cot_batch = {
                            "input_ids": model_inputs["student_cot_input_ids"],
                            "attention_mask": model_inputs["student_cot_attention_mask"],
                            "position_ids": model_inputs["student_cot_position_ids"],
                            "responses": model_inputs["student_cot"],
                        }
                        if "multi_modal_inputs" in model_inputs:
                            student_cot_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        student_cot_vpreds = self._forward_micro_batch(student_cot_batch, compute_teacher=False)
                        student_cot_scores = student_cot_vpreds.sum(dim=-1)

                        # Teacher forward (WITH gradients, batched)
                        teacher_caption_batch = {
                            "input_ids": teacher_caption_input_ids,
                            "attention_mask": teacher_caption_attention_mask,
                            "position_ids": teacher_caption_position_ids,
                            "responses": teacher_caption,
                        }
                        if "multi_modal_inputs" in model_inputs:
                            teacher_caption_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        teacher_caption_vpreds = self._forward_micro_batch(teacher_caption_batch, compute_teacher=False)
                        teacher_caption_scores = teacher_caption_vpreds.sum(dim=-1)

                        teacher_cot_batch = {
                            "input_ids": teacher_cot_input_ids,
                            "attention_mask": teacher_cot_attention_mask,
                            "position_ids": teacher_cot_position_ids,
                            "responses": teacher_cot,
                        }
                        if "multi_modal_inputs" in model_inputs:
                            teacher_cot_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                        teacher_cot_vpreds = self._forward_micro_batch(teacher_cot_batch, compute_teacher=False)
                        teacher_cot_scores = teacher_cot_vpreds.sum(dim=-1)

                        if self.use_score_length_correction:
                            student_caption_lens = self._extract_response_lengths(
                                model_inputs["student_caption_attention_mask"],
                                model_inputs["student_caption"],
                            )
                            student_cot_lens = self._extract_response_lengths(
                                model_inputs["student_cot_attention_mask"],
                                model_inputs["student_cot"],
                            )
                            teacher_caption_lens = self._extract_response_lengths(
                                teacher_caption_attention_mask,
                                teacher_caption,
                            )
                            teacher_cot_lens = self._extract_response_lengths(
                                teacher_cot_attention_mask,
                                teacher_cot,
                            )

                            student_caption_scores = self._apply_score_length_correction(
                                student_caption_scores, student_caption_lens, self.caption_length_correction_beta
                            )
                            student_cot_scores = self._apply_score_length_correction(
                                student_cot_scores, student_cot_lens, self.cot_length_correction_beta
                            )
                            teacher_caption_scores = self._apply_score_length_correction(
                                teacher_caption_scores, teacher_caption_lens, self.caption_length_correction_beta
                            )
                            teacher_cot_scores = self._apply_score_length_correction(
                                teacher_cot_scores, teacher_cot_lens, self.cot_length_correction_beta
                            )

                        if self.score_clip is not None:
                            c = self.score_clip
                            student_caption_scores = student_caption_scores.clamp(-c, c)
                            student_cot_scores = student_cot_scores.clamp(-c, c)
                            teacher_caption_scores = teacher_caption_scores.clamp(-c, c)
                            teacher_cot_scores = teacher_cot_scores.clamp(-c, c)

                        # Use a single computation graph on every rank to avoid collective desync.
                        valid_weights = valid_mask.float()
                        local_valid_count = valid_weights.sum()
                        global_valid_count = local_valid_count.detach().clone()
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.all_reduce(global_valid_count, op=torch.distributed.ReduceOp.SUM)

                        cap_diff_all = teacher_caption_scores - student_caption_scores
                        cot_diff_all = teacher_cot_scores - student_cot_scores
                        cap_pair_loss = (-F.logsigmoid(cap_diff_all) * valid_weights).sum() / local_valid_count.clamp_min(1.0)
                        cot_pair_loss = (-F.logsigmoid(cot_diff_all) * valid_weights).sum() / local_valid_count.clamp_min(1.0)
                        pairwise_loss = cap_pair_loss + cot_pair_loss

                        # Fallback when the global batch has no valid sample.
                        small_coef = 0.01
                        fallback_loss = small_coef * (
                            -F.logsigmoid(teacher_caption_scores.mean()) - F.logsigmoid(teacher_cot_scores.mean())
                        )
                        has_global_valid = float(global_valid_count.item() > 0)
                        d_loss = has_global_valid * pairwise_loss + (1.0 - has_global_valid) * fallback_loss
                        cap_loss = has_global_valid * cap_pair_loss + (1.0 - has_global_valid) * (
                            small_coef * (-F.logsigmoid(teacher_caption_scores.mean()))
                        )
                        cot_loss = has_global_valid * cot_pair_loss + (1.0 - has_global_valid) * (
                            small_coef * (-F.logsigmoid(teacher_cot_scores.mean()))
                        )

                        score_reg = (
                            student_caption_scores.pow(2).mean()
                            + student_cot_scores.pow(2).mean()
                            + teacher_caption_scores.pow(2).mean()
                            + teacher_cot_scores.pow(2).mean()
                        ) / 4.0
                        if self.score_reg_coef > 0:
                            d_loss = d_loss + self.score_reg_coef * score_reg

                        local_valid_any = valid_mask.any()
                        if has_global_valid > 0 and local_valid_any:
                            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                            cap_acc = (cap_diff_all[valid_indices] > 0).float().mean().detach().item()
                            cot_acc = (cot_diff_all[valid_indices] > 0).float().mean().detach().item()
                            d_acc = (cap_acc + cot_acc) / 2
                        else:
                            DEFAULT_ACC_NO_VALID_SAMPLES = 0.92
                            cap_acc = DEFAULT_ACC_NO_VALID_SAMPLES
                            cot_acc = DEFAULT_ACC_NO_VALID_SAMPLES
                            d_acc = DEFAULT_ACC_NO_VALID_SAMPLES

                        valid_ratio = valid_mask.float().mean().item()

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            loss = d_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = d_loss * loss_scale_factor

                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "critic/d_loss": d_loss.detach().item() * loss_scale_factor,
                                "critic/cap_loss": cap_loss.detach().item() * loss_scale_factor,
                                "critic/cot_loss": cot_loss.detach().item() * loss_scale_factor,
                                "critic/d_acc": d_acc,
                                "critic/cap_acc": cap_acc,
                                "critic/cot_acc": cot_acc,
                                "critic/valid_ratio": valid_ratio,
                                "critic/score_reg": score_reg.detach().item(),
                                "critic/score_reg_loss": (self.score_reg_coef * score_reg).detach().item() * loss_scale_factor,
                                "critic/student_caption_mean": student_caption_scores[valid_mask].mean().detach().item() if local_valid_any else 0.0,
                                "critic/student_cot_mean": student_cot_scores[valid_mask].mean().detach().item() if local_valid_any else 0.0,
                                "critic/teacher_caption_mean": teacher_caption_scores[valid_mask].mean().detach().item() if local_valid_any else 0.0,
                                "critic/teacher_cot_mean": teacher_cot_scores[valid_mask].mean().detach().item() if local_valid_any else 0.0,
                            }
                        )

                    else:
                        # standard PPO mode: value loss
                        response_mask = model_inputs["response_mask"]
                        values = model_inputs["values"]
                        returns = model_inputs["returns"]

                        vpreds = self._forward_micro_batch(model_inputs, compute_teacher=False)
                        vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                            vpreds=vpreds,
                            values=values,
                            returns=returns,
                            response_mask=response_mask,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                            loss = vf_loss * loss_scale_factor
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                            loss = vf_loss * loss_scale_factor

                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                                "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                                "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                            }
                        )

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.critic_optimizer.zero_grad()
        return metrics
