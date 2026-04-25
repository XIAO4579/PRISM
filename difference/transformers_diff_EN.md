# Qwen-related changes in PRISM's bundled `transformers-4.57.0` vs. upstream

> This document compares **PRISM's bundled `transformers-4.57.0`** against the **official `transformers v4.57.0`**.
> Scope: all `qwen*` sub-directories under `src/transformers/models/`.
> Conclusion: all changes are concentrated in **3 directories**, related to new MoE vision-language models and RL critic support.

## 0. Overview

PRISM ships 14 `qwen*` directories; upstream has 12. Per-directory `git diff --no-index` results:

| Directory | Status | Notes |
|---|---|---|
| `qwen2` | identical | no changes |
| `qwen2_5_omni` | identical | no changes |
| `qwen2_5_vl` | identical | no changes |
| `qwen2_5_vl_moe` | **new** | PRISM's own MoE variant of Qwen2.5-VL |
| `qwen2_audio` | identical | no changes |
| `qwen2_moe` | identical | no changes |
| `qwen2_vl` | identical | no changes |
| `qwen2_vl_moe` | **new** | PRISM's own MoE variant of Qwen2-VL |
| `qwen3` | identical | no changes |
| `qwen3_moe` | identical | no changes |
| `qwen3_next` | identical | no changes |
| `qwen3_omni_moe` | identical | no changes |
| `qwen3_vl` | identical | no changes |
| `qwen3_vl_moe` | **modified** | 2 files; new `Qwen3VLMoeForTokenClassification` class + 1 small fix |

---

## 1. New directory: `qwen2_5_vl_moe/`

### 1.1 File listing

| File | Size | LOC (approx.) | Description |
|---|---|---|---|
| `__init__.py` | 1.0 KB | 29 | Standard lazy-import entry point |
| `configuration_qwen2_5_vl_moe.py` | 14 KB | ~330 | Configuration classes |
| `modeling_qwen2_5_vl_moe.py` | 81 KB | ~1800 | Full standalone implementation |
| `modular_qwen2_5_vl_moe.py` | 25 KB | ~570 | Modular inheritance-based implementation |

### 1.2 Key new classes

- **Configs**: `Qwen2_5_VLMoeTextConfig`, `Qwen2_5_VLMoeVisionConfig`, `Qwen2_5_VLMoeConfig`. New MoE parameters: `num_experts`, `moe_intermediate_size`, `num_experts_per_tok`, `decoder_sparse_step`, `router_aux_loss_coef`, `output_router_logits`, etc.
- **MoE modules**:
  - `Qwen2_5_VLMoeTextExperts`: fused `gate_up_proj` style multi-expert layer (parameter shape `(num_experts, hidden_size, 2 * expert_dim)`).
  - `Qwen2_5_VLMoeTextSparseMoeBlock`: top-k routing + softmax + scatter weights.
  - `Qwen2_5_VLMoeTextDecoderLayer`: chooses dense MLP vs. MoE based on `(layer_idx + 1) % decoder_sparse_step == 0`.
- **Vision encoder**: `Qwen2_5_VLMoeVisionModel` etc. (aligned with the `qwen2_5_vl` vision stack).
- **Top-level models**: `Qwen2_5_VLMoeModel` (VLM body), `Qwen2_5_VLMoeForConditionalGeneration` (generative), `Qwen2_5_VLMoeForTokenClassification` (**RL critic with value head**).
- **Helpers**: `load_balancing_loss_func` and other standard MoE auxiliary-loss utilities.

### 1.3 Relation to upstream `qwen2_5_vl`

Effectively replaces the dense text decoder with a sparse MoE decoder, back-porting the structure of `qwen3_vl_moe` to Qwen2.5-VL so the PRISM 3-stage pipeline (SFT → alignment → RLVR) can use a Qwen2.5-VL MoE checkpoint directly.

> **Note**: This directory is **not registered** in `models/__init__.py`, `auto/configuration_auto.py`, or `auto/modeling_auto.py`. It cannot be loaded via `AutoModel.from_pretrained("...qwen2_5_vl_moe...")`; use explicit `from transformers.models.qwen2_5_vl_moe import ...` instead, or add the auto-mapping entries before use.

---

## 2. New directory: `qwen2_vl_moe/`

### 2.1 File listing

| File | Size | LOC (approx.) | Description |
|---|---|---|---|
| `__init__.py` | 1.0 KB | 28 | Standard lazy-import entry point |
| `configuration_qwen2_vl_moe.py` | 17 KB | ~390 | Configuration classes |
| `modeling_qwen2_vl_moe.py` | 85 KB | ~1900 | Full implementation |

(No `modular_*.py`; this is a fully standalone implementation.)

### 2.2 Key new classes

- **Configs**: `Qwen2VLMoeVisionConfig`, `Qwen2VLMoeTextConfig`, `Qwen2VLMoeConfig`.
- **MoE modules**:
  - `Qwen2VLMoeTextExperts`: docstring explicitly says "Fused experts implementation (Qwen3 MoE style)"; `gate_up_proj` and `down_proj` are 3D parameters driving expert routing.
  - `Qwen2VLMoeSparseMoeBlock`: router logits → softmax → top-k → scatter to `(batch*seq, num_experts)`.
  - `Qwen2VLMoeDecoderLayer`: uses MoE every `decoder_sparse_step` layers, otherwise `Qwen2VLMoeMLP` (dense).
- **Vision encoder**: `Qwen2VLMoeVisionTransformerPretrainedModel`, `PatchEmbed`, `PatchMerger`, `VisionRotaryEmbedding`, `VisionAttention`, etc. (keeps the original Qwen2-VL 2D-RoPE vision stack).
- **Top-level models**: `Qwen2VLMoeModel`, `Qwen2VLMoeForConditionalGeneration`. `forward` supports `output_router_logits` and applies `router_aux_loss_coef`-weighted auxiliary loss; outputs `Qwen2VLMoeCausalLMOutputWithPast` with `aux_loss` and `router_logits` fields.
- **Helpers**: `load_balancing_loss_func` supports EP-distributed slicing (via `rank`).

### 2.3 Relation to upstream `qwen2_vl`

Upgrades the Qwen2-VL text side from a dense FFN to a Qwen3-MoE-style sparse expert structure, while keeping the original Qwen2 vision implementation (a different vision encoder than `qwen2_5_vl_moe`, closer to the original Qwen2-VL).

> **Note**: This directory **is registered** in `models/__init__.py` (line 286), `auto/configuration_auto.py` (with both `qwen2_vl_moe` and `qwen2_vl_moe_text` model types), and `auto/modeling_auto.py` (mapped under `AutoModel`, `AutoModelForCausalLM`, `AutoModelForVision2Seq`). It can be loaded via the Auto APIs out of the box.

---

## 3. Modified directory: `qwen3_vl_moe/`

`git diff --stat`: 2 files changed, **+157 / -1**.

### 3.1 `modeling_qwen3_vl_moe.py`: +80 / -1

#### 3.1.1 dtype fix (1 line)

In `Qwen3VLMoeTextSparseMoeBlock.forward`, the scatter operation now explicitly specifies dtype to avoid mixing fp32 `routing_weights` with potentially bf16/fp16 `router_logits`, which could cause precision regressions or runtime errors:

```python
# before
router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
# after
router_weights = torch.zeros_like(router_logits, dtype=routing_weights.dtype).scatter_(1, router_indices, routing_weights)
```

#### 3.1.2 New `Qwen3VLMoeForTokenClassification` class (~79 lines)

A multimodal critic model with a value head, added for RL training:

- Uses the full `Qwen3VLMoeModel` (VLM backbone) as the trunk.
- Applies `nn.Dropout` + `nn.Linear(hidden_size, num_labels=1, bias=False)` on the last-layer hidden states as the score head.
- `forward` directly forwards the VLM's multimodal inputs: `pixel_values`, `pixel_values_videos`, `image_grid_thw`, `video_grid_thw`, and returns a `loss / logits / hidden_states / attentions` dict.
- Appends `"Qwen3VLMoeForTokenClassification"` to `__all__`.

The class docstring states the intent clearly:

> Qwen3VLMoe model for Token Classification (with value head).
> References GenericForTokenClassification but supports vision inputs (pixel_values, image_grid_thw, etc.).
> Can be loaded directly by verl's `load_valuehead_model` as a critic model.

### 3.2 `modular_qwen3_vl_moe.py`: +78 / 0

Adds the same `Qwen3VLMoeForTokenClassification` source to the modular implementation, keeping the `modular → modeling` auto-generation consistent, and appends to `__all__`:

```python
"Qwen3VLMoeForTokenClassification",  # newly added
```

> **Note**: The modular file does **not** include the dtype fix. If the modular-to-modeling auto-conversion is rerun in the future, the dtype fix could be lost. Either port the same patch into the modular file, or pin the current `modeling_qwen3_vl_moe.py` before any RLVR run.

---

## 4. Why these changes (in one sentence)

To make PRISM's "SFT → on-policy distillation (PRISM) → RLVR" 3-stage pipeline work on the Qwen MoE multimodal family, the authors did three things:

1. **Filled in MoE versions for Qwen2-VL / Qwen2.5-VL** (two entirely new directories), back-porting the Qwen3-MoE sparse-expert structure to the earlier VL series so the whole Qwen-VL family can benefit from MoE capacity.
2. **Added a `ForTokenClassification` value head to Qwen3-VL-MoE**, so it can be loaded directly by verl's critic loader as a multimodal reward / scoring model, supporting the discriminator-style training in the GAD/PPO stage (in conjunction with verl's `compute_discriminator_loss`).
3. **Fixed a dtype bug in the MoE router scatter**, avoiding errors and precision degradation under bf16/fp16 training.

All other 11 `qwen*` directories are byte-identical to upstream v4.57.0.
