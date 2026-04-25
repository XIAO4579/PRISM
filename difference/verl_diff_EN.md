# Changes in PRISM's bundled `verl` vs. upstream v0.6.1

> This document compares **PRISM's bundled `verl`** against the **official `verl-release-v0.6.1`**.
> Scope: `verl/workers/critic/` and `verl/utils/reward_score/`.
> Conclusion: all changes were introduced to support PRISM's **mm_gad (multimodal adversarial discriminator distillation)** training pipeline.

## 0. Overview

| Directory | Changes |
|---|---|
| `verl/workers/critic/` | Only `dp_critic.py` is modified (+875 / -64). `base.py`, `megatron_critic.py`, `__init__.py` are byte-identical to upstream |
| `verl/utils/reward_score/` | 2 files modified + 4 files added; total +1248 / -13 |

---

# Part 1: `verl/workers/critic/dp_critic.py`

A single file, but huge changes: **+875 / -64 lines**. The core is to upgrade the dense PPO Critic into a unified worker that supports three modes: PPO Critic / GAD discriminator / mm_gad caption-cot dual discriminator.

## 1.1 New configuration switches (in `__init__`)

```python
self._critic_supports_mm    # auto-detect whether the critic backbone has vision_config
self.use_gad                # GAD mode: teacher vs. student whole-response discrimination
self.use_mm_gad             # mm_gad mode: separate caption / cot scoring
self.score_clip             # hard clip of scores into [-c, c]
self.use_score_length_correction  # score - beta * log(1 + length) length debiasing
self.caption_length_correction_beta
self.cot_length_correction_beta
self.score_reg_coef         # L2 regularization on scores to bound their magnitude
self._tokenizer / _tokenizer_path  # lazy-loaded tokenizer for mm_gad text-format validation
```

New static / helper methods:
- `tokenizer` property: lazily loads `AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)`.
- `_extract_response_lengths(full_attention_mask, response_tokens)`: takes the response slice from `attention_mask` and returns the per-sample valid-token count (≥1).
- `_apply_score_length_correction(scores, lengths, beta)`: applies `score - beta * log1p(length)` debiasing.

## 1.2 `_forward_micro_batch` refactor

- **New `compute_teacher` parameter**: selects student (`input_ids/attention_mask/position_ids/responses`) or teacher (`teacher_*`) fields from `micro_batch` to feed into the critic.
- **mrope position_ids compatibility**: 3D `position_ids` (`(batch, 4, seq)`) now uses only the first dimension `position_ids[:, 0, :]`, so a text-only critic (e.g., `Qwen3MoeForTokenClassification`) can accept it correctly.
- **Multimodal-input gating**: `multi_modal_inputs` is injected only if `self._critic_supports_mm == True`, preventing illegal-kwarg errors on text-only critics.
- **Value-head output compatibility**: upstream only supports the `(loss, logits, value)` tuple from `trl.AutoModelForCausalLMWithValueHead`. PRISM also supports custom dict-returning value heads such as `Qwen3VLMoeForTokenClassification`, reading `output['logits']`.
- **Slice-position switching**:
  - Standard PPO: `values = values[:, -response_length-1:-1]` (predict next-token value, original behavior).
  - GAD / mm_gad: `values = values[:, -response_length:] * last_token_mask`, keeping only the score at the last valid token of the response (used as the whole-segment discriminator score).
- **Numerical safety**: detects NaN/Inf and replaces with `nan_to_num(0.0)`; optionally clips by `score_clip`.

## 1.3 `compute_values` refactor

- In `mm_gad` mode, dispatches to a new method `_compute_values_mm_gad` that returns a dict (`values / caption_scores / cot_scores / format_scores`).
- In standard mode, explicitly calls `_forward_micro_batch(model_inputs, compute_teacher=False)` per micro batch; the original concat logic is preserved.
- The return type is now `Union[Tensor, dict]`; the docstring is updated accordingly.

## 1.4 New `_compute_values_mm_gad`

The full mm_gad inference path:

1. Decodes each response back to text via `tokenizer.decode` and matches `<caption>...</caption><think>...</think><answer>...</answer>` with regex.
2. For each sample, finds the caption end position by re-decoding token by token, yielding `caption_end / cot_start` boundaries.
3. Invalid-format samples use fallback boundaries (`response_length // 2`) so every rank performs the same number of forwards (avoids NCCL deadlock); their scores are then forcibly zeroed.
4. Builds `prompt + caption` and `prompt + cot` inputs, calls `_forward_micro_batch` to get scores (`sum(dim=-1)` to aggregate).
5. Optional length correction and `score_clip`.
6. Writes `caption_score + cot_score` to the last valid token position before returning.

Output dict:
```python
{
    "values": (batch, response_length),  # combined score at the last position, 0 elsewhere
    "caption_scores": (batch, 1),         # used by GRPO multi-branch advantage
    "cot_scores":     (batch, 1),
    "format_scores":  (batch, 1),         # 1 = format valid, 0 = invalid
}
```

## 1.5 New `_apply_format_validation_mask`

A standalone format validator: tokenizer-decode + regex; samples that violate the `<caption><think><answer>` structure get their reward set to `0` (FORMAT_PENALTY = 0.0; no negative penalty).

## 1.6 New `_preprocess_student_inputs`

Used during `update_critic` training:

- Parses caption/cot boundaries on every response up front.
- Pads all caption/cot tensors to uniform lengths.
- Concatenates with the prompt to form full `student_caption_input_ids / student_cot_input_ids`, plus matching attention_mask and position_ids (extending RoPE positions from the end of the prompt).
- Injects everything back into `data.batch` so subsequent `_forward_micro_batch` calls can run as a true batch and **retain a complete backward graph** (unlike the per-sample inference path).

## 1.7 Three-mode branch in `update_critic`

The single `compute_value_loss` path is replaced with a switch over three losses:

### 1.7.1 Standard PPO (default, equivalent to upstream)
Keeps `core_algos.compute_value_loss`. Metrics: `vf_loss / vf_clipfrac / vpred_mean`.

### 1.7.2 `use_gad`: whole-response discrimination
- Forwards both student and teacher, calls `core_algos.compute_discriminator_loss(student_vpreds, teacher_vpreds, student_mask, teacher_mask)` (Bradley-Terry style `-logsigmoid(teacher_score - student_score)`).
- Metrics: `critic/d_loss`, `critic/d_acc` (fraction of samples where teacher scores higher), `critic/student_value_mean`, `critic/teacher_value_mean`.

### 1.7.3 `use_mm_gad`: caption + cot dual discrimination
**4 forwards per micro batch**: student-caption / student-cot / teacher-caption / teacher-cot.

- Loss is two BT pairwise terms: `cap_pair_loss + cot_pair_loss`.
- Only samples with valid format (`format_valid_mask`) contribute to the numerator; the denominator is the local valid count.
- **Cross-rank sync of valid_count**: `torch.distributed.all_reduce(global_valid_count, SUM)`. Switches between the main loss and a fallback loss based on whether any rank has any valid sample, preventing zero-grad deadlocks when some ranks have no valid samples.
- **Fallback loss**: `small_coef * (-logsigmoid(teacher_caption_scores.mean()) - logsigmoid(teacher_cot_scores.mean()))`.
- Optional L2 score regularization: `score_reg_coef * mean(student_cap² + student_cot² + teacher_cap² + teacher_cot²) / 4`.
- Detailed metrics (each rescaled by `loss_scale_factor`):
  - `critic/d_loss / cap_loss / cot_loss / score_reg / score_reg_loss`
  - `critic/d_acc / cap_acc / cot_acc / valid_ratio`
  - `critic/student_caption_mean / student_cot_mean / teacher_caption_mean / teacher_cot_mean`
  - When the local rank has no valid samples, accuracy falls back to `DEFAULT_ACC_NO_VALID_SAMPLES = 0.92` so wandb does not show NaN spikes.

## 1.8 Extended `select_keys` in `update_critic`

In mm_gad mode, `select_keys` additionally includes:

```python
"teacher_input_ids", "teacher_response", "teacher_attention_mask", "teacher_position_ids",
"teacher_caption", "teacher_caption_input_ids", "teacher_caption_attention_mask", "teacher_caption_position_ids",
"teacher_cot", "teacher_cot_input_ids", "teacher_cot_attention_mask", "teacher_cot_position_ids",
```

Immediately after `select`, `_preprocess_student_inputs(data)` is called to inject `student_caption_*` and `student_cot_*` fields.

---

# Part 2: `verl/utils/reward_score/`

## 2.1 Modified: `__init__.py`

`default_compute_score` and `_default_compute_score` both gain an `equivalent_answers=None` parameter (for math problems with multiple correct answers).

The dispatch table adds two new `data_source` branches:

| data_source | Routes to | Purpose |
|---|---|---|
| `"mm_gad"` | `mm_gad_no_llm.compute_score(solution_str, ground_truth, extra_info=extra_info)` | Default mm_gad training reward |
| `"math-77k"`, `"visual_logic-26k"` | `math_verify_with_format.compute_score(...)` | DeepVision-style math reward |

> Note: `vl_agent` and `mm_gad_llm` are **not registered** in the dispatch table; they can only be used via explicit `from verl.utils.reward_score.vl_agent import compute_score`-style imports.

## 2.2 Modified: `geo3k.py`

Switches the reward format from LaTeX `\boxed{}` to the same XML three-section structure as mm_gad:

| Item | Before | After |
|---|---|---|
| `format_reward` regex | `<think>.*</think>.*\boxed{...}` | `<caption>.*</caption>.*<think>.*</think>.*<answer>.*</answer>` |
| Answer extraction | `extract_boxed_content(predict_str)` | New `extract_answer_content` using `<answer>(.*?)</answer>` |
| Parameter name | `use_boxed: bool = True` | `use_answer_tag: bool = True` |
| Return value | `float` (scalar) | `dict(score, acc_reward, format_reward)` |

This makes the interface identical to `mm_gad/math_verify_with_format` (scalar → dict), so upstream loggers can record everything uniformly.

## 2.3 New: `math_verify_with_format.py` (139 lines)

DeepVision-style math scoring with mm_gad-style format gating:

- **Format gating**: matches `<caption>...</caption><think>...</think><answer>...</answer>` first; if it fails, returns `{score: 0, acc: 0, format: 0}`.
- **Answer extraction**: takes `answer_text` from `<answer>`, wraps it as `\boxed{answer_text}`, then feeds into `math_verify`.
- **Multiple correct answers**: iterates `[ground_truth] + equivalent_answers` and returns 1.0 if any matches; preserves the `__EMPTY__` placeholder convention.
- **Robust fallbacks**: `math_verify`'s `TimeoutException` returns `0.0`; overlong answers (≥1024 chars) skip the acc check.
- **Final score**: `score = 0.8 * acc + 0.2 * format` (matches `mm_gad_no_llm`).

## 2.4 New: `mm_gad_llm.py` (142 lines)

LLM-as-judge variant of the mm_gad reward:

- **Env-var-tunable**: `MM_GAD_MAX_RETRIES` / `MM_GAD_RETRY_BACKOFF` / `MM_GAD_TIMEOUT` / `MM_GAD_LOG_LEVEL`.
- Same format gating and `<answer>` extraction.
- Calls `gpt-4o-mini` over an OpenAI-compatible API to judge 1/0; retries with exponential backoff on failure.
- Returns `{score: acc, acc_reward: acc, format_reward: 1.0}`.

> **Usage note**: before use, fill in a valid OpenAI-compatible credential for `openai_api_key` and `openai_api_base_list`; ideally read them from environment variables.

## 2.5 New: `mm_gad_no_llm.py` (378 lines)

The **purely rule-based** reward used by mm_gad training by default (no LLM calls; fast and reproducible):

Main blocks:
- **Format gating**: same three-section regex as above.
- **MCQ recognition** (`_parse_mcq`): supports `(A)`, `A.`, `Answer is A`, etc.; ranks candidates by priority.
- **Numeric matching** (`_try_parse_number`): strips thousands separators and units; uses relative-error comparison.
- **LaTeX normalization** (`_normalize_latex`): `\dfrac→\frac`, strips `\text{}`, `\left`/`\right`, etc.
- **Unit recognition** (`_UNIT_PATTERN`): 30+ Chinese and English units built in.
- **Multi-value answer splitting** (`_split_multi_value`): handles `(1) ... (2) ...`, semicolons, top-level commas.
- **`math_verify` as a fallback** (`_math_verify_match`): tries multiple LaTeX variants with `parse + verify`.
- **Entry point** `compute_score(...)`: combines format (weight 0.2) + acc (weight 0.8).

## 2.6 New: `vl_agent.py` (545 lines)

LLM-as-judge toolkit for **step-level visual-reasoning scoring**:

- Connects to volcengine-hosted `qwen3-vl-thinking` as the judge.
- Provides two prompts: `prompt_score` (depends on previous steps) and `prompt_score_independent` (perception-only scoring).
- Utilities: `encode_image_to_base64`, `generate_batch_score` (multi-threaded), `generate_batch_score_perception`, etc.
- Entry point `compute_score(predict_str, ground_truth, extra_info=None, **kwargs) -> float`: aggregates per-step visual-CoT scores into the final reward.

> **Usage note**: before use, fill in valid credentials for `BASE_URL` and `API_KEY` (the visual judge service); ideally read them from environment variables.

---

## 3. Why these changes (in one sentence)

To make verl support **multimodal adversarial discriminator training** within PRISM's "on-policy distillation → RLVR" pipeline:

1. **On the Critic side**: the worker is rebuilt as a unified switchable PPO / GAD (whole-response) / mm_gad (caption + cot dual) Critic, with all the necessary engineering glue: multimodal inputs, custom value heads, length correction, distributed format validation, numerical stability, and so on.
2. **On the reward side**: four new rewards are added: `mm_gad_no_llm` (rule-based) / `mm_gad_llm` (LLM judge) / `vl_agent` (visual step scoring) / `math_verify_with_format` (DeepVision-style math). `geo3k` and the dispatch interface are aligned to the same `<caption><think><answer>` + dict-output style, matching the mm_gad training data format.
