# PRISM 内置 `verl` 相对官方 v0.6.1 的改动说明

> 本文件对比 **PRISM 内置的 `verl`** 与 **官方 `verl-release-v0.6.1`**。
> 范围：`verl/workers/critic/` 与 `verl/utils/reward_score/` 两个目录。
> 结论：所有改动都是为支持 PRISM 的 **mm_gad（多模态对抗式判别器蒸馏）** 训练流程而引入的。

## 0. 总览

| 目录 | 改动情况 |
|---|---|
| `verl/workers/critic/` | 仅 `dp_critic.py` 修改（+875 / -64），`base.py`、`megatron_critic.py`、`__init__.py` 与官方完全一致 |
| `verl/utils/reward_score/` | 2 个文件修改 + 4 个文件新增，共 +1248 / -13 |

---

# Part 1: `verl/workers/critic/dp_critic.py`

只有 1 个文件改动，但改动量巨大：**+875 / -64 行**。核心是把 dense 的 PPO Critic 改造成「PPO Critic / GAD 判别器 / mm_gad caption-cot 双判别器」的三模式 worker。

## 1.1 新增配置开关（`__init__` 内）

```python
self._critic_supports_mm    # 自动检测 critic backbone 是否支持 vision_config
self.use_gad                # GAD 模式：教师 vs 学生整 response 判别
self.use_mm_gad             # mm_gad 模式：caption / cot 分别打分
self.score_clip             # 分数硬裁剪 [-c, c]
self.use_score_length_correction  # score - beta * log(1 + length) 长度去偏
self.caption_length_correction_beta
self.cot_length_correction_beta
self.score_reg_coef         # 分数 L2 正则系数，约束 score 幅度
self._tokenizer / _tokenizer_path  # mm_gad 文本格式校验用 tokenizer，懒加载
```

并新增静态/辅助方法：
- `tokenizer` 属性：懒加载 `AutoTokenizer.from_pretrained(self._tokenizer_path, trust_remote_code=True)`。
- `_extract_response_lengths(full_attention_mask, response_tokens)`：从 attention_mask 中切出 response 段，按 sum 求实际有效 token 数（≥1）。
- `_apply_score_length_correction(scores, lengths, beta)`：执行 `score - beta * log1p(length)` 长度去偏。

## 1.2 `_forward_micro_batch` 重构

- **新增 `compute_teacher` 参数**：根据该参数从 `micro_batch` 中选择学生 (`input_ids/attention_mask/position_ids/responses`) 或教师 (`teacher_*`) 字段送入 critic。
- **mrope position_ids 兼容**：3D 的 `position_ids`（`(batch, 4, seq)`）改成只取第一维 `position_ids[:, 0, :]`，使纯文本 critic（如 `Qwen3MoeForTokenClassification`）也能正确接受。
- **多模态输入门控**：仅当 `self._critic_supports_mm == True` 才注入 `multi_modal_inputs`，避免文本 critic 收到非法 kwargs 报错。
- **value head 输出兼容**：原版只支持 `trl.AutoModelForCausalLMWithValueHead` 的 `(loss, logits, value)` tuple；新增对自定义 `Qwen3VLMoeForTokenClassification` 等返回 `dict` 的支持，从 `output['logits']` 取值。
- **取值位置切换**：
  - 标准 PPO 模式：`values = values[:, -response_length-1:-1]`（预测下一 token 价值，原行为）。
  - GAD / mm_gad 模式：`values = values[:, -response_length:] * last_token_mask`，只保留 response 最后一个有效 token 的分数（用作整段判别分）。
- **数值稳定性**：检查 NaN/Inf 并 `nan_to_num(0.0)`，可选地按 `score_clip` 裁剪。

## 1.3 `compute_values` 改造

- 在 `mm_gad` 模式下转走新方法 `_compute_values_mm_gad`，返回 dict（含 `values / caption_scores / cot_scores / format_scores`）。
- 在标准模式下，对每个 micro batch 显式 `compute_teacher=False` 调用 `_forward_micro_batch`，原始拼接逻辑保留。
- 返回类型从 `torch.Tensor` 扩展为 `Union[Tensor, dict]`，docstring 同步更新。

## 1.4 新增 `_compute_values_mm_gad`

完全的 mm_gad 推理路径：

1. 用 tokenizer.decode 把每条 response 变回文本，正则匹配 `<caption>...</caption><think>...</think><answer>...</answer>`。
2. 对每个样本通过 token 解码逐位回找 caption 子串结束位置，得到 `caption_end / cot_start` 边界。
3. 无效格式样本走 fallback 边界（`response_length // 2`），保证所有 rank 调用次数一致，避免 NCCL 死锁，最后强行 `caption_score = cot_score = 0`。
4. 分别拼接 `prompt + caption` / `prompt + cot` 作为输入，调用 `_forward_micro_batch` 拿到分数（`sum(dim=-1)` 求和）。
5. 可选长度修正、score_clip。
6. 把 `caption_score + cot_score` 写到响应最后一个有效 token 位置返回。

输出 dict：
```python
{
    "values": (batch, response_length),  # 末位填充组合分数，其余位置 0
    "caption_scores": (batch, 1),         # GRPO 多分支 advantage 用
    "cot_scores":     (batch, 1),
    "format_scores":  (batch, 1),         # 1 = 格式正确，0 = 错误
}
```

## 1.5 新增 `_apply_format_validation_mask`

独立的格式校验工具：基于 tokenizer 解码 + 正则把不符合 `<caption><think><answer>` 三段式的样本 reward 置 0（FORMAT_PENALTY = 0.0，无负罚分）。

## 1.6 新增 `_preprocess_student_inputs`

为 `update_critic` 训练阶段服务：

- 提前对每条响应解析 caption/cot 边界。
- 把所有样本 caption/cot pad 到统一长度。
- 拼接 prompt 形成完整 `student_caption_input_ids / student_cot_input_ids`，并构造对应的 attention_mask、position_ids（基于 prompt 末端 RoPE 偏移延伸）。
- 一次性塞回 `data.batch`，使后续 `_forward_micro_batch` 可以直接走批量前向，**保留完整反传梯度**（与推理阶段每样本逐次前向不同）。

## 1.7 `update_critic` 三模式分支

把单一的 `compute_value_loss` 路径改为根据开关选择三种损失：

### 1.7.1 标准 PPO（默认，等价原版）
保留 `core_algos.compute_value_loss`，metrics：`vf_loss / vf_clipfrac / vpred_mean`。

### 1.7.2 `use_gad`：整 response 判别
- 同时前向学生 + 教师，调用 `core_algos.compute_discriminator_loss(student_vpreds, teacher_vpreds, student_mask, teacher_mask)`（Bradley-Terry 形式 `-logsigmoid(teacher_score - student_score)`）。
- metrics：`critic/d_loss`、`critic/d_acc`（教师总分高于学生的比例）、`critic/student_value_mean`、`critic/teacher_value_mean`。

### 1.7.3 `use_mm_gad`：caption + cot 分别判别
对每个 micro batch 跑 **4 次前向**：student-caption / student-cot / teacher-caption / teacher-cot。

- 损失为两项 BT pairwise：`cap_pair_loss + cot_pair_loss`。
- 仅对格式合法（`format_valid_mask`）的样本计入分子，分母为本地 valid_count。
- **跨 rank 同步 valid_count**：`torch.distributed.all_reduce(global_valid_count, SUM)`，根据全局是否有 valid 样本切换主损失/兜底损失，防止某个 rank 没有合法样本时梯度全 0 导致死锁。
- **兜底损失**：`small_coef * (-logsigmoid(teacher_caption_scores.mean()) - logsigmoid(teacher_cot_scores.mean()))`。
- 可选 L2 score 正则：`score_reg_coef * mean(student_cap² + student_cot² + teacher_cap² + teacher_cot²)/4`。
- 详细 metrics（每条都按 `loss_scale_factor` 重新缩放）：
  - `critic/d_loss / cap_loss / cot_loss / score_reg / score_reg_loss`
  - `critic/d_acc / cap_acc / cot_acc / valid_ratio`
  - `critic/student_caption_mean / student_cot_mean / teacher_caption_mean / teacher_cot_mean`
  - 当本 rank 没有 valid 样本时，准确率回落到 `DEFAULT_ACC_NO_VALID_SAMPLES = 0.92`，避免 wandb 出现 NaN 跳变。

## 1.8 `update_critic` 数据 select_keys 扩展

mm_gad 模式下，`select_keys` 额外纳入：

```python
"teacher_input_ids", "teacher_response", "teacher_attention_mask", "teacher_position_ids",
"teacher_caption", "teacher_caption_input_ids", "teacher_caption_attention_mask", "teacher_caption_position_ids",
"teacher_cot", "teacher_cot_input_ids", "teacher_cot_attention_mask", "teacher_cot_position_ids",
```

并在 select 后立即调用 `_preprocess_student_inputs(data)` 注入 `student_caption_*` 与 `student_cot_*` 字段。

---

# Part 2: `verl/utils/reward_score/`

## 2.1 修改：`__init__.py`

`default_compute_score` 与 `_default_compute_score` 函数签名都新增 `equivalent_answers=None` 参数（用于 math 类多正确答案）。

dispatch 表新增两个 data_source 分支：

| data_source | 路由到的模块 | 用途 |
|---|---|---|
| `"mm_gad"` | `mm_gad_no_llm.compute_score(solution_str, ground_truth, extra_info=extra_info)` | mm_gad 训练默认奖励 |
| `"math-77k"`, `"visual_logic-26k"` | `math_verify_with_format.compute_score(...)` | DeepVision 风格 math 奖励 |

> 注意：`vl_agent` 与 `mm_gad_llm` 模块**未在 dispatch 表中注册**，目前只能由外部代码直接 `from verl.utils.reward_score.vl_agent import compute_score` 显式引用。

## 2.2 修改：`geo3k.py`

把奖励格式从 LaTeX `\boxed{}` 换成与 mm_gad 同款的 XML 标签三段式：

| 项 | 修改前 | 修改后 |
|---|---|---|
| `format_reward` 正则 | `<think>.*</think>.*\boxed{...}` | `<caption>.*</caption>.*<think>.*</think>.*<answer>.*</answer>` |
| 答案抽取 | `extract_boxed_content(predict_str)` | 新加 `extract_answer_content`，用 `<answer>(.*?)</answer>` 抽取 |
| 参数命名 | `use_boxed: bool = True` | `use_answer_tag: bool = True` |
| 返回值 | `float`（标量） | `dict(score, acc_reward, format_reward)` |

变成与 mm_gad/math_verify_with_format 完全一致的接口（标量 → dict），便于上层统一记录。

## 2.3 新增：`math_verify_with_format.py`（139 行）

DeepVision 风格的数学打分 + mm_gad 格式门控：

- **格式门控**：先匹配 `<caption>...</caption><think>...</think><answer>...</answer>`，不通过则 `{score: 0, acc: 0, format: 0}`。
- **答案抽取**：从 `<answer>` 中取出 `answer_text`，用 `\boxed{answer_text}` 包装后送入 `math_verify`。
- **多正确答案**：循环遍历 `[ground_truth] + equivalent_answers`，任一通过即认为正确；保留 `__EMPTY__` 占位约定。
- **稳健兜底**：`math_verify` 的 `TimeoutException` 直接返回 `0.0`；超长 answer（≥1024 字符）跳过 acc 判定。
- **最终分数**：`score = 0.8 * acc + 0.2 * format`（与 mm_gad_no_llm 一致）。

## 2.4 新增：`mm_gad_llm.py`（142 行）

LLM-as-judge 版本的 mm_gad 奖励：

- **环境变量可调**：`MM_GAD_MAX_RETRIES` / `MM_GAD_RETRY_BACKOFF` / `MM_GAD_TIMEOUT` / `MM_GAD_LOG_LEVEL`。
- **同样的格式门控**与 `<answer>` 抽取。
- 调用 `gpt-4o-mini` 通过 OpenAI 兼容 API 判 1/0；失败按指数退避重试。
- 返回 `{score: acc, acc_reward: acc, format_reward: 1.0}`。

> **使用提醒**：使用前需要在 `openai_api_key` 与 `openai_api_base_list` 中填入合法的 OpenAI 兼容 API 凭证；建议改成读环境变量。

## 2.5 新增：`mm_gad_no_llm.py`（378 行）

mm_gad 训练默认使用的**纯规则**奖励（无 LLM 调用，速度快、可复现）：

主要模块：
- **格式门控**：与上述一致的三段式正则。
- **多选题识别**（`_parse_mcq`）：支持 `(A)` / `A.` / `Answer is A` 等多种 MCQ 表达，按 priority 排序选最优候选。
- **数值答案匹配**（`_try_parse_number`）：去千分位逗号、去单位、按相对误差判等。
- **LaTeX 归一化**（`_normalize_latex`）：`\dfrac→\frac`、去 `\text{}`、去 `\left`/`\right` 等。
- **单位识别**（`_UNIT_PATTERN`）：内置 30+ 中英文单位。
- **多值答案分割**（`_split_multi_value`）：识别 `(1) ... (2) ...`、分号、顶层逗号等多值格式。
- **`math_verify` 加成匹配**（`_math_verify_match`）：在多 LaTeX 变体下用 `parse + verify` 兜底。
- **入口** `compute_score(...)`：组合 format（0.2 权重）+ acc（0.8 权重）。

## 2.6 新增：`vl_agent.py`（545 行）

针对**视觉推理 step-level 评分**的 LLM-as-judge 工具集：

- 接 volcengine 上的 `qwen3-vl-thinking` 模型作为评委。
- 提供两种 prompt：依赖前序步骤的 `prompt_score` 和独立感知评分的 `prompt_score_independent`。
- 工具函数：`encode_image_to_base64`、`generate_batch_score`（多线程）、`generate_batch_score_perception` 等。
- 入口 `compute_score(predict_str, ground_truth, extra_info=None, **kwargs) -> float`：用于把视觉 CoT 的每一步打分聚合成最终 reward。

> **使用提醒**：使用前需要在 `BASE_URL` 与 `API_KEY` 中填入合法的视觉模型评判服务凭证；建议改成读环境变量。

---

## 3. 改动动机汇总（一句话）

为了让 verl 在 PRISM 的「on-policy 蒸馏 → RLVR」流程中支持**多模态对抗式判别器训练**：

1. **Critic 端**改造成可在 PPO / GAD（整段判别）/ mm_gad（caption + cot 双判别）三种模式间切换的统一 worker，并补齐多模态输入、自定义 value-head、长度修正、分布式格式校验、数值稳定性等所有必要工程细节。
2. **Reward 端**新增 `mm_gad_no_llm`（规则）/ `mm_gad_llm`（LLM）/ `vl_agent`（视觉 step 评分）/ `math_verify_with_format`（DeepVision 风格 math）四套奖励，并把 `geo3k` 与 dispatch 接口统一成 `<caption><think><answer>` + dict-output 风格，与 mm_gad 训练数据格式对齐。
