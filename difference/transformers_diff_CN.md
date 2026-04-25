# PRISM 内置 `transformers-4.57.0` 相对官方版的 Qwen 改动说明

> 本文件对比 **PRISM 内置的 `transformers-4.57.0`** 与 **官方 `transformers v4.57.0`**。
> 范围：`src/transformers/models/` 下所有 `qwen*` 子目录。
> 结论：所有改动集中在 **3 个目录**，与 MoE 视觉模型的新增和 RL critic 训练支持相关。

## 0. 总览

PRISM 共有 14 个 `qwen*` 目录，官方有 12 个。逐目录 `git diff --no-index` 结果：

| 目录 | 状态 | 说明 |
|---|---|---|
| `qwen2` | 一致 | 无改动 |
| `qwen2_5_omni` | 一致 | 无改动 |
| `qwen2_5_vl` | 一致 | 无改动 |
| `qwen2_5_vl_moe` | **新增** | PRISM 自实现的 Qwen2.5-VL MoE 变体 |
| `qwen2_audio` | 一致 | 无改动 |
| `qwen2_moe` | 一致 | 无改动 |
| `qwen2_vl` | 一致 | 无改动 |
| `qwen2_vl_moe` | **新增** | PRISM 自实现的 Qwen2-VL MoE 变体 |
| `qwen3` | 一致 | 无改动 |
| `qwen3_moe` | 一致 | 无改动 |
| `qwen3_next` | 一致 | 无改动 |
| `qwen3_omni_moe` | 一致 | 无改动 |
| `qwen3_vl` | 一致 | 无改动 |
| `qwen3_vl_moe` | **修改** | 2 个文件，新增 `Qwen3VLMoeForTokenClassification` 类 + 1 处小改 |

---

## 1. 新增目录：`qwen2_5_vl_moe/`

### 1.1 文件清单

| 文件 | 大小 | 行数（约） | 说明 |
|---|---|---|---|
| `__init__.py` | 1.0 KB | 29 | 标准 lazy import 入口 |
| `configuration_qwen2_5_vl_moe.py` | 14 KB | ~330 | 新模型配置类 |
| `modeling_qwen2_5_vl_moe.py` | 81 KB | ~1800 | 完整 standalone 实现 |
| `modular_qwen2_5_vl_moe.py` | 25 KB | ~570 | 模块化继承式实现 |

### 1.2 核心新增类

- **配置类**：`Qwen2_5_VLMoeTextConfig`、`Qwen2_5_VLMoeVisionConfig`、`Qwen2_5_VLMoeConfig`，新增 MoE 参数：`num_experts`、`moe_intermediate_size`、`num_experts_per_tok`、`decoder_sparse_step`、`router_aux_loss_coef`、`output_router_logits` 等。
- **MoE 模块**：
  - `Qwen2_5_VLMoeTextExperts`：fused gate-up-proj 形式的多专家层（参数形状 `(num_experts, hidden_size, 2 * expert_dim)`）。
  - `Qwen2_5_VLMoeTextSparseMoeBlock`：top-k 路由 + softmax + scatter 权重。
  - `Qwen2_5_VLMoeTextDecoderLayer`：根据 `(layer_idx + 1) % decoder_sparse_step == 0` 决定使用 dense MLP 还是 MoE。
- **视觉编码器**：`Qwen2_5_VLMoeVisionModel` 等（与 `qwen2_5_vl` 视觉端结构对齐）。
- **顶层模型**：`Qwen2_5_VLMoeModel`（VLM 主体）、`Qwen2_5_VLMoeForConditionalGeneration`（生成式）、`Qwen2_5_VLMoeForTokenClassification`（**RL critic 用 value head**）。
- **辅助函数**：`load_balancing_loss_func` 等 MoE 标准辅助损失。

### 1.3 与官方 `qwen2_5_vl` 的关系

整体上把 dense 文本解码器替换为 sparse MoE，参考 `qwen3_vl_moe` 的结构在 Qwen2.5-VL 上回填，便于 PRISM 三阶段流程中 SFT/RLVR 直接使用 Qwen2.5 系列的 MoE 版本。

> **注意**：该目录目前**未注册**到 `models/__init__.py`、`auto/configuration_auto.py`、`auto/modeling_auto.py`，意味着无法通过 `AutoModel.from_pretrained("...qwen2_5_vl_moe...")` 自动加载，需要直接 `from transformers.models.qwen2_5_vl_moe import ...` 显式导入，或在使用前手动补齐 auto 注册。

---

## 2. 新增目录：`qwen2_vl_moe/`

### 2.1 文件清单

| 文件 | 大小 | 行数（约） | 说明 |
|---|---|---|---|
| `__init__.py` | 1.0 KB | 28 | 标准 lazy import 入口 |
| `configuration_qwen2_vl_moe.py` | 17 KB | ~390 | 新模型配置类 |
| `modeling_qwen2_vl_moe.py` | 85 KB | ~1900 | 完整实现 |

（无 `modular_*.py`，是一份完整 standalone 实现。）

### 2.2 核心新增类

- **配置类**：`Qwen2VLMoeVisionConfig`、`Qwen2VLMoeTextConfig`、`Qwen2VLMoeConfig`。
- **MoE 模块**：
  - `Qwen2VLMoeTextExperts`：注释中明确写 "Fused experts implementation (Qwen3 MoE style)"，`gate_up_proj` 与 `down_proj` 为 3D 参数，跑专家路由。
  - `Qwen2VLMoeSparseMoeBlock`：路由 logits → softmax → top-k → scatter 至 `(batch*seq, num_experts)`。
  - `Qwen2VLMoeDecoderLayer`：每隔 `decoder_sparse_step` 层用 MoE，否则用 `Qwen2VLMoeMLP`（dense）。
- **视觉编码器**：`Qwen2VLMoeVisionTransformerPretrainedModel`、`PatchEmbed`、`PatchMerger`、`VisionRotaryEmbedding`、`VisionAttention` 等（保留 Qwen2-VL 原始 2D RoPE 视觉栈）。
- **顶层模型**：`Qwen2VLMoeModel`、`Qwen2VLMoeForConditionalGeneration`，`forward` 中支持 `output_router_logits`、`router_aux_loss_coef` 加权辅助损失，输出 `Qwen2VLMoeCausalLMOutputWithPast`（含 `aux_loss` 与 `router_logits` 字段）。
- **辅助函数**：`load_balancing_loss_func` 支持 EP 分布式（`rank` 切片）。

### 2.3 与官方 `qwen2_vl` 的关系

将 Qwen2-VL 文本端从 dense FFN 升级为 Qwen3-MoE 风格的稀疏专家结构，但保留了 Qwen2 的 vision 实现（与 `qwen2_5_vl_moe` 不同的视觉端，更接近原始 Qwen2-VL）。

> **注意**：该目录**已注册**到 `models/__init__.py`（第 286 行）、`auto/configuration_auto.py`（含 `qwen2_vl_moe` 与 `qwen2_vl_moe_text` 两个 model_type）、`auto/modeling_auto.py`（`AutoModel`、`AutoModelForCausalLM`、`AutoModelForVision2Seq` 三处映射），可直接通过 Auto API 加载。

---

## 3. 修改目录：`qwen3_vl_moe/`

`git diff --stat`：2 files changed, **+157 / -1**。

### 3.1 `modeling_qwen3_vl_moe.py`：+80 / -1

#### 3.1.1 dtype 修复（1 行）

`Qwen3VLMoeTextSparseMoeBlock.forward` 中 scatter 操作显式指定 dtype，避免 `routing_weights`（fp32）与 `router_logits`（可能为 bf16/fp16）混合导致的精度回退或运行错误：

```python
# 修改前
router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
# 修改后
router_weights = torch.zeros_like(router_logits, dtype=routing_weights.dtype).scatter_(1, router_indices, routing_weights)
```

#### 3.1.2 新增 `Qwen3VLMoeForTokenClassification` 类（约 79 行）

为 RL 训练新增带 value head 的多模态 critic 模型：

- 以完整的 `Qwen3VLMoeModel`（VLM backbone）为骨干。
- 在末层 hidden state 上接 `nn.Dropout` + `nn.Linear(hidden_size, num_labels=1, bias=False)` 作为 score head。
- `forward` 直接透传 VLM 的多模态输入：`pixel_values`、`pixel_values_videos`、`image_grid_thw`、`video_grid_thw`，并返回 `loss / logits / hidden_states / attentions` 字典。
- 在 `__all__` 中追加 `"Qwen3VLMoeForTokenClassification"`。

类 docstring 明确写出意图：

> Qwen3VLMoe 模型用于 Token Classification 任务（带 value head）
> 参考 GenericForTokenClassification，但支持视觉输入 (pixel_values, image_grid_thw 等)
> 可以被 verl 的 load_valuehead_model 直接加载作为 critic model

### 3.2 `modular_qwen3_vl_moe.py`：+78 / 0

把同一个 `Qwen3VLMoeForTokenClassification` 类的源码补到 modular 实现中，保证 `modular → modeling` 自动生成的一致性，同时在 `__all__` 末尾追加：

```python
"Qwen3VLMoeForTokenClassification",  # 新增
```

> **注意**：modular 文件中没有 dtype 修复那一行，意味着如果未来重新执行 `transformers` 的 modular-to-modeling 自动转换，dtype 修复可能被覆盖丢失，需要把同一处补丁同步到 modular 文件，或在 RLVR 训练前固定使用现有 `modeling_qwen3_vl_moe.py`。

---

## 4. PRISM 改动动机汇总（一句话）

为了让 PRISM 的「SFT → on-policy 蒸馏（PRISM）→ RLVR」三阶段流程在 Qwen MoE 多模态家族上跑通，作者做了三件事：

1. **补齐 Qwen2-VL / Qwen2.5-VL 的 MoE 版本**（新增两个完整目录），把 Qwen3-MoE 的稀疏专家结构回填到更早的 VL 系列上，使整个 Qwen-VL 家族都能享受 MoE 容量。
2. **给 Qwen3-VL-MoE 加一个 `ForTokenClassification` value-head 头**，使其可被 verl 的 critic 加载器直接当成多模态奖励/评分模型，支撑 GAD/PPO 阶段的对抗式判别器训练（与 `verl` 端的 `compute_discriminator_loss` 配合）。
3. **修复一处 MoE router scatter 的 dtype bug**，避免在 bf16/fp16 训练时报错或精度劣化。

除此以外的 11 个 `qwen*` 目录与官方 v4.57.0 完全一致，未做任何改动。
