# Beyond SFT-to-RL: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning

<div align="center">

[![arXiv](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.28123)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](https://xiao4579.github.io/PRISM/)
[![GitHub](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/XIAO4579/PRISM)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/prism-vlm)
[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/prism-vlm)

</div>

---

## 🎉 News

- **[2026-05-01]** We release our **paper** on arXiv: [Beyond SFT-to-RL: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning](https://arxiv.org/abs/2604.28123).
- **[2026-05-01]** We release the **code**, **data**, and **model checkpoints** of PRISM. Check out the [PRISM Collection](https://github.com/XIAO4579/PRISM).

---

## Table of Contents

- [News](#-news)
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [1. Environment setup](#1-environment-setup)
  - [2. Stage 1 — SFT (LLaMA-Factory)](#2-stage-1--sft-llama-factory)
  - [3. Stage 2 — PRISM Alignment (`qwen3_vl_prism`)](#3-stage-2--prism-alignment)
  - [4. Stage 3 — RLVR after PRISM (`qwen3_vl_xxpo_after_prism`)](#4-stage-3--rlvr-after-prism)
  - [5. Evaluation](#5-evaluation)
  - [6. Reproduction hyperparameters](#6-reproduction-hyperparameters)
  - [7. Repo structure](#7-repo-structure)
- [Local modifications](#local-modifications)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

<p align="center">
  <img src="alignment_pipeline.png" alt="PRISM pipeline overview" width="100%">
</p>

The standard post-training recipe for large multimodal models (LMMs), namely supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR), implicitly assumes that SFT produces a well-aligned initialization for online optimization. In practice, however, the post-SFT policy often drifts substantially from the supervision distribution, and this gap is especially harmful in multimodal reasoning, where perception errors and reasoning failures follow distinct drift patterns that compound during RL.

We introduce **PRISM** (**PR**e-alignment via on-policy d**IS**tillation for **M**ultimodal post-training), a three-stage pipeline that bridges this gap by inserting an explicit **distribution-alignment stage** between SFT and RLVR. PRISM formulates alignment as a response-level adversarial game between the policy and a **Mixture-of-Experts (MoE) discriminator** with dedicated **perception** and **reasoning** experts, providing disentangled corrective signals that steer the policy toward the supervision distribution **without requiring access to teacher logits**. To support high-fidelity supervision, we curate **113K** multimodal reasoning demonstrations from Gemini 3 Flash targeting the hardest unsolved problems, combined with **1.26M** public demonstrations for SFT. Experiments on Qwen3-VL show that PRISM consistently improves downstream RLVR performance across multiple RL algorithms (**GRPO**, **DAPO**, **GSPO**) and diverse multimodal benchmarks.

---

## Getting Started

### 1. Environment setup

#### 1.1 Create a conda env

```bash
conda create -n verl_0.6 python=3.12 -y
conda activate verl_0.6
```

#### 1.2 Install backends (vLLM / SGLang / FlashAttention / FlashInfer)

Run from the repo root (skipping Megatron):

```bash
USE_MEGATRON=0 bash verl/scripts/install_vllm_sglang_mcore.sh
```

#### 1.3 Install the bundled `verl`

Install the bundled `verl` in editable mode:

```bash
cd verl && pip install --no-deps -e . && cd ..
```

> #### Authenticate with HuggingFace (once, shared by all three stages)
>
> Before downloading any of the assets used below, log in once so the token is
> stored in `~/.cache/huggingface/` and picked up automatically:
>
> ```bash
> huggingface-cli login           # paste your token when prompted
> # or:  export HF_TOKEN=<YOUR_HF_TOKEN>
> ```
>
> Do **not** pass `--token` on the command line, since tokens leak into shell
> history and job logs.

PRISM is run as a **three-stage** pipeline. Each stage has its own data and
model preparation block below; SFT is run with LLaMA-Factory, while the
alignment and RLVR stages share the same `verl`-based training entrypoints
under `scripts/train/experiment/`.

| Stage | Tool / Script | Initialization | Objective |
|---|---|---|---|
| **1. SFT**         | LLaMA-Factory                                                                             | Qwen3-VL Instruct                       | Standard supervised fine-tuning on multimodal demonstrations |
| **2. Alignment**   | `qwen3_vl_prism.sh`                                                                       | post-SFT checkpoint                 | On-policy adversarial distillation against an MoE discriminator |
| **3. RLVR**        | `qwen3_vl_{grpo,dapo,gspo}_after_prism.sh`<br/>(collectively `qwen3_vl_xxpo_after_prism`) | post-alignment (PRISM) checkpoint   | Verifiable-reward RL (accuracy + format) |

### 2. Stage 1 — SFT (LLaMA-Factory)

Stage 1 supervised-fine-tunes the Qwen3-VL base on multimodal demonstrations,
producing a **post-SFT checkpoint** that initializes Stage 2.

#### 2.1 Data preparation

Two HuggingFace datasets are published in **LlamaFactory `sharegpt`-multimodal
format** and can be fed straight to LLaMA-Factory without writing any
conversion code:

| HuggingFace dataset                                                                                | Contents                                                                              |
|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [`prism-vlm/gemini_distill`](https://huggingface.co/datasets/prism-vlm/gemini_distill)             | ~113K curated Gemini-3-Flash multimodal reasoning demonstrations (the "hard" subset). |
| [`prism-vlm/gemini_public_mmr1`](https://huggingface.co/datasets/prism-vlm/gemini_public_mmr1)     | ~1.26M public demonstrations (incl. MMR1) used as the broad-coverage SFT mixture.     |

```bash
huggingface-cli download prism-vlm/gemini_distill \
    --repo-type dataset \
    --local-dir /path/to/datasets/gemini_distill
huggingface-cli download prism-vlm/gemini_public_mmr1 \
    --repo-type dataset \
    --local-dir /path/to/datasets/gemini_public_mmr1
```

#### 2.2 Model preparation

Start from the official Qwen3-VL base (pick the size that matches your
compute budget):

```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --repo-type model \
    --local-dir /path/to/models/Qwen3-VL-4B-Instruct
# (or Qwen/Qwen3-VL-8B-Instruct for the 8B variant)
```

#### 2.3 Launch SFT

Register the two datasets from §2.1 in your LLaMA-Factory `dataset_info.json`
(or list them under `dataset:` in your training YAML), point
`model_name_or_path` at the §2.2 base model, and run the standard
LLaMA-Factory entrypoint:

```bash
llamafactory-cli train /path/to/your_qwen3vl_sft.yaml
```

See §6 for the exact SFT hyperparameters used in our runs.

> **Skip Stage 1.** The post-SFT checkpoints used in our paper are released
> on HuggingFace; download one of them and proceed directly to §3:
>
> ```bash
> huggingface-cli download prism-vlm/Qwen3-VL-4B-Instruct-SFT \
>     --repo-type model \
>     --local-dir /path/to/models/Qwen3-VL-4B-Instruct-SFT
> # (or prism-vlm/Qwen3-VL-8B-Instruct-SFT for the 8B variant)
> ```

### 3. Stage 2 — PRISM Alignment

Stage 2 takes the post-SFT Qwen3-VL and pulls its on-policy rollouts back to
the supervision distribution via on-policy adversarial distillation against
an **MoE (perception + reasoning) discriminator**. The output is a
**PRISM-aligned policy** that initializes Stage 3.

#### 3.1 Data preparation

Stage 2 consumes the **alignment-stage** corpus — the on-policy prompts used
by `qwen3_vl_prism.sh` for adversarial distillation against the MoE
discriminator. It lives in the shared `prism-vlm/rl_dataset` HuggingFace
dataset repo as `rl_training_data_5.9k.parquet`:

```bash
huggingface-cli download prism-vlm/rl_dataset \
    rl_training_data_5.9k.parquet \
    --repo-type dataset \
    --local-dir /path/to/datasets/prism_rl_dataset
```

#### 3.2 Model preparation

Stage 2 needs two checkpoints: the **post-SFT policy** (from §2 or downloaded
via the "Skip Stage 1" note above) and the **MoE discriminator** (the
adversary).

##### 3.2.a Use the released, pre-warmed MoE discriminator

The merged + warmed-up MoE discriminator from the paper is published on
HuggingFace. Pull it directly and skip §3.2.b:

```bash
huggingface-cli download prism-vlm/Qwen3-VL-2B-4X-Moe-warmup-120k \
    --repo-type model \
    --local-dir /path/to/models/Qwen3-VL-2B-4X-Moe-warmup-120k
```

##### 3.2.b Train the MoE discriminator from scratch

Two steps: (1) sparse-upcycle a dense Qwen3-VL into a 4-expert MoE checkpoint,
and (2) pairwise-warmup that MoE on the teacher / student response corpus.
Both steps are wrapped in `scripts/train/moe_warmup/`.

**(1) Sparse upcycling.** Edit the path block at the top of
`scripts/train/moe_warmup/create_moe.sh` (or override via env vars) so it
points at the dense checkpoint and the desired output directory:

```bash
# inside create_moe.sh
PRISM_ROOT=/path/to/PRISM
DENSE_MODEL=/path/to/models/Qwen3-VL-2B-Instruct
OUTPUT_MOE_DIR=/path/to/models/Qwen3-VL-2B-MoE-4x
NUM_EXPERTS=4
NUM_EXPERTS_PER_TOK=2
```

Then run:

```bash
bash scripts/train/moe_warmup/create_moe.sh
```

This produces a fresh `Qwen3VLMoeForConditionalGeneration` checkpoint at
`OUTPUT_MOE_DIR` (vision encoder + attention + embeddings copied from the
dense model; per-expert MLPs initialized from the dense MLP plus small
gaussian noise; routers randomly initialized).

**(2) Pairwise warmup.** The warmup data — 120K teacher / student response
pairs, one perception (`caption`) and one reasoning (`cot`) comparison per
prompt — is published on HuggingFace:

```bash
huggingface-cli download prism-vlm/qwen3_vl_moe_warmup_pairwise_120k \
    --repo-type dataset \
    --local-dir /path/to/datasets/qwen3_vl_moe_warmup_pairwise_120k
```

Edit the path block at the top of
`scripts/train/moe_warmup/train_moe_warmup.sh`:

```bash
# inside train_moe_warmup.sh
PRISM_ROOT=/path/to/PRISM
MOE_MODEL_PATH=/path/to/models/Qwen3-VL-2B-MoE-4x      # output of step (1)
DATA_PATH=/path/to/datasets/qwen3_vl_moe_warmup_pairwise_120k/warmup_pairwise.jsonl
OUTPUT_DIR=/path/to/models/Qwen3-VL-2B-4X-Moe-warmup-120k
NUM_PROCESSES=8                                          # GPUs on this node
```

Then run (uses `accelerate launch` + DeepSpeed ZeRO-2 under the hood):

```bash
bash scripts/train/moe_warmup/train_moe_warmup.sh
```

The resulting checkpoint at `OUTPUT_DIR` is functionally equivalent to
`prism-vlm/Qwen3-VL-2B-4X-Moe-warmup-120k` and can be plugged into
`critic.model.path` in §3.3.

#### 3.3 Launch alignment

Open `scripts/train/experiment/qwen3_vl_prism.sh` and update the path block at
the top to match what you downloaded:

```bash
# inside qwen3_vl_prism.sh
BASE_DIR=/path/to/PRISM
EXPERIMENT_NAME=qwen3_vl_prism
data.train_files=/path/to/datasets/prism_rl_dataset/rl_training_data_5.9k.parquet
actor_rollout_ref.model.path=/path/to/models/Qwen3-VL-4B-Instruct-SFT
critic.model.path=/path/to/models/Qwen3-VL-2B-4X-Moe-warmup-120k
```

Single-node:

```bash
bash scripts/train/experiment/qwen3_vl_prism.sh
```

Multi-node (driven by `launch.sh`, see §4.4 for the env vars):

```bash
bash scripts/train/experiment/launch.sh \
     scripts/train/experiment/qwen3_vl_prism.sh
```

> **Skip Stage 2.** The post-alignment (PRISM) checkpoint used in our paper
> is released on HuggingFace; download it and proceed directly to §4:
>
> ```bash
> huggingface-cli download prism-vlm/Qwen3-VL-4B-Instruct-SFT-PRISM \
>     --repo-type model \
>     --local-dir /path/to/models/Qwen3-VL-4B-Instruct-SFT-PRISM
> ```

### 4. Stage 3 — RLVR after PRISM

Stage 3 runs **verifiable-reward RL** on top of the PRISM-aligned checkpoint
from Stage 2. Three RL algorithms are supported, each with its own dedicated
script:

| Script                          | Algorithm |
|---------------------------------|-----------|
| `qwen3_vl_grpo_after_prism.sh`  | GRPO      |
| `qwen3_vl_dapo_after_prism.sh`  | DAPO      |
| `qwen3_vl_gspo_after_prism.sh`  | GSPO      |

These are referred to collectively as `qwen3_vl_xxpo_after_prism`, where
`xxpo ∈ {grpo, dapo, gspo}`.

#### 4.1 Data preparation

Stage 3 consumes the **RL training set** with verifiable rewards (final-answer
correctness + format). It lives in the same `prism-vlm/rl_dataset` repo as
Stage 2's data, under the `rl_training_data_filtered_2k.parquet` file:

```bash
huggingface-cli download prism-vlm/rl_dataset \
    rl_training_data_filtered_2k.parquet \
    --repo-type dataset \
    --local-dir /path/to/datasets/prism_rl_dataset
```

#### 4.2 Model preparation

Stage 3 starts from the **PRISM-aligned policy** — either the checkpoint
produced by your own Stage 2 run, or the released checkpoint from the
"Skip Stage 2" note above:

```bash
# Option A: use the released post-alignment checkpoint
huggingface-cli download prism-vlm/Qwen3-VL-4B-Instruct-SFT-PRISM \
    --repo-type model \
    --local-dir /path/to/models/Qwen3-VL-4B-Instruct-SFT-PRISM

# Option B: point at the checkpoint dumped by your own qwen3_vl_prism.sh, e.g.
#   /path/to/PRISM/checkpoints/qwen3_vl_prism/<step>/actor/huggingface
```

Stage 3 does **not** require the MoE discriminator — it is only used by
Stage 2.

#### 4.3 Launch RLVR

Each `qwen3_vl_{grpo,dapo,gspo}_after_prism.sh` opens with the same path block
to update:

```bash
# inside qwen3_vl_xxpo_after_prism.sh
BASE_DIR=/path/to/PRISM
EXPERIMENT_NAME=qwen3_vl_<xxpo>_after_prism
data.train_files=/path/to/datasets/prism_rl_dataset/rl_training_data_filtered_2k.parquet
actor_rollout_ref.model.path=/path/to/models/Qwen3-VL-4B-Instruct-SFT-PRISM
```

Single-node:

```bash
bash scripts/train/experiment/qwen3_vl_grpo_after_prism.sh
bash scripts/train/experiment/qwen3_vl_dapo_after_prism.sh
bash scripts/train/experiment/qwen3_vl_gspo_after_prism.sh
```

Multi-node (any of the three scripts):

```bash
bash scripts/train/experiment/launch.sh \
     scripts/train/experiment/qwen3_vl_grpo_after_prism.sh
```

#### 4.4 Multi-node env vars (shared between Stage 2 and Stage 3)

Multi-node training requires a Ray cluster. `launch.sh` takes care of bringing
the head / worker roles up, waiting for readiness, and then running the
training script on the head while keeping workers alive. Every node runs the
**same** command; role is inferred from `MLP_WORKER_RACK_RANK_INDEX`
(head = 0):

| Variable | Meaning | Example |
|---|---|---|
| `MLP_WORKER_0_HOST`         | Head node IP                    | `192.168.1.100` |
| `MLP_WORKER_0_PRIMARY_HOST` | Head node internal IP           | `192.168.1.100` |
| `MLP_WORKER_NUM`            | Total number of nodes           | `2` |
| `MLP_WORKER_RACK_RANK_INDEX`| Node rank (head = 0)            | `0` or `1` |
| `MLP_WORKER_GPU`            | GPUs per node                   | `8` |

Manual example with two nodes (Stage 2 shown; Stage 3 is identical with the
script swapped):

```bash
# --- head node (NODE_RANK=0) ---
MLP_WORKER_0_HOST=192.168.1.100 MLP_WORKER_0_PRIMARY_HOST=192.168.1.100 \
MLP_WORKER_NUM=2 MLP_WORKER_RACK_RANK_INDEX=0 MLP_WORKER_GPU=8 \
bash scripts/train/experiment/launch.sh \
     scripts/train/experiment/qwen3_vl_prism.sh

# --- worker node (NODE_RANK=1), same command except the rank ---
MLP_WORKER_0_HOST=192.168.1.100 MLP_WORKER_0_PRIMARY_HOST=192.168.1.100 \
MLP_WORKER_NUM=2 MLP_WORKER_RACK_RANK_INDEX=1 MLP_WORKER_GPU=8 \
bash scripts/train/experiment/launch.sh \
     scripts/train/experiment/qwen3_vl_prism.sh
```

`launch.sh` and the training scripts also respect a few env vars for
overriding without editing the script:

```bash
# force single-node mode even if multi-node vars are set
SINGLENODE=true bash scripts/train/experiment/launch.sh \
    scripts/train/experiment/qwen3_vl_prism.sh

# swap data / model without editing the script
BASE_DIR=/path/to/PRISM \
    bash scripts/train/experiment/qwen3_vl_grpo_after_prism.sh
```

### 5. Evaluation

`scripts/eval/` contains a self-contained reference script that evaluates the
trained model with [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval).

Layout:

```
scripts/eval/
├── eval_qwen3vl.sh                       # entrypoint
├── chat_template/
│   └── qwen3vl_bridge_eval.jinja
└── tasks/                                # custom task overrides
    ├── mathvista/ mathvision/ mathverse/
    ├── wemath/    mmmu/       mmmu_pro/
    └── hallusion_bench/
```

The script is self-locating: the chat template and the task overrides are
resolved relative to the script itself via `$SCRIPT_DIR`, so you only need to
fill in the **external** paths at the top of the file:
`LMMS_EVAL_ROOT`, `ENV_PATH`, `HF_CACHE`, `MODEL`, `JUDGE_MODEL`.

Task list (vLLM data-parallel on GPUs 1-7, local vLLM judge on GPU 0):

```
mathvista_testmini, mathvision_testmini, mathverse_testmini,
wemath_testmini_reasoning, mmmu_val, mmmu_pro, hallusion_bench_image
```

Run:

```bash
bash scripts/eval/eval_qwen3vl.sh
# or on SLURM:
sbatch scripts/eval/eval_qwen3vl.sh
```

Our task overrides under `scripts/eval/tasks/` are registered via
`--include_path`, so **no patching of the upstream `lmms-eval` repo is
required**.

### 6. Reproduction hyperparameters

We reproduce the full set of hyperparameters used in the paper. The SFT stage
is run with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); the
alignment and RLVR stages are run with the bundled `verl`.

| Component                | SFT (LLaMA-Factory) | PRISM Alignment (verl) | RLVR (verl)        |
|--------------------------|---------------------|------------------------|--------------------|
| Optimizer                | AdamW               | AdamW                  | AdamW              |
| Scheduler                | cosine              | constant               | constant           |
| Learning rate            | 1e-5                | 1e-6                   | 1e-6               |
| Weight decay             | -                   | 0.01                   | 0.01               |
| Epochs / Steps           | 1 epoch             | 500 steps              | 1500 steps         |
| Warmup ratio / steps     | 0.1                 | -                      | -                  |
| Global batch size        | 2                   | 4                      | 32                 |
| Max prompt length        | -                   | 2048                   | 2048               |
| Max response length      | 8192                | 6144                   | 8192               |
| Rollout temperature      | -                   | 1.0                    | 1.0                |
| Rollout group size N     | -                   | 16                     | 16                 |
| α (MoE expert weight)    | -                   | 0.5                    | -                  |
| Accuracy reward weight   | -                   | -                      | 0.8                |
| Format reward weight     | -                   | -                      | 0.2                |
| Dynamic batch size       | -                   | True                   | True               |
| Remove padding           | -                   | True                   | True               |
| KL regularization        | -                   | 0.0 (disabled)         | per-algorithm default |
| Hardware                 | 8 × H100-80GB       | 8 × H100-80GB          | 8 × H100-80GB      |

> See `scripts/train/experiment/qwen3_vl_prism.sh` and the `qwen3_vl_*_after_prism.sh`
> scripts for the exact `verl` config used in our runs. SFT-stage hyperparameters
> can be plugged into LLaMA-Factory's standard YAML config.

### 7. Repo structure

```
PRISM/
├── scripts/
│   ├── train/experiment/   # Stage 2 / Stage 3 training entrypoints + launch.sh
│   ├── train/moe_warmup/   # Stage 2 (optional): MoE discriminator from scratch
│   └── eval/               # lmms-eval reference + custom tasks
├── verl/                   # verl framework (GRPO / DAPO / GSPO recipes)
├── transformers-4.57.0/    # patched transformers (editable install)
├── moe/                    # MoE upcycling + warmup implementations
├── tools/                  # misc helpers
└── difference/             # diff reports vs. upstream verl / transformers (CN + EN)
```

---

## Local modifications

We document every non-trivial change made on top of the upstream `verl` and
`transformers` releases that this repo vendors. Reports are provided in both
Chinese and English under [`difference/`](difference/):

| Topic | Chinese | English |
|---|---|---|
| `transformers-4.57.0` (Qwen-related changes: new `qwen2_vl_moe`, `qwen2_5_vl_moe`, and `Qwen3VLMoeForTokenClassification`) | [`transformers_diff_CN.md`](difference/transformers_diff_CN.md) | [`transformers_diff_EN.md`](difference/transformers_diff_EN.md) |
| `verl` (PRISM critic + reward_score additions for the mm_gad pipeline) | [`verl_diff_CN.md`](difference/verl_diff_CN.md) | [`verl_diff_EN.md`](difference/verl_diff_EN.md) |

Each report includes a per-file diff summary, the rationale for the change,
and any caveats relevant to reproducing or extending PRISM.

---

## Citation

If you find this project helpful, please consider giving us a star and citing
our paper with:

```bibtex
% TODO: citation will be added once the paper is released.
```

---

## Acknowledgements

We gratefully acknowledge the following open-source projects that made this
work possible. They form the backbone of the three building blocks of PRISM,
namely SFT, RL, and evaluation:

- [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory): used for the
  Stage-1 **SFT** cold-start training.
- [**verl**](https://github.com/volcengine/verl): used for the Stage-2
  **alignment** (adversarial on-policy distillation) and the Stage-3 **RLVR**
  (GRPO / DAPO / GSPO) training.
- [**lmms-eval**](https://github.com/EvolvingLMMs-Lab/lmms-eval): used as the
  **evaluation** framework for all multimodal benchmarks reported in the paper.

We thank the developers and contributors of these projects for their excellent
work and for making their code publicly available.

We warmly welcome contributions from the community to help improve and
stabilize the latest `verl` integration. Furthermore, we deeply appreciate your
feedback and support in any form, whether it be reporting issues, submitting
pull requests, or providing fixes.

---

## License

This project is released under the [MIT License](LICENSE).

Note: this repository vendors modified copies of two upstream projects that
keep their original Apache-2.0 licenses:

- `verl/` is a modified copy of [verl](https://github.com/volcengine/verl)
  (Apache-2.0). See [`verl/LICENSE`](verl/LICENSE).
- `transformers-4.57.0/` is a modified copy of
  [transformers](https://github.com/huggingface/transformers) (Apache-2.0).
  See [`transformers-4.57.0/LICENSE`](transformers-4.57.0/LICENSE).
