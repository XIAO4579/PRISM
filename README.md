# PRISM: PRe-alignment via on-policy dIStillation for Multimodal RL

<div align="center">

[![arXiv](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](#)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](#)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff)](#)
[![Models](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff)](#)

</div>

---

## 🎉 News

- **[2026-04-25]** We release the **code**, **data**, and **model checkpoints** of PRISM. Check out the [PRISM Collection](https://github.com/XIAO4579/PRISM).

---

## Table of Contents

- [News](#-news)
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [1. Environment setup](#1-environment-setup)
  - [2. Data & model preparation](#2-data--model-preparation)
  - [3. Training](#3-training)
  - [4. Evaluation](#4-evaluation)
  - [5. Reproduction hyperparameters](#5-reproduction-hyperparameters)
  - [6. Repo structure](#6-repo-structure)
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

### 2. Data & model preparation

#### 2.1 Authenticate with HuggingFace

Before downloading private assets, log in once so the token is stored in
`~/.cache/huggingface/` and picked up automatically:

```bash
huggingface-cli login           # paste your token when prompted
# or:  export HF_TOKEN=<YOUR_HF_TOKEN>
```

> Do **not** pass `--token` on the command line, since tokens leak into shell
> history and job logs.

#### 2.2 Download the distilled model (for the RL-after-PRISM stage)

```bash
huggingface-cli download \
    xiao45791/Qwen3-VL-4B-Instruct-Gemini-Distill-method2-stage1-500steps-woKL \
    --repo-type model \
    --local-dir /path/to/models/Qwen3-VL-4B-Instruct-Gemini-Distill-method2-stage1-500steps-woKL
```

#### 2.3 Download the RL training data

```bash
huggingface-cli download xiao45791/rl_dataset \
    --repo-type dataset \
    --local-dir /path/to/datasets/rl_dataset
```

Then update the `data.train_files`, `actor_rollout_ref.model.path`, etc. inside
the training scripts (or export `BASE_DIR` / env vars; see §3.3).

### 3. Training

All training entrypoints live in `scripts/train/experiment/`:

| Script | Stage | Algorithm |
|---|---|---|
| `qwen3_vl_prism.sh`            | Stage 1: PRISM distillation    | GRPO w/ multi-reward + critic |
| `qwen3_vl_grpo_after_prism.sh` | Stage 2: RL on distilled model | GRPO |
| `qwen3_vl_dapo_after_prism.sh` | Stage 2: RL on distilled model | DAPO  |
| `qwen3_vl_gspo_after_prism.sh` | Stage 2: RL on distilled model | GSPO  |

Every script begins with a short block of path variables
(`/path/to/...`) and a `BASE_DIR` / `EXPERIMENT_NAME` pair. Update those to
point at your local layout.

#### 3.1 Single-node

A single node runs with a plain `bash` invocation, and verl starts a local Ray
cluster automatically:

```bash
bash scripts/train/experiment/qwen3_vl_prism.sh
bash scripts/train/experiment/qwen3_vl_grpo_after_prism.sh
bash scripts/train/experiment/qwen3_vl_dapo_after_prism.sh
bash scripts/train/experiment/qwen3_vl_gspo_after_prism.sh
```

#### 3.2 Multi-node

Multi-node training requires a Ray cluster. `launch.sh` takes care of bringing
the head / worker roles up, waiting for readiness, and then running the
training script on the head while keeping workers alive.

Every node runs the **same** command; role is inferred from
`MLP_WORKER_RACK_RANK_INDEX` (head = 0):

```bash
bash scripts/train/experiment/launch.sh \
     scripts/train/experiment/qwen3_vl_prism.sh
```

The required environment variables are usually injected by the job scheduler,
but can also be set by hand:

| Variable | Meaning | Example |
|---|---|---|
| `MLP_WORKER_0_HOST`         | Head node IP                    | `192.168.1.100` |
| `MLP_WORKER_0_PRIMARY_HOST` | Head node internal IP           | `192.168.1.100` |
| `MLP_WORKER_NUM`            | Total number of nodes           | `2` |
| `MLP_WORKER_RACK_RANK_INDEX`| Node rank (head = 0)            | `0` or `1` |
| `MLP_WORKER_GPU`            | GPUs per node                   | `8` |

Manual example with two nodes:

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

#### 3.3 Overriding without editing the script

`launch.sh` and the training scripts respect a few env vars:

```bash
# force single-node mode even if multi-node vars are set
SINGLENODE=true bash scripts/train/experiment/launch.sh \
    scripts/train/experiment/qwen3_vl_prism.sh

# swap data / model without editing the script
BASE_DIR=/path/to/PRISM \
    bash scripts/train/experiment/qwen3_vl_grpo_after_prism.sh
```

### 4. Evaluation

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

### 5. Reproduction hyperparameters

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

### 6. Repo structure

```
PRISM/
├── scripts/
│   ├── train/experiment/   # training entrypoints + launch.sh
│   └── eval/               # lmms-eval reference + custom tasks
├── verl/                   # verl framework (GRPO / DAPO / GSPO recipes)
├── transformers-4.57.0/    # patched transformers (editable install)
├── moe/                    # MoE modules used by the training recipes
├── tools/                  # misc helpers
└── difference.md           # notes on local modifications vs. upstream
```

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
