#!/bin/bash
# Step 2 of "MoE discriminator from scratch":
#   Pairwise warmup of the upcycled Qwen3-VL MoE discriminator on the
#   teacher/student-response dataset (DeepSpeed ZeRO-2 + dynamic padding).
#
# Edit the path block below (or override via env vars) to point at your local
# layout, then run:
#     bash scripts/train/moe_warmup/train_moe_warmup.sh

set -euo pipefail

# ---- paths -------------------------------------------------------------
PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"

# MoE checkpoint to warm up (output of step 1, or any pre-upcycled checkpoint).
export MOE_MODEL_PATH="${MOE_MODEL_PATH:-/path/to/models/Qwen3-VL-2B-MoE-4x}"
export PROCESSOR_PATH="${PROCESSOR_PATH:-${MOE_MODEL_PATH}}"

# Pairwise warmup data (teacher / student responses with the prism schema).
export DATA_PATH="${DATA_PATH:-/path/to/datasets/qwen3_vl_moe_warmup_pairwise_120k.jsonl}"

# Optional: prefix for relative image paths inside DATA_PATH (leave empty if
# the JSONL already contains absolute paths or URLs).
export IMAGE_BASE_PATH="${IMAGE_BASE_PATH:-}"

# Where to save the warmed-up MoE discriminator.
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/models/Qwen3-VL-2B-4X-Moe-warmup}"

# ---- hyper-parameters (forwarded to train_moe_vl_warmup_fast.py) -------
export MAX_LENGTH="${MAX_LENGTH:-8192}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export SEED="${SEED:-42}"

# ---- distributed -------------------------------------------------------
export NUM_PROCESSES="${NUM_PROCESSES:-8}"

# ---- env ---------------------------------------------------------------
export PYTHONPATH="${PRISM_ROOT}/moe:${PRISM_ROOT}/transformers-4.57.0/src:${PYTHONPATH:-}"

# ---- launch ------------------------------------------------------------
bash "${PRISM_ROOT}/moe/train_moe_vl_warmup_fast.sh"
