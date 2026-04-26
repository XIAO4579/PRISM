#!/bin/bash
# 8-GPU launcher for the dense (non-MoE) Qwen3-VL warmup baseline.
# Set PRISM_ROOT to the repo root before running.

set -euxo pipefail

PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Debug: cap how many lines warmup reads from DATA_PATH (override at submit time).
export WARMUP_MAX_RECORDS="${WARMUP_MAX_RECORDS:-100}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

cd "${SCRIPT_DIR}"

accelerate launch \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_z2_config.json" \
    --mixed_precision bf16 \
    train_dense_vl_warmup.py \
    2>&1 | tee "${LOG_DIR}/dense_vl_warmup_$(date +%Y%m%d_%H%M%S).log"
