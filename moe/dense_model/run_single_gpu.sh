#!/bin/bash
# Single-GPU launcher for the dense Qwen3-VL warmup baseline (debugging).

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train_dense_vl_warmup.py
