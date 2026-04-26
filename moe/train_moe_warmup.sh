#!/bin/bash
# Text-only MoE warmup launcher (Qwen3 dense -> MoE upcycled).
# Set PRISM_ROOT to the repo root and have the patched transformers on
# PYTHONPATH; expects accelerate / deepspeed to be installed.

set -euo pipefail

PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"
export PYTHONPATH="${PRISM_ROOT}/transformers-4.57.0/src:${PYTHONPATH:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_z2_config.json" \
    "${SCRIPT_DIR}/train_moe_warmup.py"
