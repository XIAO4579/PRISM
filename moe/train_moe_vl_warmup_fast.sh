#!/bin/bash
# Internal launcher for moe/train_moe_vl_warmup_fast.py.
#
# Expects the caller to have already exported:
#   - PYTHONPATH (with PRISM/transformers-4.57.0/src on it)
#   - MOE_MODEL_PATH / PROCESSOR_PATH / DATA_PATH / OUTPUT_DIR
#   - (optional) IMAGE_BASE_PATH, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, ...
#
# The user-facing entrypoint is scripts/train/moe_warmup/train_moe_warmup.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_z2_config.json" \
    "${SCRIPT_DIR}/train_moe_vl_warmup_fast.py"
