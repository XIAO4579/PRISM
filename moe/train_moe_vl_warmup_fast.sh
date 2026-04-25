#!/bin/bash
export PYTHONPATH="/data/home/scwb352/run/test/mm_gad/transformers-4.57.0/src:${PYTHONPATH}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_z2_config.json" \
    "${SCRIPT_DIR}/train_moe_vl_warmup_fast.py"
