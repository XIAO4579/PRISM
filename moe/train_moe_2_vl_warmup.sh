#!/bin/bash
# Pairwise warmup for the upcycled Qwen2-VL MoE discriminator.
# Set PRISM_ROOT to the repo root before running.

set -euo pipefail

PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"
export PYTHONPATH="${PRISM_ROOT}/moe:${PRISM_ROOT}/transformers-4.57.0/src:${PYTHONPATH:-}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch --num_processes="${NUM_PROCESSES}" \
    "${PRISM_ROOT}/moe/train_moe_2_vl_warmup.py"
