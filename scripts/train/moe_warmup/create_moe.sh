#!/bin/bash
# Step 1 of "MoE discriminator from scratch":
#   Sparse-upcycle a dense Qwen3-VL checkpoint into a Qwen3VLMoe checkpoint.
#
# Edit the path block below (or override via env vars) to point at your local
# layout, then run:
#     bash scripts/train/moe_warmup/create_moe.sh

set -euo pipefail

# ---- paths -------------------------------------------------------------
PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"

# Dense Qwen3-VL checkpoint to upcycle (HF id or local dir).
DENSE_MODEL="${DENSE_MODEL:-/path/to/models/Qwen3-VL-2B-Instruct}"

# Where to write the freshly upcycled MoE checkpoint.
OUTPUT_MOE_DIR="${OUTPUT_MOE_DIR:-/path/to/models/Qwen3-VL-2B-MoE-4x}"

# ---- hyper-parameters --------------------------------------------------
NUM_EXPERTS="${NUM_EXPERTS:-4}"
NUM_EXPERTS_PER_TOK="${NUM_EXPERTS_PER_TOK:-2}"
NOISE_STD="${NOISE_STD:-0.01}"

# ---- env ---------------------------------------------------------------
export PYTHONPATH="${PRISM_ROOT}/transformers-4.57.0/src:${PYTHONPATH:-}"

# ---- launch ------------------------------------------------------------
python3 "${PRISM_ROOT}/moe/create_vl_moe.py" \
    --dense-model "${DENSE_MODEL}" \
    --save-path "${OUTPUT_MOE_DIR}" \
    --num-experts "${NUM_EXPERTS}" \
    --num-experts-per-tok "${NUM_EXPERTS_PER_TOK}" \
    --noise-std "${NOISE_STD}" \
    "$@"
