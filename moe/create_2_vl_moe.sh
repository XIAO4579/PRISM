#!/bin/bash
# Sparse-upcycle a dense Qwen2-VL checkpoint into a Qwen2VLMoe checkpoint.
# Set PRISM_ROOT to the repo root before running.

set -euo pipefail

PRISM_ROOT="${PRISM_ROOT:-/path/to/PRISM}"
export PYTHONPATH="${PRISM_ROOT}/transformers-4.57.0/src:${PYTHONPATH:-}"

python "${PRISM_ROOT}/moe/create_2_vl_moe.py"
