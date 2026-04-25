#!/bin/bash
#SBATCH -p acd_u
#SBATCH -n 32
#SBATCH --mem=960G
#SBATCH --gres=gpu:8
#SBATCH -o /data/user/swang886/gad_project/mm_gad/logs/moe-warmup_text_%j.out
#SBATCH -e /data/user/swang886/gad_project/mm_gad/logs/moe-warmup_text_%j.err
#SBATCH -J moe-warmup-text

source /data/user/swang886/llama_factory_env/bin/activate
cd /data/user/swang886/gad_project/mm_gad

# unset ROCR_VISIBLE_DEVICES
# unset CC CXX CUDAHOSTCXX

# set -x  
export PYTHONPATH="/data/user/swang886/gad_project/mm_gad/transformers-4.57.0/src:${PYTHONPATH}"

SCRIPT_DIR="/data/user/swang886/gad_project/mm_gad/moe"

accelerate launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/ds_z2_config.json" \
    "${SCRIPT_DIR}/train_moe_warmup.py"
