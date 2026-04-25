#!/bin/bash
#SBATCH -p acd_u
#SBATCH -n 64
#SBATCH --mem=960G
#SBATCH --gres=gpu:8
#SBATCH -o /data/user/swang886/gad_project/mm_gad/moe/dense_model/logs/dense_vl_warmup_%j.log
#SBATCH -e /data/user/swang886/gad_project/mm_gad/moe/dense_model/logs/dense_vl_warmup_%j.err
#SBATCH -J dense_warmup_4B
set -x
set -o pipefail

# ===================== 环境 =====================
source /data/user/swang886/llama_factory_env/bin/activate

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export XDG_DATA_HOME=/data/user/swang886/.cache
export HF_HOME=/data/user/swang886/.cache/huggingface
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Debug: 限制 warmup 读取的数据条数（可在提交时覆盖）
export WARMUP_MAX_RECORDS=${WARMUP_MAX_RECORDS:-100}

# ===================== 训练 =====================
cd /data/user/swang886/gad_project/mm_gad/moe/dense_model

mkdir -p /data/user/swang886/gad_project/mm_gad/moe/dense_model/logs

accelerate launch \
    --num_processes 8 \
    --use_deepspeed \
    --deepspeed_config_file /data/user/swang886/gad_project/mm_gad/moe/dense_model/ds_z2_config.json \
    --mixed_precision bf16 \
    train_dense_vl_warmup.py \
    2>&1 | tee /data/user/swang886/gad_project/mm_gad/logs/dense_vl_warmup_$(date +%Y%m%d_%H%M%S).log
