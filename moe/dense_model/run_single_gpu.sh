#!/bin/bash
source /data/user/swang886/verl_0.6_env/bin/activate

export CUDA_VISIBLE_DEVICES=0

cd /data/user/swang886/gad_project/mm_gad/moe/dense_model

accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train_dense_vl_warmup.py
