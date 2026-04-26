#!/bin/bash
# Dense VL Discriminator Warmup Training
# 使用 Qwen3-VL-8B dense 模型进行 pairwise warmup 训练

# 单卡训练
# python train_dense_vl_warmup.py

# 多卡训练 (例如 4 卡)
accelerate launch --num_processes 4 train_dense_vl_warmup.py
