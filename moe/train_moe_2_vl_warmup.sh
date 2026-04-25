#!/bin/bash
export PYTHONPATH="/data/home/scwb352/run/test/mm_gad/transformers-4.57.0/src:${PYTHONPATH}" 

accelerate launch --num_processes=8 /data/home/scwb352/run/test/dev/mm_gad/moe/train_moe_2_vl_warmup.py
