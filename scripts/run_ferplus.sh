#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python trainAugcomp.py \
    --dataset_name 'ferplus' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name ferplus \
    --eta_prime 2 \
    --worst_weight 0.2 \
    --cluster_epochs 50 \
    --cluser 0.2 \
    --LSR_weight 0.3

