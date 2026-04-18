#!/bin/bash
set -e
PYTHON=/home/introai26/miniconda3/envs/dxmi_medical/bin/python

for seed in 1 2; do
    echo "=== repro seed=$seed ==="
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        --master_port=$((29500 + seed)) \
        scripts/train.py \
        --config configs/ebm_weighted_cd_r${seed}.yaml \
        --seed $seed
done
