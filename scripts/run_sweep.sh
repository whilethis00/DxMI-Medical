#!/bin/bash
cd /home/introai26/.agile/user/hsjung/DxMI_Medical
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py --config configs/ebm_collapse_sweep.yaml
