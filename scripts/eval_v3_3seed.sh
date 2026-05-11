#!/bin/bash
cd /home/introai26/.agile/user/hsjung/DxMI_Medical

python scripts/eval_test.py \
    --config configs/ebm_fm_gate_v3.yaml \
             configs/ebm_fm_gate_v3_s2.yaml \
             configs/ebm_fm_gate_v3_s3.yaml \
    --ckpt outputs/ebm_fm_gate_v3_20260423/ckpt_best_val.pt \
           outputs/ebm_fm_gate_v3_s2_20260425/ckpt_best_val.pt \
           outputs/ebm_fm_gate_v3_s3_20260426/ckpt_best_val.pt
