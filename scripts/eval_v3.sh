#!/bin/bash
cd /home/introai26/.agile/user/hsjung/DxMI_Medical

python scripts/eval_test.py \
    --config configs/ebm_fm_gate_v1.yaml \
             configs/ebm_fm_gate_v2.yaml \
             configs/ebm_fm_gate_v3.yaml \
    --ckpt outputs/ebm_fm_gate_v1_20260421/ckpt_best_val.pt \
           outputs/ebm_fm_gate_v2_20260422/ckpt_best_val.pt \
           outputs/ebm_fm_gate_v3_20260423/ckpt_best_val.pt
