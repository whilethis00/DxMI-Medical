#!/bin/bash
#PBS -N dxmi_v3_seed2
#PBS -q coss_agpu
#PBS -l select=1:ncpus=6:mem=192g:ngpus=2:Qlist=agpu:container_engine=singularity
#PBS -l walltime=72:00:00
#PBS -o /home/introai26/.agile/user/hsjung/DxMI_Medical/outputs/pbs_logs/v3_seed2.out
#PBS -e /home/introai26/.agile/user/hsjung/DxMI_Medical/outputs/pbs_logs/v3_seed2.err

set -e

PROJECT_DIR="/home/introai26/.agile/user/hsjung/DxMI_Medical"
PYTHON="/home/introai26/miniconda3/envs/dxmi_medical/bin/python"

mkdir -p "${PROJECT_DIR}/outputs/pbs_logs"

echo "=============================="
echo "Job ID: $PBS_JOBID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"
echo "=============================="

cd "$PROJECT_DIR"

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ebm_fm_gate_v3_s2.yaml \
    --seed 2

echo "=============================="
echo "Done: $(date)"
echo "=============================="
