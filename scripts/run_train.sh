#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# DxMI Medical 학습 실행 스크립트
# PBS GPU 노드 진입 후 이 스크립트를 실행:
#   bash scripts/run_train.sh [ablation] [ngpus] [resume_ckpt]
#
# 예시:
#   bash scripts/run_train.sh A 1
#   bash scripts/run_train.sh B 1
#   bash scripts/run_train.sh C 2
#   bash scripts/run_train.sh A 1 outputs/ebm_baseline/ckpt_epoch0100.pt
#
# PBS 잡 진입 (coss_agpu, A100-80GB):
#   /opt/pbs/bin/qsub -I -q coss_agpu \
#       -l select=1:ncpus=6:mem=192g:ngpus=1 -l walltime=72:00:00
# ─────────────────────────────────────────────────────────────────────────────

set -e

ABLATION=${1:-A}
NGPUS=${2:-1}
RESUME=${3:-""}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="dxmi_medical"
PYTHON="/home/introai26/miniconda3/envs/${CONDA_ENV}/bin/python"

# conda 환경 확인
if [ ! -f "${PYTHON}" ]; then
    echo "[ERROR] Python not found: ${PYTHON}"
    exit 1
fi

cd "${PROJECT_DIR}"
echo "[INFO] Working dir: ${PROJECT_DIR}"
echo "[INFO] Ablation: ${ABLATION}, GPUs: ${NGPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# Ablation → config 매핑
case "${ABLATION}" in
    A) CONFIG="configs/ebm_baseline.yaml"     ;;
    B) CONFIG="configs/supervised_reward.yaml";;
    C) CONFIG="configs/irl_maxent.yaml"        ;;
    *) echo "[ERROR] Unknown ablation '${ABLATION}'. Use A, B, or C."; exit 1 ;;
esac

echo "[INFO] Config: ${CONFIG}"

# Resume 인수 구성
RESUME_ARG=""
if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume ${RESUME}"
    echo "[INFO] Resuming from: ${RESUME}"
fi

# 단일 GPU vs DDP
if [ "${NGPUS}" -le 1 ]; then
    echo "[INFO] Single GPU training"
    "${PYTHON}" scripts/train.py \
        --config "${CONFIG}" \
        ${RESUME_ARG}
else
    echo "[INFO] DDP training with ${NGPUS} GPUs"
    "${PYTHON}" -m torch.distributed.run \
        --nproc_per_node="${NGPUS}" \
        --master_port=29500 \
        scripts/train.py \
        --config "${CONFIG}" \
        ${RESUME_ARG}
fi
