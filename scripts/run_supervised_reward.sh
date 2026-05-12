#!/bin/bash
# Strong supervised variance-regression baseline.
#
# Usage:
#   bash scripts/run_supervised_reward.sh [seed] [ngpus]
#
# Examples:
#   bash scripts/run_supervised_reward.sh
#   bash scripts/run_supervised_reward.sh 42 1
#   bash scripts/run_supervised_reward.sh 42 2

set -euo pipefail

SEED="${1:-42}"
NGPUS="${2:-1}"

PROJECT_DIR="/home/introai26/.agile/user/hsjung/DxMI_Medical"
PYTHON="/home/introai26/miniconda3/envs/dxmi_medical/bin/python"
CONFIG="configs/supervised_reward.yaml"
DATE_STR="$(date +%Y%m%d)"
OUT_DIR="outputs/supervised_reward_${DATE_STR}"
CKPT="${OUT_DIR}/ckpt_best_val.pt"

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "[ERROR] Python not found or not executable: ${PYTHON}"
    exit 1
fi

echo "[INFO] Project: ${PROJECT_DIR}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] GPUs: ${NGPUS}"
echo "[INFO] Expected output dir: ${OUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ "${NGPUS}" -le 1 ]; then
    "${PYTHON}" scripts/train.py \
        --config "${CONFIG}" \
        --seed "${SEED}"
else
    "${PYTHON}" -m torch.distributed.run \
        --nproc_per_node="${NGPUS}" \
        --master_port="$((29600 + SEED))" \
        scripts/train.py \
        --config "${CONFIG}" \
        --seed "${SEED}"
fi

if [ ! -f "${CKPT}" ]; then
    BEST_EPOCH="$(
        grep -E 'best so far: .* @ ep[0-9]+' "${OUT_DIR}/train.log" \
            | tail -n 1 \
            | sed -E 's/.* @ ep([0-9]+).*/\1/'
    )"
    if [ -n "${BEST_EPOCH}" ]; then
        CKPT="${OUT_DIR}/ckpt_epoch$(printf '%04d' "${BEST_EPOCH}").pt"
        echo "[WARN] ckpt_best_val.pt not found; falling back to parsed best epoch: ${CKPT}"
    fi
fi

if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] No evaluation checkpoint found."
    echo "[INFO] Expected best ckpt: ${OUT_DIR}/ckpt_best_val.pt"
    echo "[INFO] Existing supervised_reward checkpoints:"
    find "${OUT_DIR}" -maxdepth 1 -type f -name 'ckpt_epoch*.pt' -print | sort
    exit 1
fi

echo "[INFO] Running one-time test evaluation: ${CKPT}"
"${PYTHON}" scripts/eval_test.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --num-workers 0 \
    | tee "${OUT_DIR}/test.log"

echo "[DONE] Supervised reward baseline complete."
echo "[DONE] Output dir: ${OUT_DIR}"
