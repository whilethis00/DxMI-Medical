"""
평가 메트릭
- ECE  (Expected Calibration Error): energy ↔ uncertainty 보정
- AUROC: 악성/양성 분류 (malignancy_mean ≥ 3 = positive)
- Spearman ρ: energy ↔ annotator disagreement (Var) 상관관계
              Clinical 합격 조건: p < 0.05
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass


# ── Spearman ρ (핵심 임상 지표) ────────────────────────────────────────────────

def spearman_energy_disagreement(
    energies: np.ndarray,
    mal_vars: np.ndarray,
) -> tuple[float, float]:
    """
    Spearman ρ(energy, malignancy_variance).
    에너지가 높을수록 판독 불일치가 커야 함.

    Returns:
        rho: Spearman 상관계수
        p:   p-value (< 0.05 이면 임상 합격)
    """
    rho, p = spearmanr(energies, mal_vars)
    return float(rho), float(p)


# ── AUROC (악성 분류) ──────────────────────────────────────────────────────────

def auroc_malignancy(
    energies: np.ndarray,
    mal_means: np.ndarray,
    threshold: float = 3.0,
) -> float:
    """
    AUROC: score = -energy (낮은 에너지 = 높은 확신 = 더 명확한 양성/음성).
    Labels: malignancy_mean >= threshold → 1 (악성)

    Args:
        energies:  (N,) EBM energy 값
        mal_means: (N,) malignancy score 평균
        threshold: 악성 기준 (기본 3.0)
    """
    labels = (mal_means >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    scores = -energies  # 낮은 에너지 = 더 확실 = 높은 score
    return float(roc_auc_score(labels, scores))


# ── ECE (Calibration) ─────────────────────────────────────────────────────────

def expected_calibration_error(
    energies: np.ndarray,
    mal_vars: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: energy를 uncertainty proxy로 보고 실제 disagreement(var)와 보정 평가.

    에너지 → sigmoid로 [0,1] uncertainty 변환 후 bin별 calibration 측정.

    Args:
        energies:  (N,) EBM energy
        mal_vars:  (N,) malignancy variance (실제 불확실성)
        n_bins:    bin 수
    """
    # energy → uncertainty probability (sigmoid)
    unc_pred = 1 / (1 + np.exp(-energies))  # sigmoid
    # mal_var → binary: var > 0 이면 disagreement 있음
    unc_true = (mal_vars > 0).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    N    = len(energies)

    for i in range(n_bins):
        mask = (unc_pred >= bins[i]) & (unc_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = unc_true[mask].mean()
        conf = unc_pred[mask].mean()
        ece += mask.sum() / N * abs(acc - conf)

    return float(ece)


# ── 통합 평가 함수 ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    rho: float
    p_value: float
    auroc: float
    ece: float
    n_samples: int

    def passed_clinical(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:
        clinical = "PASS" if self.passed_clinical() else "FAIL"
        return (
            f"N={self.n_samples} | "
            f"Spearman ρ={self.rho:.4f} (p={self.p_value:.4f}, {clinical}) | "
            f"AUROC={self.auroc:.4f} | ECE={self.ece:.4f}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> EvalResult:
    """
    DataLoader에서 배치를 받아 에너지 추출 후 전체 메트릭 계산.

    Args:
        model:      EBM (forward: patch → energy)
        dataloader: val 또는 test DataLoader
        device:     torch device
    """
    model.eval()
    all_energies  = []
    all_mal_means = []
    all_mal_vars  = []

    for batch in dataloader:
        x        = batch["patch"].to(device)
        energy   = model(x).cpu().numpy()
        mal_mean = batch["malignancy_mean"].numpy()
        mal_var  = batch["malignancy_var"].numpy()

        all_energies.append(energy)
        all_mal_means.append(mal_mean)
        all_mal_vars.append(mal_var)

    energies  = np.concatenate(all_energies)
    mal_means = np.concatenate(all_mal_means)
    mal_vars  = np.concatenate(all_mal_vars)

    rho, p   = spearman_energy_disagreement(energies, mal_vars)
    auroc    = auroc_malignancy(energies, mal_means)
    ece      = expected_calibration_error(energies, mal_vars)

    return EvalResult(
        rho=rho, p_value=p, auroc=auroc, ece=ece, n_samples=len(energies)
    )
