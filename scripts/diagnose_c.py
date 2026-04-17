"""
Step 2: 훈련된 C checkpoint 분리 난이도 진단

질문: "C가 배우는 negative가 너무 쉬워서 ranking task가 binary separation으로 붕괴하느냐"

계측 항목:
  1. rollout() clamp hit ratio — trained VF 기준
  2. pos(demo) vs neg(policy sample) energy overlap — overlap이 0이면 trivial separation
  3. policy negative separation hardness — separation / std ratio
  4. energy rank correlation 내 분산 — ranking이 살아있는지

Usage:
    cd /home/introai26/.agile/user/hsjung/DxMI_Medical
    /home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/diagnose_c.py
"""

import sys
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ebm import EBM
from src.models.flow_matching import VelocityField, rollout
from src.data.dataset import LIDCDataset

DEVICE = torch.device("cpu")
VAL_CSV  = ROOT / "data/splits/val.csv"
CKPT_C   = ROOT / "outputs/irl_maxent/ckpt_epoch0180.pt"
N_ROLLOUT_STEPS = 8    # CPU OOM 방지 — 경향 파악용
BATCH_SIZE = 1         # base_ch=32 VF + 48³은 CPU에서 배치 1로 제한
N_ENERGY_BATCHES = 20  # val 전체 대신 20배치(20샘플)만 사용


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_c_models():
    ck = torch.load(CKPT_C, map_location="cpu")
    print(f"Loaded checkpoint: epoch {ck['epoch']}")

    ebm = EBM(base_ch=32)
    vf  = VelocityField(base_ch=32)

    for name, model, key in [("EBM", ebm, "ebm"), ("VF", vf, "vf")]:
        state = ck[key]
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        try:
            model.load_state_dict(state)
            print(f"  {name}: loaded OK")
        except RuntimeError as e:
            print(f"  {name}: load failed — {e}")
            raise

    return ebm.to(DEVICE).eval(), vf.to(DEVICE).eval()


# ── 1. clamp hit ratio ────────────────────────────────────────────────────────

def measure_clamp_ratio(vf: VelocityField, n_samples: int = 64) -> dict:
    """rollout 전체 경로에서 clamp에 걸린 비율 계측."""
    x = torch.randn(n_samples, 1, 48, 48, 48, device=DEVICE)
    dt = 1.0 / N_ROLLOUT_STEPS
    clamp_counts = []

    with torch.no_grad():
        for i in range(N_ROLLOUT_STEPS):
            t_val = i * dt
            t = torch.full((n_samples,), t_val, device=DEVICE)
            v = vf(x, t)
            x_next = x + v * dt
            hit = ((x_next < 0.0) | (x_next > 1.0)).float().mean().item()
            clamp_counts.append(hit)
            x = x_next.clamp(0.0, 1.0)

    x_final = x
    final_hit = ((x_final <= 0.0) | (x_final >= 1.0)).float().mean().item()

    return {
        "clamp_rate_final": final_hit,
        "clamp_rate_mean_over_steps": float(np.mean(clamp_counts)),
        "clamp_rate_max_step": float(np.max(clamp_counts)),
        "x_final_min": x_final.min().item(),
        "x_final_max": x_final.max().item(),
        "x_final_mean": x_final.mean().item(),
        "x_final_std": x_final.std().item(),
    }


# ── 2. pos vs neg energy overlap ──────────────────────────────────────────────

def measure_energy_overlap(ebm: EBM, vf: VelocityField, val_loader, max_batches: int = 999) -> dict:
    """
    demo(real patch) vs policy sample energy 분포 비교.
    overlap coefficient: 두 분포가 얼마나 겹치는지 (0=완전 분리, 1=완전 겹침)
    """
    e_pos_all, e_neg_all = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            x_demo = batch["patch"].to(DEVICE)
            x0     = torch.randn_like(x_demo)
            x_neg  = rollout(vf, x0, n_steps=N_ROLLOUT_STEPS)

            e_pos_all.append(ebm(x_demo).cpu().numpy())
            e_neg_all.append(ebm(x_neg).cpu().numpy())

    e_pos = np.concatenate(e_pos_all)
    e_neg = np.concatenate(e_neg_all)

    # overlap coefficient (histogram 기반)
    lo = min(e_pos.min(), e_neg.min())
    hi = max(e_pos.max(), e_neg.max())
    bins = np.linspace(lo, hi, 50)
    h_pos, _ = np.histogram(e_pos, bins=bins, density=True)
    h_neg, _ = np.histogram(e_neg, bins=bins, density=True)
    bin_w = bins[1] - bins[0]
    overlap = float(np.minimum(h_pos, h_neg).sum() * bin_w)

    sep = abs(e_pos.mean() - e_neg.mean())
    avg_std = (e_pos.std() + e_neg.std()) / 2 + 1e-8
    sep_ratio = sep / avg_std

    return {
        "e_pos_mean": float(e_pos.mean()),
        "e_pos_std":  float(e_pos.std()),
        "e_neg_mean": float(e_neg.mean()),
        "e_neg_std":  float(e_neg.std()),
        "separation": float(sep),
        "sep_ratio":  float(sep_ratio),   # > 3이면 trivial separation
        "overlap_coef": overlap,           # 0에 가까울수록 trivially 분리됨
        "e_pos": e_pos,
        "e_neg": e_neg,
    }


# ── 3. within-demo energy rank variance ───────────────────────────────────────

def measure_demo_rank_quality(ebm: EBM, val_loader) -> dict:
    """
    demo 내부에서 energy가 disagreement(mal_var)를 실제로 ranking하는지.
    ranking이 죽었으면 energy std가 작거나 Spearman이 0에 가까울 것.
    """
    from scipy.stats import spearmanr

    energies, mal_vars = [], []
    with torch.no_grad():
        for batch in val_loader:
            x    = batch["patch"].to(DEVICE)
            e    = ebm(x).cpu().numpy()
            mv   = batch["malignancy_var"].numpy()
            energies.append(e)
            mal_vars.append(mv)

    e_all  = np.concatenate(energies)
    mv_all = np.concatenate(mal_vars)
    rho, p = spearmanr(e_all, mv_all)

    return {
        "e_demo_mean": float(e_all.mean()),
        "e_demo_std":  float(e_all.std()),
        "e_demo_min":  float(e_all.min()),
        "e_demo_max":  float(e_all.max()),
        "spearman_rho": float(rho),
        "spearman_p":   float(p),
        "ranking_alive": abs(rho) > 0.05 and p < 0.2,
    }


# ── 종합 출력 ──────────────────────────────────────────────────────────────────

def diagnose():
    ebm, vf = load_c_models()
    val_ds  = LIDCDataset(VAL_CSV, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Val: {len(val_ds)} samples\n")

    # 1. clamp
    print("─" * 55)
    print("[1] rollout() clamp hit ratio (trained C VF)")
    cr = measure_clamp_ratio(vf, n_samples=32)
    print(f"  clamp rate (final step): {cr['clamp_rate_final']:.1%}")
    print(f"  clamp rate (mean over {N_ROLLOUT_STEPS} steps): {cr['clamp_rate_mean_over_steps']:.1%}")
    print(f"  x_final  min={cr['x_final_min']:.3f}  max={cr['x_final_max']:.3f}"
          f"  mean={cr['x_final_mean']:.3f}  std={cr['x_final_std']:.3f}")
    if cr['clamp_rate_final'] > 0.30:
        print("  [WARN] 경계 포화 심각 — reward_term_grad_norm이 구조적으로 낮아질 수 있음")
    else:
        print("  [OK] clamp 포화 낮음")

    # 2. energy overlap
    print()
    print("─" * 55)
    print("[2] pos(demo) vs neg(policy) energy 분포")
    ov = measure_energy_overlap(ebm, vf, val_loader, max_batches=N_ENERGY_BATCHES)
    print(f"  demo  energy: mean={ov['e_pos_mean']:>8.3f}  std={ov['e_pos_std']:.3f}")
    print(f"  policy energy: mean={ov['e_neg_mean']:>8.3f}  std={ov['e_neg_std']:.3f}")
    print(f"  separation: {ov['separation']:.3f}  sep/std ratio: {ov['sep_ratio']:.2f}")
    print(f"  overlap coefficient: {ov['overlap_coef']:.4f}")

    if ov['overlap_coef'] < 0.05:
        print("  [FAIL] overlap≈0 — 완전 분리. trivial separation → binary saturation 유력")
    elif ov['overlap_coef'] < 0.20:
        print("  [WARN] overlap 낮음 — separation이 지나치게 쉬울 가능성")
    else:
        print("  [OK] 분포 겹침 있음 — ranking 학습 가능한 상태")

    if ov['sep_ratio'] > 3.0:
        print(f"  [FAIL] sep/std={ov['sep_ratio']:.1f} > 3 — FM policy sample이 trivially easy negative")
    else:
        print(f"  [OK] sep/std={ov['sep_ratio']:.1f} ≤ 3")

    # 3. demo ranking quality
    print()
    print("─" * 55)
    print("[3] demo 내부 energy ranking quality")
    rq = measure_demo_rank_quality(ebm, val_loader)
    print(f"  energy: mean={rq['e_demo_mean']:.3f}  std={rq['e_demo_std']:.3f}"
          f"  range=[{rq['e_demo_min']:.3f}, {rq['e_demo_max']:.3f}]")
    print(f"  Spearman ρ(energy, mal_var) = {rq['spearman_rho']:.4f}  p={rq['spearman_p']:.4f}")
    if rq['ranking_alive']:
        print("  [OK] ranking 신호 살아있음 (약하더라도)")
    else:
        print("  [WARN] ranking 신호 거의 없음 — energy가 disagreement를 배우지 못함")

    # 종합 판정
    print()
    print("═" * 55)
    print("종합 진단")
    print("─" * 55)

    trivial_neg = ov['overlap_coef'] < 0.10 or ov['sep_ratio'] > 3.0
    clamp_bad   = cr['clamp_rate_final'] > 0.30
    ranking_dead = not rq['ranking_alive']

    if trivial_neg:
        print("  [핵심] FM policy sample이 trivially easy negative — ranking이 binary separation으로 붕괴")
        print("         → SGLD hybrid 또는 warm-up 기간 SGLD 복귀 검토 필요")
    if clamp_bad:
        print("  [경고] rollout clamp 포화 — policy gradient path 약화")
        print("         → rollout() 내 clamp 제거 / tanh 스케일 검토")
    if ranking_dead:
        print("  [경고] energy가 demo 내부 ranking을 배우지 못함")
        print("         → saturation 선행 해결 필요")
    if not trivial_neg and not clamp_bad and not ranking_dead:
        print("  [OK] 구조적 문제 없음 — hyperparameter 재탐색으로 넘어갈 수 있음")


if __name__ == "__main__":
    diagnose()
