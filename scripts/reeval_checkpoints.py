"""
기존 best checkpoint 재평가 — AUROC convention 확정용

W2 실험 로그는 구버전 metrics.py(score=-energy 단일)로 찍혔음.
현재 metrics.py는 auroc_neg_energy / auroc_energy 둘 다 계산.
이 스크립트로 best checkpoint를 새 코드로 재평가해 convention을 확정한다.

Usage:
    cd /home/introai26/.agile/user/hsjung/DxMI_Medical
    /home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/reeval_checkpoints.py
"""

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.ebm import EBM
from src.data.dataset import LIDCDataset
from src.evaluation.metrics import evaluate

DEVICE = torch.device("cpu")
VAL_CSV = ROOT / "data/splits/val.csv"
BATCH_SIZE = 32

CHECKPOINTS = {
    "A_best (epoch10)": ROOT / "outputs/ebm_baseline/ckpt_epoch0010.pt",
    "B_best (epoch80)": ROOT / "outputs/supervised_reward/ckpt_epoch0080.pt",
    "C_best (epoch180)": ROOT / "outputs/irl_maxent/ckpt_epoch0180.pt",
}


def load_ebm(ckpt_path: Path) -> EBM:
    ck = torch.load(ckpt_path, map_location="cpu")
    model = EBM(base_ch=32)
    # A/B: {'epoch', 'model', 'opt', 'scheduler'}
    # C:   {'epoch', 'ebm', 'vf'}
    state = ck.get("model") or ck.get("ebm")
    if state is None:
        raise KeyError(f"Unknown checkpoint format, keys: {list(ck.keys())}")
    # DDP로 저장된 경우 'module.' 접두어 제거
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


def main():
    val_ds = LIDCDataset(VAL_CSV, augment=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Val set: {len(val_ds)} samples")
    print("=" * 70)
    print(f"{'실험':<22} {'ρ':>7} {'p':>8} {'AUROC(-E)':>10} {'AUROC(E)':>10} {'ECE':>7}")
    print("-" * 70)

    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"{name:<22}  checkpoint not found: {ckpt_path}")
            continue

        model = load_ebm(ckpt_path).to(DEVICE)
        r = evaluate(model, val_loader, DEVICE)
        results[name] = r

        flag = "PASS" if r.passed_clinical() else "FAIL"
        print(
            f"{name:<22} {r.rho:>7.4f} {r.p_value:>8.4f} "
            f"{r.auroc_neg_energy:>10.4f} {r.auroc_energy:>10.4f} "
            f"{r.ece:>7.4f}  [{flag}]"
        )

    print("=" * 70)
    print()
    print("AUROC convention 판단:")
    print("  Spearman ρ > 0 이면 energy ↑ = disagreement ↑")
    print("  따라서 'high energy = positive' → score = energy 쪽이 맞는 방향")
    print()
    for name, r in results.items():
        if r.rho > 0:
            consistent = "auroc_energy" if r.auroc_energy > 0.5 else "auroc_neg_energy (주의: 0.5 미만)"
            print(f"  {name}: ρ={r.rho:.4f} (양수) → 일관된 AUROC: {consistent}")
            print(f"    auroc_energy={r.auroc_energy:.4f}  auroc_neg_energy={r.auroc_neg_energy:.4f}")
        else:
            print(f"  {name}: ρ={r.rho:.4f} (음수/0) → AUROC 해석 주의")


if __name__ == "__main__":
    main()
