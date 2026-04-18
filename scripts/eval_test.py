"""
Test set 1회 평가 스크립트.

용도:
  val에서 고른 best checkpoint(ckpt_best_val.pt 또는 지정 .pt)를
  test set에 딱 한 번 돌려 최종 수치를 확정한다.
  A/B/C 비교 시 세 실험을 같은 규칙(best-val 선택 → test 평가)으로 정렬.

사용법:
  # 단일 실험 (ckpt_best_val.pt 자동 탐색)
  python scripts/eval_test.py --config configs/ebm_weighted_cd_r3.yaml

  # checkpoint 직접 지정
  python scripts/eval_test.py --config configs/ebm_weighted_cd_r3.yaml \
      --ckpt outputs/ebm_weighted_cd_r3/ckpt_best_val.pt

  # A/B/C 한번에 비교
  python scripts/eval_test.py \
      --config configs/ebm_baseline.yaml \
              configs/supervised_reward.yaml \
              configs/ebm_weighted_cd_r1.yaml \
      --ckpt  outputs/ebm_baseline/ckpt_best_val.pt \
              outputs/supervised_reward/ckpt_best_val.pt \
              outputs/ebm_weighted_cd_r1/ckpt_best_val.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import LIDCDataset
from src.models.ebm import EBM
from src.evaluation.metrics import evaluate
from torch.utils.data import DataLoader


def load_ebm(ckpt_path: Path, base_ch: int, device: torch.device) -> EBM:
    model = EBM(base_ch=base_ch).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("ebm", ckpt)  # IRL ckpt는 "ebm" 키, EBM-only ckpt는 flat
    model.load_state_dict(state)
    model.eval()
    return model


def eval_one(cfg_path: str, ckpt_path: str | None, device: torch.device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    out_dir   = Path(cfg["logging"]["output_dir"])
    base_ch   = cfg["model"]["base_ch"]

    # checkpoint 탐색 순서: 명시 → ckpt_best_val.pt → 없으면 에러
    if ckpt_path:
        ckpt = Path(ckpt_path)
    else:
        ckpt = out_dir / "ckpt_best_val.pt"

    if not ckpt.exists():
        print(f"[SKIP] {exp_name}: checkpoint not found ({ckpt})")
        return None

    # test loader (non-DDP, rank0 only)
    splits_dir = Path(cfg["data"]["splits_dir"])
    test_csv   = splits_dir / "test.csv"
    if not test_csv.exists():
        print(f"[SKIP] {exp_name}: test.csv not found ({test_csv})")
        return None

    test_ds = LIDCDataset(test_csv, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model = load_ebm(ckpt, base_ch, device)
    result = evaluate(model, test_loader, device)

    print(f"[TEST] {exp_name} | ckpt={ckpt.name} | {result}")
    return exp_name, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--ckpt",   nargs="*", default=None,
                        help="checkpoint 경로. config 수와 같거나 생략 (ckpt_best_val.pt 자동 탐색)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpts = args.ckpt or [None] * len(args.config)
    if len(ckpts) != len(args.config):
        parser.error("--ckpt 개수가 --config 개수와 다릅니다.")

    results = []
    for cfg_path, ckpt_path in zip(args.config, ckpts):
        r = eval_one(cfg_path, ckpt_path, device)
        if r is not None:
            results.append(r)

    if len(results) > 1:
        print("\n── 비교 요약 ──────────────────────────────")
        print(f"{'실험':<30} {'Spearman ρ':>12} {'p-value':>10} {'AUROC(E)':>10} {'ECE':>8}")
        for name, res in results:
            clinical = "PASS" if res.passed_clinical() else "FAIL"
            print(
                f"{name:<30} {res.rho:>+12.4f} {res.p_value:>10.4f} "
                f"({clinical}) {res.auroc_energy:>10.4f} {res.ece:>8.4f}"
            )


if __name__ == "__main__":
    main()
