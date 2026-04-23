"""
Post-hoc temperature scaling for ECE 개선.

val set에서 optimal T를 찾고 test set에 적용.

사용법:
  python scripts/temperature_scaling.py \
      --config configs/ebm_fm_gate_v1.yaml \
              configs/ebm_fm_gate_v2.yaml \
      --ckpt  outputs/ebm_fm_gate_v1_20260421/ckpt_best_val.pt \
              outputs/ebm_fm_gate_v2_20260422/ckpt_best_val.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import LIDCDataset
from src.models.ebm import EBM
from src.evaluation.metrics import evaluate, expected_calibration_error, EvalResult
from scripts.eval_test import load_ebm


def collect_energies(model, loader, device):
    model.eval()
    energies, mal_means, mal_vars = [], [], []
    with torch.no_grad():
        for batch in loader:
            e = model(batch["patch"].to(device)).cpu().numpy()
            energies.append(e)
            mal_means.append(batch["malignancy_mean"].numpy())
            mal_vars.append(batch["malignancy_var"].numpy())
    return (
        np.concatenate(energies),
        np.concatenate(mal_means),
        np.concatenate(mal_vars),
    )


def find_optimal_temperature(energies_val, mal_vars_val):
    def ece_at_T(log_T):
        T = np.exp(log_T)
        return expected_calibration_error(energies_val / T, mal_vars_val)

    result = minimize_scalar(ece_at_T, bounds=(-3, 6), method="bounded")
    return float(np.exp(result.x)), float(result.fun)


def run_one(cfg_path, ckpt_path, device):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", Path(cfg_path).stem)
    out_dir   = Path(cfg["logging"]["output_dir"])
    base_ch   = cfg["model"]["base_ch"]

    ckpt = Path(ckpt_path) if ckpt_path else out_dir / "ckpt_best_val.pt"
    if not ckpt.exists():
        print(f"[SKIP] {exp_name}: checkpoint not found ({ckpt})")
        return None

    splits_dir = Path(cfg["data"]["splits_dir"])
    val_csv  = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    model = load_ebm(ckpt, base_ch, device)

    val_loader = DataLoader(
        LIDCDataset(val_csv, augment=False),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        LIDCDataset(test_csv, augment=False),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    e_val,  _, mv_val  = collect_energies(model, val_loader,  device)
    e_test, mm_test, mv_test = collect_energies(model, test_loader, device)

    # baseline (T=1)
    ece_val_base  = expected_calibration_error(e_val,  mv_val)
    ece_test_base = expected_calibration_error(e_test, mv_test)

    # optimal T from val
    T_opt, ece_val_opt = find_optimal_temperature(e_val, mv_val)
    ece_test_opt = expected_calibration_error(e_test / T_opt, mv_test)

    print(f"\n[{exp_name}]  ckpt={ckpt.name}")
    print(f"  T=1.000  │  val ECE={ece_val_base:.4f}  test ECE={ece_test_base:.4f}")
    print(f"  T={T_opt:.3f}  │  val ECE={ece_val_opt:.4f}  test ECE={ece_test_opt:.4f}  ← temperature scaled")

    return dict(
        name=exp_name,
        T=T_opt,
        ece_val_base=ece_val_base,
        ece_test_base=ece_test_base,
        ece_val_opt=ece_val_opt,
        ece_test_opt=ece_test_opt,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--ckpt",   nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpts = args.ckpt or [None] * len(args.config)
    if len(ckpts) != len(args.config):
        parser.error("--ckpt 개수가 --config 개수와 다릅니다.")

    results = []
    for cfg, ckpt in zip(args.config, ckpts):
        r = run_one(cfg, ckpt, device)
        if r:
            results.append(r)

    if len(results) > 1:
        print(f"\n{'─'*70}")
        print(f"{'실험':<28} {'T':>6}  {'val ECE':>9} {'→':>3} {'val ECE(T)':>10}  "
              f"{'test ECE':>9} {'→':>3} {'test ECE(T)':>11}")
        for r in results:
            print(f"{r['name']:<28} {r['T']:>6.3f}  "
                  f"{r['ece_val_base']:>9.4f} {'→':>3} {r['ece_val_opt']:>10.4f}  "
                  f"{r['ece_test_base']:>9.4f} {'→':>3} {r['ece_test_opt']:>11.4f}")


if __name__ == "__main__":
    main()
