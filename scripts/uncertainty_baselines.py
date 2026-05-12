"""
Standard model uncertainty baselines for experiment block 1.

This script trains a malignancy classifier, then evaluates whether common
classification uncertainty scores recover expert disagreement
(`malignancy_var`) on the same split used by the EBM evaluations.

Scores reported:
  - predictive_entropy: entropy of the single-model predictive probability
  - margin_uncertainty: 1 - |2p - 1|
  - mc_entropy: entropy of MC-dropout mean probability
  - mc_variance: variance of MC-dropout probabilities
  - ensemble_entropy / ensemble_variance when multiple seeds are available

The key clinical metrics are Spearman rho(score, malignancy_var), AUROC for
high-disagreement cases, and top-k enrichment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import LIDCDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        groups = min(8, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MalignancyClassifier3D(nn.Module):
    """Small 3D CNN for binary malignancy classification."""

    def __init__(self, base_ch: int = 16, dropout: float = 0.2):
        super().__init__()
        ch = base_ch
        self.encoder = nn.Sequential(
            ConvBlock3D(1, ch, dropout),
            ConvBlock3D(ch, ch * 2, dropout),
            ConvBlock3D(ch * 2, ch * 4, dropout),
            ConvBlock3D(ch * 4, ch * 8, dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch * 8, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)


@dataclass
class TrainResult:
    seed: int
    best_epoch: int
    best_val_auc: float
    ckpt_path: Path


def make_loader(
    splits_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = LIDCDataset(splits_dir / f"{split}.csv", augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def batch_labels(batch: dict, threshold: float, device: torch.device) -> torch.Tensor:
    return (batch["malignancy_mean"].to(device) >= threshold).float()


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


@torch.no_grad()
def evaluate_classifier_auc(
    model: nn.Module,
    loader: DataLoader,
    threshold: float,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    labels, probs = [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["patch"].to(device)
        y = batch_labels(batch, threshold, device)
        p = torch.sigmoid(model(x))
        labels.append(y.cpu().numpy())
        probs.append(p.cpu().numpy())
    return binary_auc(np.concatenate(labels), np.concatenate(probs))


def train_one_seed(
    seed: int,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
) -> TrainResult:
    set_seed(seed)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    splits_dir = Path(data_cfg["splits_dir"])
    batch_size = int(args.batch_size or data_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg["num_workers"])
    threshold = float(train_cfg.get("malignancy_threshold", 3.0))

    train_loader = make_loader(splits_dir, "train", batch_size, num_workers, augment=True, shuffle=True)
    val_loader = make_loader(splits_dir, "val", batch_size, num_workers, augment=False, shuffle=False)

    model = MalignancyClassifier3D(
        base_ch=int(args.base_ch or model_cfg.get("base_ch", 16)),
        dropout=float(args.dropout if args.dropout is not None else model_cfg.get("dropout", 0.2)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr or train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(args.epochs or train_cfg["epochs"])
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    patience = int(train_cfg.get("early_stop_patience", epochs))

    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break
            x = batch["patch"].to(device)
            y = batch_labels(batch, threshold, device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val_auc = evaluate_classifier_auc(
            model,
            val_loader,
            threshold,
            device,
            max_batches=args.max_eval_batches,
        )
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[seed {seed}] epoch {epoch:03d} loss={mean_loss:.4f} val_malignancy_auc={val_auc:.4f}")

        ckpt_path = seed_dir / "ckpt_best.pt"
        is_better = (
            not ckpt_path.exists()
            or (not math.isnan(val_auc) and (math.isnan(best_auc) or val_auc > best_auc))
        )
        if is_better:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            ckpt = {
                "model": model.state_dict(),
                "seed": seed,
                "epoch": epoch,
                "val_malignancy_auc": val_auc,
                "config": cfg,
            }
            torch.save(ckpt, ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[seed {seed}] early stop at epoch {epoch}")
                break

    return TrainResult(seed=seed, best_epoch=best_epoch, best_val_auc=best_auc, ckpt_path=seed_dir / "ckpt_best.pt")


def load_model(ckpt_path: Path, cfg: dict, device: torch.device, args: argparse.Namespace) -> nn.Module:
    model_cfg = cfg["model"]
    model = MalignancyClassifier3D(
        base_ch=int(args.base_ch or model_cfg.get("base_ch", 16)),
        dropout=float(args.dropout if args.dropout is not None else model_cfg.get("dropout", 0.2)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def collect_predictions(
    models: list[nn.Module],
    loader: DataLoader,
    threshold: float,
    device: torch.device,
    mc_samples: int,
    max_batches: int | None = None,
) -> pd.DataFrame:
    if not models:
        raise ValueError("At least one model is required for prediction collection.")

    rows = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = batch["patch"].to(device)
        n = x.shape[0]

        single_logits = models[0](x)
        single_probs = torch.sigmoid(single_logits).cpu().numpy()

        mc_probs = []
        for _ in range(mc_samples):
            models[0].eval()
            enable_dropout(models[0])
            mc_probs.append(torch.sigmoid(models[0](x)).cpu().numpy())
        models[0].eval()
        mc_probs_np = np.stack(mc_probs, axis=0)

        ensemble_probs = []
        for model in models:
            model.eval()
            ensemble_probs.append(torch.sigmoid(model(x)).cpu().numpy())
        ensemble_probs_np = np.stack(ensemble_probs, axis=0)

        for i in range(n):
            rows.append(
                {
                    "nodule_id": batch["nodule_id"][i],
                    "patient_id": batch["patient_id"][i],
                    "malignancy_mean": float(batch["malignancy_mean"][i]),
                    "malignancy_var": float(batch["malignancy_var"][i]),
                    "malignancy_label": int(float(batch["malignancy_mean"][i]) >= threshold),
                    "prob_single": float(single_probs[i]),
                    "prob_mc_mean": float(mc_probs_np[:, i].mean()),
                    "prob_mc_var": float(mc_probs_np[:, i].var()),
                    "prob_ensemble_mean": float(ensemble_probs_np[:, i].mean()),
                    "prob_ensemble_var": float(ensemble_probs_np[:, i].var()),
                }
            )
    return pd.DataFrame(rows)


def entropy_binary(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-7, 1.0 - 1e-7)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def compute_score_table(
    pred_df: pd.DataFrame,
    high_disagreement_quantile: float,
    topk_fractions: Iterable[float],
) -> pd.DataFrame:
    df = pred_df.copy()
    df["predictive_entropy"] = entropy_binary(df["prob_single"].to_numpy())
    df["margin_uncertainty"] = 1.0 - np.abs(2.0 * df["prob_single"].to_numpy() - 1.0)
    df["mc_entropy"] = entropy_binary(df["prob_mc_mean"].to_numpy())
    df["mc_variance"] = df["prob_mc_var"].to_numpy()
    df["ensemble_entropy"] = entropy_binary(df["prob_ensemble_mean"].to_numpy())
    df["ensemble_variance"] = df["prob_ensemble_var"].to_numpy()

    var = df["malignancy_var"].to_numpy()
    high_cutoff = float(np.quantile(var, high_disagreement_quantile))
    high = (var >= high_cutoff).astype(int)
    high_rate = float(high.mean())
    labels = df["malignancy_label"].to_numpy()

    score_names = [
        "predictive_entropy",
        "margin_uncertainty",
        "mc_entropy",
        "mc_variance",
        "ensemble_entropy",
        "ensemble_variance",
    ]

    rows = []
    for name in score_names:
        score = df[name].to_numpy()
        rho, p_value = spearmanr(score, var)
        row = {
            "score": name,
            "n": len(df),
            "rho": float(rho),
            "p_value": float(p_value),
            "auroc_high_disagreement": binary_auc(high, score),
            "malignancy_auroc": binary_auc(labels, df["prob_single"].to_numpy()),
            "high_disagreement_cutoff": high_cutoff,
            "high_disagreement_rate": high_rate,
        }

        order = np.argsort(-score)
        for frac in topk_fractions:
            k = max(1, int(math.ceil(len(df) * float(frac))))
            top_rate = float(high[order[:k]].mean())
            row[f"top{int(float(frac) * 100)}_high_rate"] = top_rate
            row[f"top{int(float(frac) * 100)}_enrichment"] = top_rate / high_rate if high_rate > 0 else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def format_markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df[cols].iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("nan" if math.isnan(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def write_markdown_results(result_df: pd.DataFrame, path: Path, split: str) -> None:
    cols = [
        "score",
        "rho",
        "p_value",
        "auroc_high_disagreement",
        "malignancy_auroc",
        "top5_enrichment",
        "top10_enrichment",
        "top20_enrichment",
    ]
    present_cols = [c for c in cols if c in result_df.columns]
    lines = [
        f"# Uncertainty Baseline Results ({split})",
        "",
        format_markdown_table(result_df, present_cols),
        "",
        "Interpretation guide:",
        "",
        "- Low rho / AUROC near 0.5 means standard model uncertainty does not recover expert disagreement.",
        "- `malignancy_auroc` is a sanity check for classification learning, not the clinical ambiguity result.",
    ]
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_uncertainty_baselines.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-split", default="test", choices=["val", "test"])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--base-ch", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def check_environment() -> None:
    checks = {
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "yaml": yaml.__version__,
    }
    print(json.dumps(checks, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.check_env:
        check_environment()
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    seeds = args.seeds if args.seeds else list(train_cfg.get("seeds", [1]))
    threshold = float(train_cfg.get("malignancy_threshold", 3.0))
    mc_samples = int(args.mc_samples or eval_cfg.get("mc_samples", 20))

    train_results = []
    if not args.eval_only:
        for seed in seeds:
            train_results.append(train_one_seed(seed, cfg, device, out_dir, args))
        pd.DataFrame([r.__dict__ for r in train_results]).to_csv(out_dir / "train_summary.csv", index=False)

    ckpt_paths = [out_dir / f"seed{seed}" / "ckpt_best.pt" for seed in seeds]
    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    models = [load_model(path, cfg, device, args) for path in ckpt_paths]

    data_cfg = cfg["data"]
    splits_dir = Path(data_cfg["splits_dir"])
    batch_size = int(args.batch_size or data_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else data_cfg["num_workers"])
    loader = make_loader(splits_dir, args.eval_split, batch_size, num_workers, augment=False, shuffle=False)

    pred_df = collect_predictions(
        models=models,
        loader=loader,
        threshold=threshold,
        device=device,
        mc_samples=mc_samples,
        max_batches=args.max_eval_batches,
    )
    result_df = compute_score_table(
        pred_df,
        high_disagreement_quantile=float(eval_cfg.get("high_disagreement_quantile", 0.75)),
        topk_fractions=eval_cfg.get("topk_fractions", [0.05, 0.10, 0.20]),
    )

    pred_path = out_dir / f"predictions_{args.eval_split}.csv"
    result_path = out_dir / f"results_{args.eval_split}.csv"
    md_path = out_dir / f"results_{args.eval_split}.md"
    pred_df.to_csv(pred_path, index=False)
    result_df.to_csv(result_path, index=False)
    write_markdown_results(result_df, md_path, args.eval_split)

    print(f"[DONE] predictions: {pred_path}")
    print(f"[DONE] results: {result_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
