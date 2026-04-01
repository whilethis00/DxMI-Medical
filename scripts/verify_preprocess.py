"""
전처리 결과 검증 & 시각화
- 결절 수 확인
- reward 분포, malignancy 분포 저장 (outputs/eda/)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
OUT_DIR = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    all_csv = SPLITS_DIR / "all.csv"
    if not all_csv.exists():
        print("[ERROR] data/splits/all.csv 없음 — preprocess_lidc.py 먼저 실행하세요.")
        return

    df = pd.read_csv(all_csv)
    print(f"\n{'='*50}")
    print(f"총 결절 수: {len(df)}")
    print(f"총 환자 수: {df['patient_id'].nunique()}")
    print(f"\n[Split 분포]")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"  {split:5s}: {len(sub):4d}개 결절 | {sub['patient_id'].nunique():4d}명")

    print(f"\n[Malignancy 통계]")
    print(f"  mean: {df['malignancy_mean'].mean():.3f} ± {df['malignancy_mean'].std():.3f}")
    print(f"  var : {df['malignancy_var'].mean():.3f} ± {df['malignancy_var'].std():.3f}")
    print(f"\n[Reward 통계]")
    print(f"  min: {df['reward'].min():.4f}  max: {df['reward'].max():.4f}")
    print(f"  mean: {df['reward'].mean():.4f} ± {df['reward'].std():.4f}")
    print(f"{'='*50}\n")

    # ── 시각화 ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) malignancy mean 분포
    axes[0].hist(df["malignancy_mean"], bins=20, edgecolor="black", color="steelblue")
    axes[0].set_title("Malignancy Mean Distribution")
    axes[0].set_xlabel("Malignancy Mean")
    axes[0].set_ylabel("Count")

    # 2) malignancy var 분포 (annotator disagreement)
    axes[1].hist(df["malignancy_var"], bins=20, edgecolor="black", color="salmon")
    axes[1].set_title("Malignancy Variance Distribution\n(Annotator Disagreement)")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Count")

    # 3) reward 분포
    axes[2].hist(df["reward"], bins=20, edgecolor="black", color="mediumseagreen")
    axes[2].set_title("Reward Distribution\n(r = -Var(malignancy))")
    axes[2].set_xlabel("Reward")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    out_path = OUT_DIR / "data_distributions.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] 분포 시각화 저장: {out_path}")

    # ── annotator 수 분포 ────────────────────────────────────────────────────
    print(f"[Annotator 수 분포]")
    print(df["n_annotators"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
