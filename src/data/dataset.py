"""
LIDC-IDRI PyTorch Dataset
- data/processed/{nodule_id}.npz 로드
- 학습용: augmentation (flip, rotate)
- reward, malignancy_scores, patch 반환
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class LIDCDataset(Dataset):
    """
    Args:
        split_csv: data/splits/{train,val,test}.csv 경로
        augment: True면 random flip + 90° rotation (학습 시 사용)
        patch_size: 패치 크기 (기본 48)
    """

    def __init__(self, split_csv: str | Path, augment: bool = False, patch_size: int = 48):
        self.df = pd.read_csv(split_csv)
        self.augment = augment
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        data = np.load(row["path"])

        patch = data["patch"].astype(np.float32)            # (48, 48, 48)
        reward = float(data["reward"])
        mal_scores = data["malignancy_scores"].astype(np.float32)

        if self.augment:
            patch = self._augment(patch)

        # (1, D, H, W) — channel-first
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)

        return {
            "patch":           patch_tensor,
            "reward":          torch.tensor(reward, dtype=torch.float32),
            "malignancy_mean": torch.tensor(float(row["malignancy_mean"]), dtype=torch.float32),
            "malignancy_var":  torch.tensor(float(row["malignancy_var"]),  dtype=torch.float32),
            "n_annotators":    torch.tensor(int(len(mal_scores)),          dtype=torch.long),
            "nodule_id":       row["nodule_id"],
            "patient_id":      row["patient_id"],
        }

    # ── augmentation ────────────────────────────────────────────────────────────

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Random flip along each axis + random 90° rotation in axial plane."""
        for axis in range(3):
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=axis)

        k = np.random.randint(0, 4)
        patch = np.rot90(patch, k=k, axes=(1, 2))  # axial plane (H, W)

        return np.ascontiguousarray(patch)


# ── 편의 함수 ────────────────────────────────────────────────────────────────────

def make_dataloaders(
    splits_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, torch.utils.data.DataLoader]:
    """
    train/val/test DataLoader를 한꺼번에 생성.

    Usage:
        loaders = make_dataloaders("data/splits", batch_size=32)
        for batch in loaders["train"]: ...
    """
    from torch.utils.data import DataLoader

    splits_dir = Path(splits_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            continue
        augment = split == "train"
        ds = LIDCDataset(csv, augment=augment)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
