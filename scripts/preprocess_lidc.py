"""
LIDC-IDRI 전처리 스크립트
- pylidc로 결절 annotation 파싱 (malignancy score, centroid)
- pydicom으로 CT 볼륨 직접 로드 (data/raw/<SeriesUID>/ flat 구조)
- 각 결절마다 reward = -Var(malignancy_scores) 계산
- 결절 중심 기준 48³ 패치 추출
- 결과: data/processed/{nodule_id}.npz
- split: data/splits/{train,val,test}.csv (patient 단위 80/10/10)
"""

import argparse
import traceback
import numpy as np
import pandas as pd
import pylidc as pl
import pydicom
from pathlib import Path
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
RAW_DIR     = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR  = ROOT / "data" / "splits"

PATCH_SIZE = 48
HU_MIN, HU_MAX = -1000, 400
VAL_RATIO  = 0.1
TEST_RATIO = 0.1
SEED       = 42


# ── DICOM 로딩 ────────────────────────────────────────────────────────────────

def load_volume(series_uid: str) -> tuple[np.ndarray, float, float] | None:
    """
    data/raw/<series_uid>/ 의 DCM 파일을 로드해 (Z,Y,X) HU 볼륨으로 반환.
    Returns: (volume, slice_thickness, pixel_spacing) or None on failure.
    """
    series_dir = RAW_DIR / series_uid
    if not series_dir.exists():
        return None

    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    slices = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False)
            slices.append(ds)
        except Exception:
            continue

    if not slices:
        return None

    # InstanceNumber 또는 ImagePositionPatient z 기준 정렬
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except Exception:
        try:
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except Exception:
            pass

    try:
        pixel_spacing = float(slices[0].PixelSpacing[0])
        slice_thickness = float(slices[0].SliceThickness)
    except Exception:
        pixel_spacing = 1.0
        slice_thickness = 1.0

    # RescaleSlope / Intercept → HU
    imgs = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        slope     = float(getattr(s, "RescaleSlope",     1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        imgs.append(arr * slope + intercept)

    volume = np.stack(imgs, axis=0)  # (Z, Y, X)
    return volume, slice_thickness, pixel_spacing


def hu_normalize(volume: np.ndarray) -> np.ndarray:
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume.astype(np.float32)


# ── 패치 추출 ─────────────────────────────────────────────────────────────────

def extract_patch(volume: np.ndarray, centroid: np.ndarray, size: int) -> np.ndarray | None:
    """centroid (z, y, x) 기준 size³ 패치. 경계는 0 패딩."""
    half = size // 2
    z, y, x = [int(round(float(c))) for c in centroid]
    D, H, W = volume.shape

    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    pad = [(max(0, -z0), max(0, z1 - D)),
           (max(0, -y0), max(0, y1 - H)),
           (max(0, -x0), max(0, x1 - W))]

    z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
    z1, y1, x1 = min(z1, D), min(y1, H), min(x1, W)

    patch = volume[z0:z1, y0:y1, x0:x1]
    if any(p[0] + p[1] > 0 for p in pad):
        patch = np.pad(patch, pad, mode="constant", constant_values=0.0)

    if patch.shape != (size, size, size):
        return None
    return patch


# ── 메인 ──────────────────────────────────────────────────────────────────────

def process(patch_size: int = PATCH_SIZE):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    print(f"[INFO] 총 스캔 수: {len(scans)}")

    records   = []
    n_skip_vol = 0
    n_skip_patch = 0

    for scan in tqdm(scans, desc="Processing"):
        # ── 볼륨 로드 ──────────────────────────────────────────────────────
        result = load_volume(scan.series_instance_uid)
        if result is None:
            n_skip_vol += 1
            continue
        vol, slice_thickness, pixel_spacing = result
        vol = hu_normalize(vol)
        spacing = np.array([slice_thickness, pixel_spacing, pixel_spacing], dtype=np.float32)

        # ── 결절 annotation 파싱 ──────────────────────────────────────────
        try:
            nod_clusters = scan.cluster_annotations()
        except Exception:
            n_skip_vol += 1
            continue

        for nod_idx, annotations in enumerate(nod_clusters):
            scores = np.array([ann.malignancy for ann in annotations], dtype=np.float32)
            if len(scores) < 2:
                continue  # 단독 판독 결절 제외

            reward = -float(np.var(scores))

            # pylidc centroid 반환 형식: (row/y, col/x, slice/z)
            # extract_patch 는 (z, y, x) 순서를 기대하므로 축 재배열
            centroids_yxz = np.array([ann.centroid for ann in annotations])  # (N, 3) y,x,z
            c_mean = centroids_yxz.mean(axis=0)
            centroid = np.array([c_mean[2], c_mean[0], c_mean[1]], dtype=np.float32)  # → (z, y, x)

            patch = extract_patch(vol, centroid, patch_size)
            if patch is None:
                n_skip_patch += 1
                continue

            nodule_id = (f"{scan.patient_id}"
                         f"_s{scan.series_instance_uid[-8:]}"
                         f"_n{nod_idx:03d}")
            out_path = PROCESSED_DIR / f"{nodule_id}.npz"

            np.savez_compressed(
                out_path,
                patch=patch,
                reward=np.float32(reward),
                malignancy_scores=scores,
                spacing=spacing,
                centroid=centroid.astype(np.float32),
            )

            records.append({
                "nodule_id":        nodule_id,
                "patient_id":       scan.patient_id,
                "series_uid":       scan.series_instance_uid,
                "n_annotators":     int(len(scores)),
                "malignancy_mean":  float(scores.mean()),
                "malignancy_var":   float(np.var(scores)),
                "reward":           reward,
                "path":             str(out_path),
            })

    print(f"\n[INFO] 완료: {len(records)}개 결절 저장")
    print(f"[INFO] 볼륨 로드 실패: {n_skip_vol}  |  패치 크기 부족 스킵: {n_skip_patch}")

    if not records:
        print("[ERROR] 처리된 결절이 없습니다. RAW_DIR 경로를 확인하세요.")
        return

    # ── Split (patient 단위) ───────────────────────────────────────────────
    df = pd.DataFrame(records)
    patients = df["patient_id"].unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)

    n       = len(patients)
    n_test  = max(1, int(n * TEST_RATIO))
    n_val   = max(1, int(n * VAL_RATIO))

    test_pats = set(patients[:n_test])
    val_pats  = set(patients[n_test:n_test + n_val])

    df["split"] = df["patient_id"].apply(
        lambda p: "test" if p in test_pats else ("val" if p in val_pats else "train")
    )

    df.to_csv(SPLITS_DIR / "all.csv", index=False)
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        sub.to_csv(SPLITS_DIR / f"{split}.csv", index=False)
        print(f"[INFO] {split:5s}: {len(sub):4d}개 결절 | {sub['patient_id'].nunique():4d}명")

    print(f"[INFO] splits 저장 완료 → {SPLITS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE,
                        help=f"패치 크기 (기본값: {PATCH_SIZE})")
    args = parser.parse_args()
    process(patch_size=args.patch_size)
