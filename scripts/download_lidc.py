"""
LIDC-IDRI Download Script
TCIA Public REST API 사용 — 계정 불필요
저장 위치: data/raw/
"""

import os
import time
import zipfile
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# TCIA Public REST API (계정 불필요)
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LIDC-IDRI"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def get_series_list():
    """LIDC-IDRI 전체 시리즈 목록 조회"""
    url = f"{BASE_URL}/getSeries"
    params = {"Collection": COLLECTION}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    series_list = resp.json()
    print(f"[INFO] 총 시리즈 수: {len(series_list)}")
    return series_list


def download_series(series_uid: str, out_dir: Path, retries: int = 3):
    """단일 시리즈 다운로드 (zip → 압축 해제)"""
    series_dir = out_dir / series_uid
    if series_dir.exists() and any(series_dir.iterdir()):
        return series_uid, "skipped"

    url = f"{BASE_URL}/getImage"
    params = {"SeriesInstanceUID": series_uid}

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=300, stream=True)
            resp.raise_for_status()

            zip_path = out_dir / f"{series_uid}.zip"
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            series_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(series_dir)
            zip_path.unlink()

            return series_uid, "ok"

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                return series_uid, f"failed: {e}"


def main(args):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[STEP 1] 시리즈 목록 조회 중...")
    series_list = get_series_list()

    # 슬라이스 범위 지정 (기본: 전체)
    series_list = series_list[args.start:args.end]
    print(f"[STEP 2] 다운로드 대상: {len(series_list)}개 시리즈")

    uids = [s["SeriesInstanceUID"] for s in series_list]

    ok, skipped, failed = 0, 0, []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_series, uid, RAW_DIR): uid for uid in uids}
        for i, future in enumerate(as_completed(futures), 1):
            uid, status = future.result()
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed.append((uid, status))
            if i % 10 == 0 or i == len(uids):
                print(f"  [{i}/{len(uids)}] ok={ok} skipped={skipped} failed={len(failed)}")

    print("\n[완료]")
    print(f"  성공: {ok} | 스킵(이미 존재): {skipped} | 실패: {len(failed)}")
    if failed:
        print("[실패 목록]")
        for uid, reason in failed:
            print(f"  {uid}: {reason}")
        # 실패 목록 저장
        fail_log = RAW_DIR / "failed_series.txt"
        with open(fail_log, "w") as f:
            for uid, reason in failed:
                f.write(f"{uid}\t{reason}\n")
        print(f"  → {fail_log} 에 저장됨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIDC-IDRI 다운로드 (TCIA Public API)")
    parser.add_argument("--start", type=int, default=0, help="시리즈 목록 시작 인덱스")
    parser.add_argument("--end", type=int, default=None, help="시리즈 목록 끝 인덱스 (기본: 전체)")
    parser.add_argument("--workers", type=int, default=4, help="동시 다운로드 스레드 수 (기본: 4)")
    args = parser.parse_args()
    main(args)
