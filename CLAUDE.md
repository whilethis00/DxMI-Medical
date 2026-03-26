# DxMI Medical

Maximum Entropy IRL + EBM 기반 임상 의료 불확실성 추정 모델.
타겟: ICLR 2027 (제출 마감: 2026년 9월 말)

## 환경

- conda: `dxmi_medical` (Python 3.10)
- 실행: `/home/introai26/miniconda3/envs/dxmi_medical/bin/python`

## 프로젝트 구조

```
data/
  raw/        # LIDC-IDRI 원본 (다운로드 중 — screen: lidc_download)
  processed/  # 전처리된 패치/볼륨
  splits/     # train/val/test split 파일
src/
  data/       # 데이터 로딩, 전처리, reward 계산
  models/     # EBM, Flow Matching, IRL
  evaluation/ # 메트릭 (ECE, AUROC, Spearman ρ)
experiments/  # 실험별 폴더 (실험 시작 전에 구성)
configs/      # yaml 실험 설정
scripts/      # 실행 스크립트
outputs/      # 체크포인트, 로그, 결과
```

## 핵심 아이디어

- **Reward**: `r(x) = -Var(malignancy_scores)` — LIDC-IDRI 4명 판독 분산
- **EBM energy**: energy 높음 = 의사 간 disagreement 큼 = 임상적으로 불확실
- **Backbone**: OT-CFM (원본 DxMI의 DDPM 교체)

## 합격 조건 두 가지

1. Ablation: C(IRL) > B(supervised) > A(no IRL)
2. Clinical correlation: Spearman ρ(energy, annotator disagreement), p < 0.05

## 데이터

- LIDC-IDRI: CT, 1,018명, 2,635개 폐결절, 영상의학과 의사 4명 독립 판독 (악성도 1~5)
- 다운로드 스크립트: `scripts/download_lidc.py`

## 베이스 논문

- DxMI (NeurIPS 2024): https://github.com/swyoon/Diffusion-by-MaxEntIRL
