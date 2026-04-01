# April Week 1 (2026-03-30 ~ 2026-04-05)

## 오늘 한 것 (2026-03-30)

- [x] LIDC-IDRI 데이터 다운로드 완료 확인
  - `data/raw/`: 128GB, 15,116 시리즈, DCM 258,335개
  - CT 시리즈(`1.3.6.*`) 1,308개, 최대 764 슬라이스/시리즈
- [x] `dxmi_medical` conda 환경 패키지 설치
  - `torch` 2.6.0, `pylidc` 0.2.3, `pydicom` 3.0.2, `SimpleITK` 2.5.3 등
- [x] `~/.pylidcrc` 설정 (pylidc → data/raw 연결)
- [x] 전처리 스크립트 작성: `scripts/preprocess_lidc.py`
  - 결절 중심 기준 48³ 패치 추출
  - reward = -Var(malignancy_scores) 계산
  - train/val/test split (patient 단위, 80/10/10)

---

## 이번 주 계획

### Day 1-2 (2026-03-30 ~ 03-31) — 데이터 전처리
- [x] `preprocess_lidc.py` 실행 및 결과 검증
  - 결절 수: 1,880개 (버그 수정 후), 패치 스킵 0개
  - reward/malignancy 분포 시각화 → `outputs/eda/data_distributions.png`
  - `data/splits/train.csv`(1525), `val.csv`(177), `test.csv`(178) 생성
  - **버그 수정**: `ann.centroid` 반환 형식 (y,x,z) → (z,y,x) 재배열
- [x] `src/data/dataset.py` 작성 (PyTorch Dataset + make_dataloaders)

### Day 3 (2026-04-01) — EBM 모델
- [x] `src/models/ebm.py` 작성
  - 3D ResNet backbone (48→24→12→6), 7.47M params
  - scalar energy output
  - SGLD negative sampling + contrastive divergence loss
- [x] `configs/ebm_baseline.yaml` 작성

### Day 4 (2026-04-01) — Flow Matching (OT-CFM)
- [x] `src/models/flow_matching.py` 작성
  - torchdyn 설치 + 3D U-Net velocity field, 11.15M params
  - OT-CFM loss (Gaussian path: x_t = (1-t)x_0 + t·x_1)
  - Euler sampling + EBM-guided sampling
- [x] EBM + Flow Matching 결합 인터페이스 (`EBMGuidedFlowMatching`)

### Day 5 (2026-04-01) — IRL 루프
- [x] `src/models/irl.py` 작성
  - MaxEnt IRL reward 업데이트 루프 (CD loss)
  - ReplayBuffer (SGLD 안정화)
  - FM policy reward shaping
- [x] `scripts/train.py` 스켈레톤 작성
  - `train_ebm_only()` (Ablation A), `train_irl()` (Ablation C)
- [x] `configs/irl_maxent.yaml` 작성

### Day 6-7 (2026-04-01) — 초기 실험 & 검증
- [x] `src/evaluation/metrics.py`: ECE, AUROC, Spearman ρ 구현
- [x] `notebooks/eda.ipynb`: 데이터 분포 EDA 실행 → `outputs/eda/{eda_full,patch_samples,null_distribution}.png`
- [x] 파이프라인 end-to-end 검증 (DataLoader→EBM→SGLD→loss→evaluate, CPU 2-step)
  - 수정: `malignancy_scores` → collate 불가 (가변 길이) → `n_annotators` 스칼라로 교체
- [ ] GPU 잡 확보 후 Ablation A 본격 훈련 시작

---

## 목표 지표 (참고)

| 실험 | 설명 | 합격 조건 |
|------|------|-----------|
| A | No IRL | baseline |
| B | Supervised reward | A < B |
| C | MaxEnt IRL (제안) | B < C |
| Clinical | Spearman ρ(energy, disagreement) | p < 0.05 |

## 참고

- 베이스 논문: DxMI (NeurIPS 2024) — https://github.com/swyoon/Diffusion-by-MaxEntIRL
- 타겟: ICLR 2027 (제출 마감 2026년 9월 말)
- 데이터: LIDC-IDRI (1,018명, 2,635 결절, 4명 판독)
