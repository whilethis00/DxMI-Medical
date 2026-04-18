# DxMI Medical

Maximum Entropy IRL + EBM 기반 임상 의료 불확실성 추정 모델.
타겟: ICLR 2027 (제출 마감: 2026년 9월 말)

## 환경

- conda: `dxmi_medical` (Python 3.10)
- 실행: `/home/introai26/miniconda3/envs/dxmi_medical/bin/python`

## 프로젝트 구조

```
data/
  raw/        # LIDC-IDRI 원본
  processed/  # 전처리된 패치/볼륨
  splits/     # train/val/test split 파일
src/
  data/       # 데이터 로딩, 전처리, reward 계산
  models/     # EBM, Flow Matching, IRL
  evaluation/ # 메트릭 (ECE, AUROC, Spearman ρ)
configs/      # yaml 실험 설정
scripts/      # 실행 스크립트
outputs/      # 실험별 폴더 — <exp_name>_<YYYYMMDD>/
```

---

## 실험 루틴 (MUST FOLLOW)

### 1. 실험 실행

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/<실험config>.yaml
```

실험 폴더는 `train.py`가 자동 생성: `outputs/<exp_name>_<YYYYMMDD>/`

### 2. 실험 폴더 구조

```
outputs/<exp_name>_<YYYYMMDD>/
  train.log       # step/val/epoch 로그 자동 저장
  RESULT.md       # 실험 완료 시 자동 생성 (git push 포함)
  ckpt_epoch*.pt  # 체크포인트 (git 제외 — .gitignore)
```

### 3. 실험 완료 시 자동 처리 (train.py)

- `RESULT.md` 자동 생성: 개요, 실험 세팅, val 지표 파싱
- `train.log` + `RESULT.md` → git add/commit/push 자동 실행
- 체크포인트(.pt)는 git에 올리지 않음

### 4. RESULT.md 양식

실험 완료 시 아래 양식으로 작성. 섹션 순서 고정.

```markdown
# <실험명> — Result

## 1. 실험 메타
| 항목 | 내용 |
|------|------|
| **날짜** | YYYY-MM-DD |
| **베이스** | 이전 실험명 + 핵심 상태 |
| **목적** | 한 줄 |
| **설정** | 핵심 하이퍼파라미터 |
| **현황** | 학습 완료 / val 완료 등 |

## 2. 무엇을 검증하나
이전 실험의 문제 + 이번 변경 + 합격 기준 수치

## 3. 학습 손실 곡선
주요 epoch/step의 loss 테이블 + 관찰

## 4. 검증 지표
epoch별 val 지표 테이블 + 이전 실험 대비 비교 테이블 (목표 달성? ✅/❌)

## 5. 결과 해석 및 인사이트
### 긍정적
### 주의
### 근본 원인 가설 (있으면)

## 6. 다음 스텝
- [ ] 체크리스트

## 7. 저장 파일 목록
outputs/<실험명>/ 트리
```

### 5. .gitignore 확인

`outputs/**/*.pt`는 반드시 .gitignore에 포함되어야 함.

---

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
