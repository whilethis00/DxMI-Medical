# DxMI Medical

Maximum Entropy IRL + EBM 기반 임상 의료 불확실성 추정 모델.
타겟: ICLR 2027.

## 환경

- conda: `dxmi_medical` (Python 3.10)
- 실행: `/home/introai26/miniconda3/envs/dxmi_medical/bin/python`

## 프로젝트 구조

```text
configs/       yaml 실험 설정
data/          LIDC-IDRI raw/processed/splits
docs/          진행 기록, 계획, 운영 문서, 결과 메모
notebooks/     EDA
outputs/       실험별 산출물 — <exp_name>_<YYYYMMDD>/
papers/        참고 논문 PDF
scripts/       학습/평가/전처리/진단 실행 스크립트
src/           데이터, 모델, 평가 라이브러리 코드
```

문서 위치:

- `docs/progress/`: 주차별 진행 기록
- `docs/planning/`: 진단, 코드 수정 계획, 연구 방향
- `docs/operations/`: GPU/PBS 등 실행 운영 메모
- `docs/results/`: 단발 결과 요약
- `docs/README.md`: 문서 인덱스

## 실험 루틴

### 1. 실험 실행

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/<실험config>.yaml
```

실험 폴더는 `train.py`가 자동 생성: `outputs/<exp_name>_<YYYYMMDD>/`

### 2. 실험 폴더 구조

```text
outputs/<exp_name>_<YYYYMMDD>/
  train.log
  RESULT.md
  training_curves.png
  ckpt_epoch*.pt
  ckpt_best_val.pt
```

체크포인트 `.pt`는 git에 올리지 않는다.

### 3. Test 평가

```bash
python scripts/eval_test.py \
    --config configs/<실험config>.yaml \
    --ckpt outputs/<실험명>_<YYYYMMDD>/ckpt_best_val.pt
```

현재 v3 3-seed 평가는:

```bash
bash scripts/eval_v3_3seed.sh
```

## RESULT.md 양식

실험 완료 시 아래 섹션 순서를 유지한다.

```markdown
# <실험명> — Result

## 1. 실험 메타
## 2. 무엇을 검증하나
## 3. 학습 손실 곡선
## 4. 검증 지표
## 5. 결과 해석 및 인사이트
## 6. 다음 스텝
## 7. 저장 파일 목록
```

## 핵심 아이디어

- Reward: `r(x) = -Var(malignancy_scores)`
- EBM energy: energy 높음 = 의사 간 disagreement 큼 = 임상적으로 불확실
- Backbone: OT-CFM
- 현재 연구 프레이밍: model uncertainty가 아니라 clinical ambiguity estimation

## 합격 조건

1. Ablation: C(IRL) > B(supervised) > A(no IRL)
2. Clinical correlation: Spearman rho(energy, annotator disagreement), p < 0.05
3. Limited/noisy/preference expert signal에서 IRL의 필요성 입증

## 주의

- `outputs/**/*.pt`는 `.gitignore`에 유지한다.
- 기존 사용자 변경을 되돌리지 않는다.
- 실험 재현 명령은 가능한 한 기존 `scripts/*.py` 진입점을 유지한다.
