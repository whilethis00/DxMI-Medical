# April Week 2 (2026-04-06 ~ 2026-04-12)

## 이번 주 요약

- [x] Ablation A (`No IRL`) 200 epoch 학습 완료
- [x] Ablation B (`Supervised reward`) 200 epoch 학습 완료
- [x] Ablation C (`MaxEnt IRL`) 200 epoch 학습 완료
- [x] 각 실험의 validation 지표 확인
- [ ] test set 평가 결과 정리
- [ ] 통합 결과표(`csv/json/md`) 작성
- [ ] A/C 실패 원인 디버깅 및 재실험 설계

---

## 이번 주 진행 내용

### 1. Ablation 실험 실행

- `outputs/ebm_baseline/`: epoch 10 ~ 200 체크포인트 저장 완료
- `outputs/supervised_reward/`: epoch 10 ~ 200 체크포인트 저장 완료
- `outputs/irl_maxent/`: epoch 10 ~ 200 체크포인트 저장 완료
- 주요 로그:
  - `outputs/ablation_A.log`
  - `outputs/ablation_B.log`
  - `outputs/ablation_C.log`

### 2. 실험 결과 요약 (validation, N=177)

| 실험 | 설명 | 최종 Spearman ρ | p-value | AUROC | ECE | 판정 |
|------|------|------------------|---------|-------|-----|------|
| A | No IRL | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B | Supervised reward | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C | MaxEnt IRL | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

- Clinical 기준: `Spearman ρ(energy, disagreement)`에서 `p < 0.05`
- 현재 결과는 `B`만 clinical 기준 통과

### 3. Best checkpoint 관찰값

- Ablation A:
  - 초기 epoch 10이 가장 나음: `ρ=-0.0874`, `AUROC=0.6113`
  - 이후 전 구간에서 `loss=-5.0000` 근처로 포화
- Ablation B:
  - best observed at epoch 80: `ρ=0.3028 (p<0.001)`, `AUROC=0.3281`, `ECE=0.2095`
  - 전체적으로 안정적으로 positive correlation 유지
- Ablation C:
  - best observed at epoch 180: `ρ=0.1071`, `p=0.1560`
  - 전 구간 clinical 기준 미통과
  - `reward_loss=-5.0000` 포화 패턴 반복

---

## 해석

### 잘 된 점

- W1에서 미완료였던 GPU 기반 본 학습이 실제로 완료됨
- A/B/C 세 실험의 비교 가능한 로그와 체크포인트 확보
- supervised reward setting(B)에서는 disagreement 예측 신호가 유의하게 학습됨

### 문제점

- 목표였던 `A < B < C`를 달성하지 못함
- 제안 방법인 `C`가 `B`를 넘지 못했고, clinical 기준도 실패
- `A`와 `C` 모두 reward/energy가 매우 빠르게 동일 고정점으로 포화되는 양상
- 결과 요약 테이블, test 평가, 논문용 figure/table이 아직 없음

---

## A/C 실패 원인 분석

### Ablation A (`No IRL`) 실패 가설

#### 1. CD loss 자체가 상수 고정점으로 수렴할 가능성

- 현재 EBM 손실:
  - `loss = E(x_pos) - E(x_neg) + λ(E(x_pos)^2 + E(x_neg)^2)`
- 로그에서 매우 빠르게
  - `e_pos ≈ -5`
  - `e_neg ≈ 5`
  - `loss ≈ -5`
  로 포화됨
- 이는 단순한 학습 실패라기보다, 현재 손실식과 `l2_reg=0.1` 조합이 에너지 평균값을 특정 상수로 밀어붙이는 구조일 가능성이 큼
- 결과적으로 샘플 간 ranking 학습보다 `real=-5`, `neg=5` 고정점으로 가는 것이 더 쉬운 최적화 문제가 되었을 가능성

#### 2. reward ranking이 아니라 energy saturation이 먼저 발생

- validation에서 Spearman ρ가 전 구간 음수이며 clinical 기준 미통과
- ECE도 거의 `0.7956`으로 고정
- 즉, 샘플별 disagreement ordering을 배우지 못하고, 전체 energy scale만 포화된 것으로 해석 가능

#### 3. early stopping이 실질적으로 작동하지 않았을 가능성

- `early_stop_patience=20`이지만 200 epoch 전체가 모두 실행됨
- A의 val ρ는 초반부터 개선되지 않는데도 학습이 계속 진행됨
- `save_interval=10` 기준 평가만 하기 때문에 stopping 조건이 기대대로 작동하는지 점검 필요

### Ablation C (`MaxEnt IRL`) 실패 가설

#### 1. reward update가 실제로는 IRL보다 CD 재실행에 가까움

- `update_reward()`는 demonstration와 policy sample의 occupancy matching이 아니라, 다시 SGLD negative sample을 생성해 EBM contrastive divergence를 수행함
- 즉 C의 reward 학습은 본질적으로 A의 reward 학습과 매우 유사한 경로를 반복
- 로그에서도 `reward_loss=-5.0000`, `e_pos≈-5`, `e_neg≈5` 포화 패턴이 A와 거의 동일

#### 2. policy shaping gradient가 FM에 제대로 전달되지 않을 가능성

- 현재 policy update의 reward term은 `x_approx = 0.01*x0 + 0.99*x_demo`에서 계산됨
- 이 `x_approx`는 `VelocityField` 출력에 의존하지 않음
- 따라서 `reward_loss = EBM(x_approx)`가 FM 파라미터를 실제로 업데이트하지 못할 가능성이 높음
- 결과적으로 C는 “reward로 policy를 개선하는 IRL”이 아니라
  - reward model은 A처럼 따로 포화되고
  - FM은 OT-CFM loss만 별도로 학습하는 구조에 가까울 수 있음

#### 3. reward loss와 fm loss의 결합이 느슨함

- 로그상 `fm_loss`는 대체로 감소하지만 clinical metric은 개선되지 않음
- 이는 생성 품질과 uncertainty-aware reward 학습이 서로 연결되지 않았다는 신호
- 현재 구현에서는 reward가 policy trajectory를 충분히 제약하지 못할 가능성이 큼

### 공통 가설

#### 1. 평가 metric과 학습 목표의 sign 해석이 일부 어긋날 가능성

- 프로젝트 설명은 “energy가 높을수록 disagreement가 커야 함”
- 하지만 AUROC는 `score = -energy`로 계산됨
- B는 Spearman ρ는 유의미하게 양수인데 AUROC는 0.3대에 머묾
- 따라서 malignancy 분류 score sign 또는 threshold 정의가 현재 실험 목적과 완전히 맞는지 재검토 필요

#### 2. 현재 A/C는 uncertainty ranking 문제보다 density separation 문제를 더 강하게 풀고 있을 수 있음

- disagreement 예측이 핵심이면 샘플 간 상대적 순서가 중요함
- 그러나 현재 손실은 positive/negative 평균 energy 분리와 energy magnitude regularization의 영향이 매우 큼
- 이 경우 uncertainty surrogate로서의 energy 품질보다 CD 안정화가 우선되는 방향으로 학습될 수 있음

---

## 실패 원인 기반 액션 아이템

### A 수정 방향

- [ ] CD loss에서 energy magnitude 고정점 유도 성분 완화
- [ ] `l2_reg` 더 낮게 재탐색
- [ ] ranking 보존형 objective 또는 supervised auxiliary term 검토
- [ ] best epoch 기준 selection으로 최종 checkpoint 재정의

### C 수정 방향

- [ ] reward update를 실제 FM sample 기반 occupancy matching으로 변경
- [ ] reward shaping term이 `VelocityField` 출력에 직접 연결되도록 경로 수정
- [ ] IRL에서 policy sample과 SGLD negative sample의 역할 분리
- [ ] `reward_loss`, `fm_loss`, validation metric 간 상관을 함께 기록

### 평가 수정 방향

- [ ] AUROC score sign 재검토
- [ ] uncertainty metric과 malignancy metric을 분리 보고
- [ ] val 뿐 아니라 test 기준 테이블 작성

---

## 다음 액션 아이템

### 우선순위 1 — 실패 원인 분석

- [ ] Ablation A reward collapse 원인 분석
- [ ] Ablation C에서 IRL reward shaping이 실제로 동작하는지 코드 점검
- [ ] energy sign / AUROC 정의가 실험 의도와 맞는지 재검증

### 우선순위 2 — 재실험 준비

- [ ] `l2_reg`, `sgld_steps`, `sgld_step_size` 재탐색
- [ ] early stopping 기준을 clinical metric 중심으로 재정비
- [ ] best epoch 기준 checkpoint selection 규칙 명시

### 우선순위 3 — 결과 정리

- [ ] val/test 통합 결과표 작성
- [ ] 주간 보고 및 논문 초안용 figure/table 정리
- [ ] 실패 원인 및 수정 계획 문서화

---

## 참고 산출물

- EDA:
  - `outputs/eda/data_distributions.png`
  - `outputs/eda/eda_full.png`
  - `outputs/eda/patch_samples.png`
  - `outputs/eda/null_distribution.png`
- 로그:
  - `outputs/ablation_A.log`
  - `outputs/ablation_B.log`
  - `outputs/ablation_C.log`
- 체크포인트:
  - `outputs/ebm_baseline/`
  - `outputs/supervised_reward/`
  - `outputs/irl_maxent/`

## 한 줄 결론

- 실행 진행률은 높았지만, 연구 성과 기준으로는 아직 `B만 유의미`, `A/C는 재설계 필요` 상태다.
