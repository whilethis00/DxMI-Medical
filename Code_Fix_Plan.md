# Code Fix Plan

## 배경

- 현재 실험 결과는 `B만 유의미`, `A/C는 실패`로 해석된다.
- 재실험 전에 코드 레벨에서 먼저 정리해야 할 문제를 우선순위 기준으로 나눈다.

---

## P0. Ablation C가 실제 IRL이 되도록 수정

### 왜 먼저 해야 하나

- 현재 C는 이름은 `MaxEnt IRL`이지만, 실제 구현은 IRL 핵심 경로가 약하다.
- 이 상태에서 하이퍼파라미터를 먼저 튜닝해도 의미 있는 개선이 나올 가능성이 낮다.

### 수정 대상 1

- `src/models/irl.py`의 `update_policy()`

### 핵심 문제

- reward term이 FM이 만든 샘플이 아니라 `x_demo` 기반 근사값에 걸려 있음
- 따라서 reward shaping gradient가 `VelocityField`에 제대로 전달되지 않을 수 있음

### 해야 할 일

- FM 출력 샘플 또는 적어도 FM trajectory를 통해 만든 `x_t -> x_1`에 reward를 걸기
- reward term이 실제로 `vf.parameters()`에 gradient를 만드는지 확인하기

### 완료 기준

- reward term 제거 전후로 `vf` gradient norm 차이가 로그에서 확인될 것
- policy update가 reward에 반응하는 샘플 energy 변화를 보일 것

---

## P1. Ablation C의 reward update를 policy sample 중심으로 변경

### 왜 중요한가

- 지금 C의 reward update는 policy occupancy matching보다 SGLD negative 기반 CD에 가깝다.
- 그러면 C는 사실상 A를 다시 한 번 도는 구조가 된다.

### 수정 대상

- `src/models/irl.py`의 `update_reward()`

### 해야 할 일

- demo sample과 policy-generated sample을 직접 비교하는 reward update로 바꾸기
- SGLD sample은 보조 안정화 수단으로만 남기거나 제거 여부를 결정하기

### 완료 기준

- 로그에서 `reward_loss`가 A와 똑같이 `-5` 근처로 즉시 포화되지 않을 것
- C의 reward dynamics가 A와 다른 패턴을 보일 것

---

## P2. EBM loss가 상수 고정점으로 수렴하는 문제 완화

### 왜 중요한가

- A와 C 모두 `e_pos≈-5`, `e_neg≈5`로 빠르게 포화된다.
- 이 패턴은 uncertainty ranking 학습보다 energy magnitude saturation이 우세하다는 뜻이다.

### 수정 대상

- `src/models/ebm.py`
- `configs/ebm_baseline.yaml`
- `configs/irl_maxent.yaml`

### 해야 할 일

- 현재 CD + L2 구조를 재검토하기
- `l2_reg`를 낮춘 실험을 가장 먼저 수행하기
- 필요하면 margin/ranking 기반 보조 손실을 추가하기

### 완료 기준

- `e_pos`, `e_neg`가 상수값으로 일찍 고정되지 않을 것
- validation Spearman이 epoch 초반 이후에도 의미 있게 변할 것

---

## P3. 로그를 디버깅 가능하게 확장

### 왜 필요한가

- 지금 로그는 실패 결과는 보여주지만, 실패 이유를 정량적으로 분해하기 어렵다.

### 수정 대상

- `scripts/train.py`
- `src/models/irl.py`
- `src/models/ebm.py`

### 추가 로그 항목

- `cd_loss`
- `reg_loss`
- `e_pos std`, `e_neg std`
- `x_neg min/max/mean`
- FM sample energy
- reward term gradient norm
- policy term gradient norm
- best validation epoch

### 완료 기준

- 다음 실험에서 “왜 망했는지”를 로그만으로 1차 판별할 수 있을 것

---

## P4. 평가 metric 해석 정리

### 왜 필요한가

- 현재 `B`는 Spearman은 괜찮은데 AUROC가 낮다.
- 이는 모델 실패일 수도 있지만, score sign이나 label 정의가 실험 목적과 어긋났을 수도 있다.

### 수정 대상

- `src/evaluation/metrics.py`

### 해야 할 일

- `energy`와 `-energy` 모두로 AUROC를 비교해보기
- uncertainty 예측 metric과 malignancy 분류 metric을 분리해서 보고하기
- 보고서에는 “clinical uncertainty metric”과 “malignancy classification metric”을 따로 표기하기

### 완료 기준

- AUROC 해석 방향이 명확해질 것
- B 결과의 해석 혼선을 줄일 것

---

## P5. 하이퍼파라미터 재탐색

### 전제

- P0~P2 수정 후 진행하는 것이 맞다.
- 구조가 잘못된 상태에서의 튜닝은 시간 대비 효율이 낮다.

### 우선 순위

1. `l2_reg`
2. `sgld_step_size`
3. `sgld_steps`
4. `reward_steps_per_iter`
5. `fm_steps_per_iter`
6. `sgld_noise_scale`

### 완료 기준

- A는 포화 완화
- C는 A와 다른 reward dynamics 확보
- B 대비 열세 이유를 하이퍼파라미터 관점에서도 설명 가능

---

## 권장 작업 순서

1. `P0` 수정
2. `P1` 수정
3. `P2` 수정
4. `P3` 로그 확장
5. 소규모 smoke test
6. `P5` 짧은 재탐색
7. `P4` 평가 해석 정리

---

## 메모

- 지금 단계에서 가장 위험한 실수는 “구조 문제를 튜닝 문제로 착각하는 것”이다.
- 따라서 첫 실험 재개 전에는 반드시 C의 gradient path와 reward update 정의를 먼저 고쳐야 한다.
