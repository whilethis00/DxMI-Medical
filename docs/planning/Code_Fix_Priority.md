# Code Fix Priority

## 목적

- Ablation A/C 실패 원인 분석을 바탕으로, 재실험 전에 반드시 손봐야 할 코드 수정 우선순위를 정리한다.

---

## Priority 1 — IRL policy reward gradient 경로 수정

### 문제

- 현재 `src/models/irl.py`의 `update_policy()`에서 reward term이 `x_approx = 0.01*x0 + 0.99*x_demo`에 대해 계산됨
- 이 `x_approx`는 `VelocityField` 출력에 실질적으로 의존하지 않음
- 따라서 `reward_loss = EBM(x_approx)`가 FM 파라미터를 거의 업데이트하지 못할 가능성이 큼

### 영향

- 제안 방법 C의 핵심인 “reward-guided policy improvement”가 사실상 비활성화될 수 있음
- `fm_loss`는 줄어도 clinical metric이 개선되지 않는 현상과 연결 가능

### 수정 목표

- reward term이 실제 FM 생성 샘플 또는 FM trajectory에 직접 연결되도록 변경
- `reward_loss.backward()` 시 `VelocityField` 파라미터에 gradient가 흐르는지 확인

### 관련 파일

- `src/models/irl.py`

---

## Priority 2 — IRL reward update를 policy sample 기반으로 재구성

### 문제

- 현재 `update_reward()`는 FM policy sample이 아니라 SGLD negative sample로 reward model을 업데이트함
- 결과적으로 Ablation C의 reward update가 Ablation A의 contrastive divergence와 거의 같은 구조가 됨

### 영향

- C가 A와 구별되지 않음
- occupancy matching 또는 MaxEnt IRL 특성이 코드에 충분히 반영되지 않음

### 수정 목표

- demonstration vs policy-generated sample 비교 구조로 reward update 변경
- SGLD negative sample의 역할과 policy sample의 역할을 분리

### 관련 파일

- `src/models/irl.py`
- 필요 시 `src/models/flow_matching.py`

---

## Priority 3 — EBM loss 재설계 또는 regularization 약화

### 문제

- 현재 `src/models/ebm.py` 손실:
  - `loss = E_pos - E_neg + λ(E_pos^2 + E_neg^2)`
- 현재 설정에서는 `e_pos≈-5`, `e_neg≈5` 고정점으로 빠르게 수렴하는 패턴이 로그에서 반복됨

### 영향

- 샘플 간 uncertainty ranking보다 energy magnitude saturation이 먼저 발생
- Spearman ρ와 ECE가 개선되지 않음

### 수정 목표

- `l2_reg`를 더 낮추거나
- margin/ranking 성분을 도입하거나
- supervised auxiliary term을 추가하는 방향 검토

### 관련 파일

- `src/models/ebm.py`
- `configs/ebm_baseline.yaml`
- `configs/irl_maxent.yaml`

---

## Priority 4 — 하이퍼파라미터 재탐색

### 우선 점검 항목

- `l2_reg`
- `sgld_steps`
- `sgld_step_size`
- `sgld_noise_scale`
- `reward_steps_per_iter`
- `fm_steps_per_iter`

### 이유

- 현재 설정은 안정화보다는 포화를 강화하고 있을 가능성이 있음
- 코드 구조 수정 후에도 이 값들이 잘못되면 동일한 실패 패턴이 반복될 수 있음

### 관련 파일

- `configs/ebm_baseline.yaml`
- `configs/irl_maxent.yaml`

---

## Priority 5 — 평가 metric sign 및 해석 검증

### 문제

- 프로젝트 설명은 “energy가 높을수록 disagreement가 커야 함”
- 하지만 AUROC는 `score = -energy`로 계산됨
- Ablation B에서 Spearman ρ는 양호하지만 AUROC가 낮은 점은 해석 mismatch 가능성을 시사

### 수정 목표

- malignancy score에 대해 `energy`와 `-energy`를 모두 비교
- uncertainty metric과 malignancy metric의 역할을 분리해서 보고

### 관련 파일

- `src/evaluation/metrics.py`

---

## Priority 6 — 로깅 강화

### 현재 한계

- 로그에 포화 현상은 보이지만 원인을 더 세밀하게 분리하기 어려움

### 추가하면 좋은 항목

- `cd_loss`
- `reg_loss`
- `energy mean/std`
- `x_neg` 통계
- FM sample energy
- best validation epoch
- early stopping trigger 여부

### 관련 파일

- `scripts/train.py`
- `src/models/ebm.py`
- `src/models/irl.py`

---

## 권장 실행 순서

1. `Priority 1` 수정
2. `Priority 2` 수정
3. `Priority 3` 수정
4. `Priority 6` 로깅 강화
5. `Priority 4` 하이퍼파라미터 재탐색
6. `Priority 5` 평가 해석 검증

---

## 한 줄 정리

- 가장 먼저 고칠 것은 C의 “IRL처럼 보이지만 실제로는 IRL이 아닌 부분”이고, 그 다음이 A/C 공통의 EBM 포화 문제다.
