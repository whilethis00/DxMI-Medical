# April Week 3 (2026-04-13 ~ 2026-04-19)

## 이전 주(W2) 결과 요약

| 실험 | Spearman ρ | p-value | AUROC (W2 로그) | ECE | 판정 |
|------|------------|---------|-----------------|-----|------|
| A (No IRL) | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B (Supervised reward) | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C (MaxEnt IRL) | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

> **주의**: W2 AUROC는 구버전 metrics.py (`score=-energy` 단일) 기준. 현재 숫자 신뢰 보류.
> Apr 14 수정 후 metrics.py는 `auroc_neg_energy`/`auroc_energy` 둘 다 계산하나, 기존 체크포인트 재평가가 필요함.

---

## Apr 14 코드 수정 현황 (미커밋)

`git diff HEAD` 기준 8개 파일, +716 / -240 라인.

| 파일 | 주요 변경 |
|------|----------|
| `src/models/irl.py` | `_policy_sample()`, `_freeze_ebm_params()`, `_grad_norm()` 추가. `update_reward()` SGLD→FM policy sample 교체. DDP 지원. 로그 강화 |
| `src/models/flow_matching.py` | `rollout()` 추가 (Euler integration, autograd 유지) |
| `src/models/ebm.py` | EBM loss 일부 수정 |
| `scripts/train.py` | 로깅 강화, DDP 지원 |
| `configs/*.yaml` | `l2_reg`, `sgld_step_size` 등 조정 |
| `src/evaluation/metrics.py` | `auroc_neg_energy` / `auroc_energy` 둘 다 계산, `EvalResult.__str__()`에 양쪽 출력 |

---

## Step 1 (병렬) — 지금 바로 해야 할 것

### Step 1a: 기존 체크포인트 AUROC 재평가

**왜**: W2 로그는 구버전 metrics.py로 찍힘. 현재 `evaluate()`는 두 AUROC를 모두 반환하지만 기존 체크포인트를 새 코드로 재평가한 적이 없음.

**할 것**:
- B best checkpoint (`outputs/supervised_reward/epoch_080.pt`) 로드
- C best checkpoint (`outputs/irl_maxent/epoch_180.pt`) 로드
- A best checkpoint (`outputs/ebm_baseline/epoch_010.pt`) 로드
- 각각 val set에 `evaluate()` 실행 → `auroc_neg_energy`, `auroc_energy` 둘 다 기록
- Spearman ρ 방향(양수)과 일치하는 쪽을 convention으로 고정

**합격 기준**:
- B: `auroc_energy` ≈ 0.67 (예상) — Spearman ρ>0과 일치 확인
- 이후 모든 로그에서 AUROC는 energy/neg-energy 둘 다 찍는 것을 convention으로 고정

---

### Step 1b: Gradient path smoke test (CPU, 2-step)

**왜**: Apr 14 수정은 단순 리팩터가 아니라 학습 그래프 자체를 바꿨음. 검증 없이 GPU 돌리면 낭비.

**검증 항목 — 누구에게, 어느 loss에서, 어느 경로로 grad가 가야 하는가:**

**update_reward() 검증**
```python
irl.update_reward(x_demo)
# 이후 확인:
assert all(p.grad is not None and p.grad.abs().sum() > 0
           for p in raw_ebm.parameters())          # EBM에 grad 있어야 함
assert all(p.grad is None or p.grad.abs().sum() == 0
           for p in raw_vf.parameters())           # VF에 grad 없어야 함
```

**update_policy() 검증 — 두 경로 분리**
```python
metrics = irl.update_policy(x_demo)
# 확인:
assert metrics["reward_term_grad_norm"] > 0   # reward 경로가 VF에 실제 grad 보냄
assert metrics["policy_term_grad_norm"] > 0   # fm_loss 경로도 살아 있음
```

**rollout() clamp saturation 확인 (신규 리스크)**

`rollout()` 내부:
```python
x = (x + v * dt).clamp(0.0, 1.0)   # ← boundary에서 grad = 0
```
early training에서 x가 [0,1] 경계에 자주 닿으면 `reward_term_grad_norm`이 거의 0이 됨.

```python
# rollout 직후 saturation rate 체크
clamped_frac = ((x_policy <= 0.0) | (x_policy >= 1.0)).float().mean()
# clamped_frac > 0.3이면 grad path가 사실상 죽어 있을 가능성
```

**합격 기준**:
- `reward_term_grad_norm > 0` (0이면 clamp saturation 또는 graph 단절)
- EBM params: update_reward 후 grad 있음, update_policy 후 grad 없음
- VF params: update_policy 후 grad 있음, update_reward 후 grad 없음
- `clamped_frac < 0.3` (경계 포화율이 낮아야 grad 살아 있음)

**실패 시 대응**:
- `reward_term_grad_norm == 0` → clamp saturation이 원인일 가능성 → `rollout()` 내 `.clamp()` 제거 또는 `tanh` 스케일링 검토
- EBM leak detected → `_freeze_ebm_params()` 로직 재확인

---

## Step 2: Saturation mini-run (Step 1b 완료 후)

**목표**: "포화가 났는가" 확인이 아니라 "포화 원인이 무엇인가" 분해.

**필수 로그**:
- `e_pos`, `e_neg` mean
- `e_pos_std`, `e_neg_std`
- pos/neg energy histogram per epoch (겹침 영역 비율)
- epoch별 val Spearman curve
- `reward_term_grad_norm` — 0에 가까우면 clamp 또는 graph 문제
- FM policy sample energy 분포 (`fm_sample_energy`)

**원인 분기**:

| 관찰 | 진단 | 대응 |
|------|------|------|
| histogram overlap 즉시 0 | FM trivial negative | policy sample hardness 강화 or SGLD hybrid |
| std 죽고 평균만 극단 | L2 reg 과도 | `l2_reg` 낮춤 |
| std 살아있는데 overlap 0 | contrastive signal 너무 쉬움 | negative quality 문제 |
| `reward_term_grad_norm` ≈ 0 | clamp saturation or graph 단절 | rollout clamp 수정 |

**FM trivial negative 가설 (핵심 리스크)**:
`update_reward()`의 FM policy sample은 early training에서 Gaussian noise에 가까운 패치.
EBM 입장에서 real nodule patch vs noise patch는 trivially easy separation → saturation과 동일한 결과.
`fm_sample_energy` 로그와 histogram overlap으로 이 가설을 직접 확인할 수 있음.

**합격 기준**:
- `e_pos/e_neg`가 극단값으로 즉시 수렴하지 않음
- val Spearman이 epoch 초반 이후에도 의미 있게 움직임
- energy histogram overlap이 0이 되기까지 최소 20 epoch 이상 걸림

---

## Step 3: Full retrain (Step 1, 2 완료 후)

- A/C 재훈련 (수정된 코드 기준)
- `l2_reg`, `sgld_step_size` 조정은 Step 2 진단 결과에 따라 결정
- FM trivial negative 확인되면 negative 전략 수정 후 재훈련

---

## 하지 말 것

- `l2_reg` 하나 낮추고 full retrain — 원인 분해 없이 운에 맡기는 것
- Step 1 건너뛰고 GPU 잡 먼저 확보 — 측정도 그래프도 검증 안 된 상태
- 현재 W2 AUROC 숫자로 결론 내리는 것 — convention 확정 전까지 신뢰 불가

---

## 목표 지표

| 실험 | 합격 조건 |
|------|----------|
| A < B | baseline 비교 |
| B < C | 제안 방법 우위 |
| Clinical | Spearman ρ(energy, disagreement), p < 0.05 |

타겟: ICLR 2027 (제출 마감 2026년 9월 말)
