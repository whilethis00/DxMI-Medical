# April Week 3 (2026-04-13 ~ 2026-04-19)

## 이전 주(W2) 결과 — W2 로그 vs 재평가 결과

### W2 로그 (구버전 metrics.py, `score=-energy` 단일)

| 실험 | Spearman ρ | p-value | AUROC (W2) | ECE | 판정 |
|------|------------|---------|------------|-----|------|
| A (No IRL) | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B (Supervised reward) | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C (MaxEnt IRL) | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

### 재평가 결과 (`scripts/reeval_checkpoints.py`, 2026-04-17)

| 실험 | Spearman ρ | p-value | AUROC(-E) | AUROC(E) | ECE | 판정 |
|------|------------|---------|-----------|----------|-----|------|
| A best (epoch10) | -0.0878 | 0.2453 | 0.6116 | 0.3884 | 0.7956 | FAIL |
| B best (epoch80) | **0.3028** | **0.0000** | 0.3281 | **0.6719** | 0.2095 | **PASS** |
| C best (epoch180) | 0.1076 | 0.1541 | 0.5142 | 0.4858 | 0.7956 | FAIL |

**AUROC convention 확정: `score = energy` (AUROC(E)) 사용**
- B: ρ=+0.303 (양수) + AUROC(E)=0.672 → 일관됨, 유의미한 결과
- W2 로그의 B AUROC 0.33은 sign 반전 오류. 실제 분류 성능은 0.67

**재평가 핵심 발견**:
- B는 W2 로그보다 훨씬 강함: ρ=0.303, p<0.0001, AUROC(E)=0.672
- C는 여전히 전 항목 실패: ρ=0.108, p=0.154, AUROC(E)=0.486
- A/C 모두 ECE=0.7956 — 동일한 포화 패턴 확인

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

### Step 1a: 기존 체크포인트 AUROC 재평가 — **완료**

`scripts/reeval_checkpoints.py` 실행 결과:

- B: AUROC(E)=**0.6719**, ρ=0.3028 — 예상대로 sign 반전 확인, 실제로 강한 결과
- C: AUROC(E)=0.4858, ρ=0.1076 — 전 항목 실패 유지
- **convention 확정**: `score = energy` (AUROC(E)) 사용

이후 모든 로그에서 `auroc_energy` 기준으로 보고.

---

### Step 1b: Gradient path smoke test — **완료 (구조 PASS)**

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

`scripts/smoke_test.py` 실행 결과 (`[Test 1/2]` — random weight, 구조 검증):

- `[PASS]` update_reward: EBM grad O, VF grad X
- `[PASS]` update_policy: VF grad O, EBM grad X
- `[PASS]` reward_term_grad_norm > 0 (1.4e-02)
- `[PASS]` policy_term_grad_norm > 0 (1.1e+00)

**grad path 구조는 의도대로 작동. Apr 14 수정 정상.**

clamp/negative quality 테스트는 random weight라 결과가 실행마다 다름 — 실제 훈련된 checkpoint로 재측정 필요 (`Step 2`에서 진행).

---

## Step 2: 훈련된 C checkpoint 분리 난이도 진단 — **완료**

`scripts/diagnose_c.py` 실행 결과 (2026-04-17):

| 항목 | 값 | 판정 |
|------|-----|------|
| clamp hit ratio (final) | 12.5% | OK |
| demo energy | mean=-5.003, std=0.003 | — |
| policy energy | mean=+4.751, std=0.147 | — |
| sep/std ratio | **129.9** | FAIL (>3) |
| overlap coefficient | **0.0000** | FAIL (완전 분리) |
| Spearman ρ (ranking) | 0.1076, p=0.154 | FAIL |

**원인 확정: FM trivial negative.**
- EBM이 demo(real nodule)와 policy sample(FM rollout)을 완전히 이진 분리
- IRL reward signal 자체가 의미 없는 상태 — FM이 어떤 샘플을 내놔도 EBM은 즉시 +5로 찍음
- ranking이 아니라 "real vs not-real" binary separation 문제로 붕괴

**수정 완료: Gated hybrid negative strategy (`src/models/irl.py`)**

설계 (3번 기반 gated hybrid):
1. **Phase 1 (warm-up)**: SGLD-only — FM이 negative 자격을 얻기 전까지
2. **전환 gate**: sep/std EMA < `fm_gate_sep_std_threshold` (기본 10.0)를 `fm_gate_consecutive` (기본 3)회 연속 통과
3. **Phase 2 (hybrid)**: `sgld_permanent_ratio` (기본 20%) SGLD 끝까지 유지, 나머지 FM

EMA 스무딩(`sep_std_ema_alpha=0.1`)으로 배치 noise 방지. 전환 이벤트 로그에 기록.

## Step 2 이후: Saturation mini-run

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

---

## ebm_weighted_cd 강한 mini-run + 재현 2회 + 평가 파이프라인 확립 (2026-04-18)

### ebm_weighted_cd 강한 mini-run 결과 (오전)

| 지표 | 값 | 판정 |
|------|-----|------|
| Spearman ρ | 0.3041 | PASS (p=0.0000) |
| AUROC(E) | 0.7269 | — |
| ECE | 0.7945 | 경고 |
| EBM 상태 | e_pos≈-5, e_neg≈+5, e_pos_std≈0.33 | 정상 (collapse 없음) |
| fm_enabled | 0.0 (SGLD-only Phase 1) | FM 미진입 |

### 결과 요약

| 지표 | 값 | 판정 |
|------|-----|------|
| Spearman ρ | 0.3041 | PASS (p=0.0000) |
| AUROC(E) | 0.7269 | — |
| ECE | 0.7945 | 경고 |
| EBM 상태 | e_pos≈-5, e_neg≈+5, e_pos_std≈0.33 | 정상 (collapse 없음) |
| fm_enabled | 0.0 (SGLD-only Phase 1) | FM 미진입 |

### 판정 (연구자) — mini-run

> 이건 승리 선언 단계가 아니라, **드디어 실패 원인을 뚫고 유효한 학습 구간에 들어왔다는 첫 강한 증거**다.

- 진전: epoch 20에서 C가 B best(epoch 80, ρ=0.3028)를 소폭 초과. `e_pos_std≈0.33`으로 demo 내부 분산 살아남 → ranking 신호 생성 시작.
- 유보: `fm_enabled=0` SGLD-only 상태에서의 결과. "FM hybrid 성공"이 아니라 "EBM collapse 돌파 + reward-aware ranking 가능"을 확인한 것.
- 경고: ECE=0.7945 vs B의 0.2095 — ordering은 맞기 시작했지만 confidence scale 못 믿음. 논문에서 그냥 넘어가면 안 됨.

### FM gate 버그 확인

`src/models/irl.py:263`:
```python
if self._sep_std_ema < self.cfg.fm_gate_sep_std_threshold:  # threshold=10.0
```
`sep_std_ema≈98~104`이므로 조건이 절대 참이 안 됨 → **FM gate가 영원히 안 열림**.  
EBM이 잘 학습될수록 sep_std_ema가 오히려 높아지는 구조적 문제.  
threshold 방향 또는 gate metric 자체를 재설계해야 함.

---

### 재현 실험 r1/r2 완료 (2026-04-18 저녁)

동일 설정(`fm_gate_sep_std_threshold=9999`, FM 완전 비활성), seed만 변경.

| 실험 | seed | best Spearman ρ | best epoch | epoch 20 ρ | AUROC(E) ep20 | PASS |
|------|------|----------------|-----------|------------|---------------|------|
| r1 | 1 | 0.2364 (ep14) | 14 | 0.2233 | 0.6653 | ✅ |
| r2 | 2 | 0.2902 (ep17) | 17 | 0.2650 | 0.6767 | ✅ |
| **평균(ep20)** | — | — | — | **0.2442** | **0.6710** | — |

**핵심 관찰**:
- 두 run 모두 epoch 20에서 PASS. "C가 죽어있다" 단계는 끝남.
- seed 간 Spearman ρ 차이 0.042 — seed-sensitive 상태. "한 번 이겼다"와 "방법론적 우위"는 다르다.
- r2가 r1보다 빠르게 수렴(epoch 3부터 PASS, r1은 초반 진동 큼).
- mini-run(ρ=0.3041)을 r1/r2가 안정적으로 재현하지 못함 — 현재 재현 평균 ρ=0.2442.
- last checkpoint ≠ best checkpoint: r1 최고 epoch14, r2 최고 epoch17.
- ECE ≈ 0.793 전 구간 고착 — CD loss는 rank 최적화이고 calibration은 못 함. 지금 핵심 병목 아님.

**냉정한 판단 한 줄**:
> **브레이크스루가 아니라, 유망하지만 seed-sensitive한 상태다.**

---

### 코드 변경 (2026-04-18)

#### `scripts/train.py` — `save_best_val` 추가

`train_irl` 루프에 best-val 체크포인트 저장 기능 추가.

```yaml
# config에 추가
logging:
  save_best_val: true   # Spearman ρ 갱신 시마다 ckpt_best_val.pt 저장
```

`save_best_val: true`이면 val Spearman ρ가 갱신될 때마다 `ckpt_best_val.pt` 덮어씌움 + 로그:
```
[epoch N] best_val updated: rho=X.XXXX → ckpt_best_val.pt
```

**왜**: last checkpoint가 best가 아닌 경우를 이미 두 run에서 확인. 선택 규칙을 공정하게 만드는 장치.  
단, `save_best_val`은 선택 규칙이지 우위 증명이 아님. 우위는 test set에서 나와야 한다.

#### `configs/ebm_weighted_cd_r3.yaml` 생성

r2 완전 동일, seed=3, `save_best_val: true` 활성화.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/ebm_weighted_cd_r3.yaml --seed 3
```

#### `scripts/eval_test.py` 생성 — test 1회 평가

```bash
# 단일 실험 (ckpt_best_val.pt 자동 탐색)
python scripts/eval_test.py --config configs/ebm_weighted_cd_r3.yaml

# A/B/C 한번에 비교
python scripts/eval_test.py \
    --config configs/ebm_baseline.yaml \
            configs/supervised_reward.yaml \
            configs/ebm_weighted_cd_r1.yaml \
            configs/ebm_weighted_cd_r2.yaml \
            configs/ebm_weighted_cd_r3.yaml
```

비교 테이블 출력:
```
── 비교 요약 ──────────────────────────────
실험                           Spearman ρ    p-value   AUROC(E)      ECE
...
```

`save_best_val`이 선택 규칙이면, `eval_test.py`가 판정이다.

---

### 평가 파이프라인 확정

```
학습 → val에서 best 선택(ckpt_best_val.pt) → test 1회 평가(eval_test.py) → 보고
```

- A/B/C 전부 같은 규칙으로 best-val 선택 후 test 평가해야 공정한 비교
- "val에서 고른 best를 test에서 비교"해야 논문 문장이 된다
- 3개 seed 평균±std는 test 기준으로 보고

---

### 다음 우선순위 (W3 종료 시점 기준)

1. **r3 실행** (seed=3, `save_best_val: true`) — 신뢰구간 확보
2. **r3 완료 후 `eval_test.py`로 A/B/C test 비교** — best-val 기준, test 1회
3. **3개 seed 평균±std 확정** → "C > B 방법론적 우위" 문장 가능 여부 판단
4. **FM gate는 그 다음** — seed-sensitive 상태에서 FM 열면 원인 분리 다시 무너짐
5. **ECE는 더 나중** — post-hoc temperature scaling, 지금 핵심 아님
