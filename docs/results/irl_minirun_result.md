# irl_minirun 결과 분석 (2026-04-17)

실험: Gated hybrid negative strategy 검증 mini-run (20 epoch, Ablation C)
로그: `outputs/irl_minirun.log`

---

## 최종 val 지표

| epoch | Spearman ρ | p-value | AUROC(E) | ECE | 판정 |
|-------|------------|---------|----------|-----|------|
| 5     | -0.0030    | 0.9689  | 0.4157   | 0.8023 | FAIL |
| 10    |  0.0564    | 0.4563  | 0.5169   | 0.8023 | FAIL |
| 15    | -0.0252    | 0.7393  | 0.4485   | 0.5300 | FAIL |
| 20    |  0.0480    | 0.5256  | 0.4915   | 0.8023 | FAIL |

전 epoch FAIL. Spearman ρ는 랜덤(0) 수준에서 벗어나지 못함.

---

## 훈련 dynamics 분석

### Phase 1 — SGLD warm-up (step 10 ~ 2800, fm_enabled=0)

| step | e_pos | e_neg | e_pos_std | e_neg_std | fm_sample_energy | sep_std_ema |
|------|-------|-------|-----------|-----------|------------------|-------------|
| 10   | -103  | -12   | 3.53      | 3.70      | -119             | 15.9        |
| 20   | -47   | +54   | 0.71      | 0.21      | -50              | 6.3         |
| 50   | -51   | +52   | 0.40      | 0.05      | -57              | 43.2        |
| 200  | -50   | +50   | 0.023     | 0.014     | -50              | 134         |
| 1000 | -50   | +50   | 0.014     | 0.009     | -53              | 230         |
| 2800 | -50   | +50   | 0.022     | 0.006     | -50              | 4.4         |

**step 20에서 EBM 붕괴 완료.** e_pos≈-50, e_neg≈+50으로 수렴, 이후 2800 step 동안 변화 없음.
std는 step 200 이후 ≈0.01~0.03 수준으로 사실상 0 — 클래스 내 분산 없음.

### Phase 2 — Hybrid (step 3000~3800, fm_enabled=1)

| step | e_pos | e_neg | e_pos_std | e_neg_std | reward_grad_norm |
|------|-------|-------|-----------|-----------|-----------------|
| 3000 | +16   | +33   | 22.5      | 15.4      | 935             |
| 3200 | -52   | +49   | 0.41      | 0.81      | 32              |
| 3800 | -50   | +50   | 0.11      | 0.021     | 3.2             |

step 3000에서 FM 진입 충격(grad_norm=935). 2 step 만에 ±50으로 재붕괴.

---

## 원인 분석

### 1. EBM contrastive collapse (핵심)

`l2_reg=0.01` 설정 하에서 EBM의 energy 평형점:

```
reg_loss = 0.01 * (e_pos² + e_neg²).mean()
cd_loss  = e_pos - e_neg  (minimize)
```

gradient 균형: `e_pos = -50, e_neg = +50` → reg_loss=50, cd_loss=-100, total=-50.

EBM이 모든 demo 샘플에 -50, 모든 SGLD negative에 +50을 일률 배정.
**클래스 내 분산이 0이므로 랭킹 신호가 전혀 없음.**

SGLD negative는 replay buffer에서 시작해 langevin dynamics로 EBM gradient를 따라 이동하므로,
EBM이 빠르게 배운 decision boundary만 피하면 되는 "쉬운 negative"가 된다.
결과적으로 단 20 step 만에 이진 분리가 완성된다.

### 2. Gate metric 수식 오류

gate 조건: `sep_std = |e_fm - e_pos| / avg_std < 10.0`

Phase 1 전체에서 `fm_sample_energy ≈ e_pos ≈ -50`:
- FM은 demo 분포를 학습했으므로, EBM이 FM 샘플에도 demo-like energy(-50)를 배정
- `sep = |(-50) - (-50)| ≈ 0`
- `avg_std = (e_pos_std + e_fm_std) / 2 ≈ 0`
- `sep_std = 0 / 0` → 수치 불안정

sep_std_ema가 4~387을 무작위로 요동치는 이유. step 2900에서 gate가 열린 건
FM quality 개선이 아니라 **분모가 0에 가까울 때 분자도 우연히 0 근처여서 생긴 수치 artifact**.

### 3. Phase 2 진입 충격 후 재붕괴

FM 샘플 진입 시 EBM이 처음 보는 분포에 반응해 일시적으로 붕괴(step 3000 grad_norm=935).
그러나 이후 FM 샘플도 즉시 이진 분리해버려 다시 ±50으로 수렴.

---

## 인사이트

**이번 실험으로 확인된 것:**
- Gated hybrid 구조 자체(gate 전환 로직, Phase 전환)는 코드상 의도대로 동작
- 그러나 gate가 열리는 조건이 FM quality가 아닌 수치 노이즈에 의존
- EBM이 랭킹을 학습하지 못하는 한, Phase 2 진입은 의미 없음

**이번 실험으로 확인되지 않은 것:**
- Gated hybrid가 trivial negative 문제를 해결하는지 — EBM collapse가 먼저라 테스트 불가

**1차 병목 재확인:**

> EBM이 Phase 1(SGLD only) step 20에서 이미 binary separator로 붕괴했다.
> FM은 주범이 아니라 붕괴된 EBM 위에 얹힌 2차 문제다.
> "FM gate 실패"가 아니라 "EBM objective/dynamics"가 근본 원인.

**다음 수정 우선순위:**

| 순위 | 수정 항목 | 근거 |
|------|-----------|------|
| 1 | **Gate metric epsilon** — `sep_std = sep / (avg_std + 1e-3)` | 버그 수정. 0/0 불안정 제거. 즉시 적용 |
| 2 | **EBM collapse 억제** — `l2_reg` 0.01 → 0.1~0.5, energy scale/temperature 재조정 | step 20 붕괴의 직접 원인. ranking 학습 가능한 dynamics 확보 |
| 3 | (2번으로도 ±50 포화 지속 시) **hard clamp** — `energy.clamp(-10, 10)` | 수치 폭주 억제용 응급처치. collapse 자체를 해결하지는 못함 |

**다음 실험 설계:**

FM을 끄고 SGLD only 상태에서 EBM collapse 잡기가 먼저다.
- `fm_gate_sep_std_threshold` 를 999 등으로 올려 Phase 1만 강제
- `l2_reg` 스윕: 0.1 / 0.3 / 0.5
- 합격 기준: `e_pos_std > 0.5`, `e_neg_std > 0.5` 가 step 200 이후에도 유지

EBM이 ranking을 학습하는 것이 확인된 이후에 FM gate 검증으로 넘어간다.
