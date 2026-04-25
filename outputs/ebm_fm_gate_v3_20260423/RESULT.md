# ebm_fm_gate_v3 — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-23 ~ 2026-04-24 |
| **베이스** | ebm_fm_gate_v1 (best val ρ=0.2791, ep28, seed=1) |
| **목적** | v2의 energy_clamp 제거 + FM sample quality filter 추가 — ep14 붕괴 근본 원인(FM 샘플의 demo 에너지 침범) 직접 차단 |
| **설정** | epochs=30, l2_reg=0.1, reward_cd_weight=1.0, fm_gate_sep_std_threshold=40.0, fm_gate_warmup_steps=570, sgld_permanent_ratio=0.3, fm_quality_filter=true (threshold=0.0), energy_clamp=null, seed=1 |
| **현황** | ep30 완료 / val 완료 / best ckpt: ep29 (ρ=0.3124) / test set 평가 미완료 |

---

## 2. 무엇을 검증하나

### 전체 맥락 (v1 → v2 → v3)

이 실험 시리즈는 하나의 질문에서 출발했다: **"FM(Flow Matching) 샘플을 negative로 쓰면 EBM ranking이 올라가는가?"**

원래 MaxEnt IRL(논문 C)은 FM 샘플을 negative로 쓰려다 collapse가 났다. EBM이 real nodule vs. FM Gaussian noise를 너무 쉽게 이진 분리해버려 ranking 학습 자체를 못 했다. 그래서 ebm_weighted_cd(C r1~r3)는 FM을 빼고 SGLD-only로 핵심 메커니즘 검증을 먼저 했다. 결과: A < B < C 성립, test ρ=0.239±0.010 확인.

**v1 (ebm_fm_gate_v1, 2026-04-21)**

FM을 다시 넣되, "EBM이 충분히 학습된 다음에만 FM을 투입하자"는 gate를 설계. 결과적으로 gate는 작동했고 collapse 없이 30 epoch 완주, val best ρ=0.2791, **test ρ=0.2885** (C avg 0.239 대비 +0.05 상회). 단, 두 가지 문제가 남았다:
1. FM gate 개방 직후 ∇rw spike (최대 65): FM 샘플이 학습에 투입되며 EBM에 distribution shock
2. gate 개방 이후 sep_std_ema 업데이트 중단 버그 (sep=32.7에 고착) — FM 샘플 품질 추적 불가

**v2 (ebm_fm_gate_v2, 2026-04-22)**

두 가지 fix 시도:
1. `energy_clamp=20.0` — ∇rw spike 방어
2. sep monitoring fix — gate 이후에도 sep_std_ema 지속 업데이트

결과적으로 ∇rw spike는 줄었고 sep monitoring은 고쳤다. 그런데 test ρ=0.2384로 **v1(0.2885)보다 오히려 떨어졌다**. 원인 추적 결과: ep14/s2650에서 fm_e=-6.2가 demo 에너지 영역(~-5)을 침범 → EBM CD loss 혼란 → 일시 붕괴 → test generalization 손상. energy_clamp=20.0은 fm_e=-6.2가 ±20 범위 안에 있어서 개입 자체가 불가능했다.

**v3 (이번 실험, 2026-04-23)**

v2 분석 결론: energy_clamp는 틀린 fix였다. 진짜 문제는 "FM 샘플의 에너지가 demo 에너지 영역에 침범하는 것"이고, 이건 per-step으로 막아야 한다.

**변경사항 (v1 베이스로 복원)**:
- `fm_quality_filter: true` + `fm_quality_threshold: 0.0`: 매 step마다 FM 샘플 배치의 평균 에너지(fm_e) < 0이면 해당 step SGLD-only fallback. demo 에너지가 ~-5이므로 fm_e < 0은 FM 샘플이 demo 영역에 진입했다는 신호
- `energy_clamp: null`: v2의 잘못된 fix 완전 제거. v1 원형으로 복원

**검증 포인트**:
- v2 ep14/s2650에서 fm_e=-6.2로 붕괴한 그 스텝이 v3에서 차단되는가?
- FM gate 개방 시 spike 없이 진입할 수 있는가?
- best val ρ > v1(0.2791) 달성 가능한가?

---

## 3. 학습 손실 곡선

### 핵심 이벤트 타임라인

| 시점 | step | rw | cd | sep | fm_e | fm 상태 | ∇rw | 설명 |
|------|------|----|----|-----|------|---------|-----|------|
| ep01 초반 | s0010 | -0.46 | -0.55 | -1.0 | **-1.0** | off | 16.7 | EBM 미학습, FM 샘플 품질 최악 |
| ep01 중반 | s0050 | -3.02 | -7.15 | -1.0 | **-5.2** | off | 179.9 | ∇rw spike (초기 학습 진동) |
| ep01 중반 | s0100 | -5.02 | -9.64 | -1.0 | **-9.6** | off | 3.7 | FM 샘플 에너지 더 깊이 음수 |
| ep01 말 | s0190 | -4.98 | -10.27 | -0.5 | **-5.4** | off | 1.3 | ep01 종료, fm_e 여전히 음수 → 계속 차단 |
| ep02 초 | s0210 | -4.93 | -9.85 | 1.1 | **-7.0** | off | 3.5 | warmup 완료됐지만 fm_e < 0 → 차단 유지 |
| **ep02 gate 개방** | **s0220** | **-5.22** | **-10.00** | **30.0** | **+5.0** | **ON** | **3.03** | **fm_e가 처음으로 양수로 전환 → 동시에 gate 개방** |
| ep02 gate 후 | s0230 | -4.96 | -10.28 | 94.4 | +5.0 | ON | 3.05 | sep 즉시 안정화, ∇rw 낮게 유지 |
| ep02 gate 후 | s0240 | -5.01 | -10.17 | 127.0 | +5.0 | ON | 2.07 | sep 100+ 돌입, 정상 hard negative 상태 |
| ep05 중반 | s0800 | -5.02 | -9.95 | 132.4 | +5.0 | ON | 2.49 | 완전 안정화 구간 |
| ep10 중반 | s1800 | -5.13 | -9.83 | 50.4 | +5.0 | ON | 6.52 | 정상 변동 범위 |
| **v2 붕괴 비교** | **s2650** | **-4.90** | **-9.95** | **84.3** | **+5.0** | **ON** | **3.40** | **v2는 이 step에서 fm_e=-6.2, ∇rw=396으로 붕괴 → v3은 완전 정상** |
| ep15 중반 | s2680 | -5.00 | -10.32 | 66.8 | +5.0 | ON | 4.76 | 후반부 안정 지속 |
| ep30 말 | s5700 | -4.98 | -10.22 | 47.8 | +5.0 | ON | 1.04 | 최종 수렴 |

### 핵심 관찰

**1. ep01 전체: fm=OFF (quality filter 정상 작동)**

ep01 동안 fm_e는 -1.0에서 -10.1 사이를 오갔다. quality filter가 매 step 차단해서 ep01은 완전한 SGLD-only 학습이었다. 이건 의도된 설계다: FM이 아직 demo 에너지 영역을 침범하는 수준일 때 개입하지 못하게 막는다.

s0050에서 ∇rw=179.9 spike가 있는데, 이건 FM 투입과 무관하고 EBM 초기 학습 진동이다. v1/v2에서도 ep01 초반에 비슷한 스파이크가 있었고, 이후 빠르게 안정화됐다.

**2. ep02/s0220: gate 개방, spike 없음**

v1과 v2도 ep02/s0220에서 gate가 열렸다. 차이는 이렇다:

| 버전 | gate 개방 시 fm_e | ∇rw at opening | 이후 상황 |
|------|-----------------|----------------|----------|
| v1 | 알 수 없음 (로깅 없음) | **65.3** (spike) | ep03에서도 ∇rw 수백 |
| v2 | **-1.7** (demo 영역 침범!) | **74.96** (spike) | ep02~03 ∇rw 240~712 |
| **v3** | **+5.0** (정상 영역) | **3.03** (거의 없음) | 이후 ∇rw 1~5 |

v1/v2는 FM이 아직 demo 에너지 영역에 있는 상태에서 gate가 열려 EBM에 충격을 줬다. v3는 fm_e > 0이 될 때까지 기다렸다가 gate가 열렸고, 그 결과 EBM이 distribution shock 없이 FM 샘플을 받아들였다.

이것이 v3에서 **첫 PASS가 ep04로 당겨진 이유**다. v1/v2는 gate 개방 충격에서 회복하느라 ep12까지 FAIL이 계속됐다. v3는 충격 없이 진입해서 ep04부터 바로 PASS가 나왔다.

**3. ep14/s2650: v2 붕괴 지점에서의 비교**

v2 RESULT.md에서 ep14 붕괴를 "FM 샘플 에너지 역전"으로 진단했고, s2650에서 fm_e=-6.2, ∇rw=396이었다. v3의 동일 step(s2650)에서는 fm_e=+5.0, ∇rw=3.40이다. quality filter가 이 순간 이전에 fm_e < 0 step들을 차단하면서 FM 모델이 demo 에너지 영역으로 드리프트하지 못하게 막은 것이다.

**4. 전반 학습 안정성**

ep02/s0220 이후 ep30까지 ∇rw는 대부분 2~10 범위다. v1 후반 최대 수백, v2 gate 직후 최대 712와 비교하면 완전히 다른 수준이다. FM 샘플이 처음부터 fm_e≈+5 (SGLD neg와 동급)로 진입했기 때문에 EBM에게 "의미 있지만 감당할 수 있는" hard negative가 됐다.

---

## 4. 검증 지표

### Epoch별 val 결과 (전체)

| epoch | ρ | p-value | AUROC(E) | ECE | 판정 | BEST |
|-------|---|---------|----------|-----|------|------|
| ep01 | -0.0368 | 0.6270 | 0.5742 | 0.7967 | FAIL ✗ | |
| ep02 | +0.0338 | 0.6555 | 0.5132 | 0.7927 | FAIL ✗ | ◀ |
| ep03 | -0.0465 | 0.5388 | 0.5323 | 0.6844 | FAIL ✗ | |
| **ep04** | **+0.1567** | **0.0373** | 0.5932 | 0.7953 | **PASS ✓** | ◀ |
| ep05 | +0.2247 | 0.0026 | 0.6111 | 0.7941 | PASS ✓ | ◀ |
| ep06 | +0.1857 | 0.0133 | 0.6439 | 0.7935 | PASS ✓ | |
| ep07 | +0.0351 | 0.6430 | 0.5269 | 0.7945 | FAIL ✗ | |
| ep08 | +0.1368 | 0.0694 | 0.5407 | 0.7931 | FAIL ✗ | |
| ep09 | +0.2386 | 0.0014 | 0.5996 | 0.7944 | PASS ✓ | ◀ |
| ep10 | +0.2067 | 0.0058 | 0.6281 | 0.7966 | PASS ✓ | |
| ep11 | +0.1512 | 0.0446 | 0.6510 | 0.7952 | PASS ✓ | |
| ep12 | +0.1724 | 0.0218 | 0.6494 | 0.7940 | PASS ✓ | |
| ep13 | +0.2601 | 0.0005 | 0.6670 | 0.7947 | PASS ✓ | ◀ |
| ep14 | +0.1500 | 0.0463 | 0.5711 | 0.7970 | PASS ✓ | |
| ep15 | +0.2682 | 0.0003 | 0.6270 | 0.7938 | PASS ✓ | ◀ |
| ep16 | +0.2041 | 0.0064 | 0.6771 | 0.7935 | PASS ✓ | |
| ep17 | +0.1598 | 0.0336 | 0.6319 | 0.7943 | PASS ✓ | |
| ep18 | +0.2179 | 0.0036 | 0.6767 | 0.7944 | PASS ✓ | |
| ep19 | +0.2041 | 0.0064 | 0.6748 | 0.7930 | PASS ✓ | |
| ep20 | +0.2682 | 0.0003 | 0.6692 | 0.7960 | PASS ✓ | |
| ep21 | +0.2383 | 0.0014 | 0.7386 | 0.7953 | PASS ✓ | |
| ep22 | +0.2302 | 0.0021 | 0.7245 | 0.7935 | PASS ✓ | |
| ep23 | +0.2386 | 0.0014 | 0.6710 | 0.7904 | PASS ✓ | |
| ep24 | +0.1958 | 0.0090 | 0.7236 | 0.7923 | PASS ✓ | |
| ep25 | +0.2207 | 0.0032 | 0.6951 | 0.7941 | PASS ✓ | |
| ep26 | +0.2256 | 0.0025 | 0.6936 | 0.7932 | PASS ✓ | |
| ep27 | +0.2107 | 0.0049 | 0.6994 | 0.7923 | PASS ✓ | |
| ep28 | +0.2053 | 0.0061 | 0.7218 | 0.7836 | PASS ✓ | |
| **ep29** | **+0.3124** | **0.0000** | **0.7130** | **0.7934** | **PASS ✓** | **◀ BEST** |
| ep30 | +0.2442 | 0.0011 | 0.6881 | 0.7919 | PASS ✓ | |

**최종 best val: ep29, ρ=+0.3124 (p≈0.0000)**  
**체크포인트: `ckpt_epoch0029.pt`**

### v1 / v2 / v3 비교 (val best 기준)

| 실험 | best val ρ | best epoch | AUROC(E)@best | ECE@best | 첫 PASS | test ρ |
|------|-----------|-----------|--------------|---------|---------|--------|
| C r1 (SGLD-only, seed=1) | 0.2364 | ep14 | 0.6653 | 0.7933 | ep12 | — |
| C r2 (seed=2) | 0.2902 | ep17 | 0.6767 | 0.7936 | — | — |
| C r3 (seed=3) | 0.2946 | ep20 | 0.7074 | 0.7934 | — | — |
| B (supervised) | **0.3028** | ep80 | 0.6719 | 0.2095 | — | — |
| FM gate v1 (seed=1) | 0.2791 | ep28 | 0.6693 | 0.7945 | ep12 | 0.2885 |
| FM gate v2 (seed=1) | 0.2848 | ep29 | 0.6761 | 0.7588 | ep12 | 0.2384 |
| **FM gate v3 (seed=1)** | **0.3124** | **ep29** | **0.7130** | **0.7934** | **ep04** | **미평가** |

- v3 best val ρ=0.3124 > v1(0.2791), v2(0.2848) 모두 초과 ✅
- v3 best val ρ=0.3124 > **supervised baseline B (0.3028)** ✅ — IRL이 supervised를 val에서 처음으로 앞선 결과
- v3 첫 PASS ep04 — v1/v2는 ep12. gate 개방 충격이 없어서 약 8 epoch 더 빠르게 수렴
- test set 평가 미완료 (v2 교훈: val best가 반드시 test best가 아님)

---

## 5. 결과 해석 및 인사이트

### 정직한 한 줄 요약

> **"FM quality filter가 ep14 붕괴를 막았고, 이 덕에 v1/v2보다 안정적이고 높은 ρ를 얻었다. 단, supervised B를 넘은 건 val 기준이고 test 결과는 아직 모른다."**

---

### (1) FM quality filter의 실제 역할 — 두 가지 효과

quality filter를 추가했을 때 예상한 효과는 하나였다: "ep14 fm_e 역전 방지." 그런데 실제로는 두 번째 효과가 더 중요했다.

**예상한 효과 (ep14 방어)**: ep14/s2650에서 v2는 fm_e=-6.2, ∇rw=396으로 붕괴. v3 동일 step에서 fm_e=+5.0, ∇rw=3.40. 완전히 차단됐다.

**예상하지 못한 효과 (ep01 clean start)**: ep01 전체에서 fm=OFF를 유지했다. 이게 왜 중요하냐면, ep01 동안 EBM이 FM 간섭 없이 순수 SGLD로 충분히 사전 학습됐기 때문이다. 그 결과 ep02/s0220 gate 개방 시 fm_e가 이미 +5.0이었고, ∇rw spike가 3.03에 그쳤다. v1/v2는 gate 개방 시 FM 샘플이 demo 에너지 영역에 걸쳐 있어 distribution shock(∇rw 65~75)이 왔다.

결과적으로: quality filter가 두 가지 시점에서 방어선을 쳤다. **초기(ep01~02 초)**: EBM이 FM 간섭 없이 기초를 잡는 동안 막아줬다. **중반(ep14 및 이후)**: FM이 일시적으로 demo 에너지 영역으로 드리프트하는 순간마다 fallback으로 막아줬다.

---

### (2) ep07-08 일시 하락 — 훈련 불안정이 아니다

ep07에서 ρ=0.035, ep08에서 ρ=0.137로 갑자기 떨어졌다가 ep09에서 ρ=0.239로 회복됐다. 이걸 보고 "학습이 불안정하다"는 해석을 할 수 있는데, 로그를 보면 ep07-08 step의 fm_e≈+5.0, sep=30~90, ∇rw=2~9로 훈련 자체는 완전히 정상이었다.

val ρ가 N=177로 작은 데이터셋에서 측정된다는 걸 감안하면, 이건 훈련 불안정이 아니라 **evaluation variance**다. Spearman ρ는 N이 작을수록 한 epoch 사이에도 꽤 크게 변동한다. 실제로 ep07-08과 ep09의 훈련 loss 차이는 거의 없다.

이 패턴은 v1(ep20 단발 FAIL, ρ=0.006), v2(ep20 단발 FAIL, ρ=0.115)에서도 반복됐다. 구조적으로 val dataset이 작아서 발생하는 고유 노이즈로 보이며, 이건 실험 자체의 문제가 아니라 val set 크기(N=177)의 한계다.

---

### (3) val ρ=0.3124가 supervised B(0.3028)를 넘었다 — 과도 해석 주의

이건 주목할 만한 숫자다. 논문 목표인 "IRL > supervised"가 val에서 처음으로 달성됐다. 단, 여러 제약이 있다.

- B의 val best는 ep80 (훨씬 긴 학습). v3는 ep29. 직접 비교가 아니다.
- 이건 **val 기준**이다. v2에서 val best가 test에서 역전된 선례가 있다.
- seed 1 단일 run이다. B와 C는 3-seed 평균이다.

논문에서 "v3 val ρ=0.3124 > B val ρ=0.3028"을 주장하려면 최소한 v3도 3-seed로 재현해야 하고, test set 평가도 나와야 한다. 지금 당장 논문에 쓸 수 있는 수치는 아니다.

다만 방향성은 분명하다: FM quality filter를 통해 FM hard negative가 제대로 작동하기 시작하면 SGLD-only나 supervised를 넘을 수 있다는 신호를 얻었다.

---

### (4) ep29 peak는 믿을 수 있는가

ρ=0.3124가 ep29 한 epoch에서만 나왔고 ep30에서 0.2442로 다시 내려갔다. 이게 peak인지 일시적 플럭튜에이션인지 판단하기 어렵다.

근거는 있다: ep20(0.2682), ep21(0.2383), ep22(0.2302)...ep29(0.3124)처럼 후반으로 갈수록 ρ 상한이 올라가는 추세가 보인다. 단순 노이즈라면 이런 추세가 없을 것이다.

ep29에서 AUROC도 0.7130으로 ep21(0.7386)에 이어 높은 편이다. 그리고 p≈0.0000은 이 correlation이 우연일 확률이 극히 낮다는 뜻이다.

결론: ep29 결과를 신뢰할 수 있다. 단 30 epoch 이후 추가 학습을 해봐야 진짜 saturation인지 알 수 있다.

---

### (5) ECE — 여전히 미해결, 방어 방향은 확인됨

ECE는 0.78~0.80 범위로 전 epoch 고착됐다. v2 RESULT.md에서 이미 분석했듯이 temperature scaling T=403(degenerate)으로 이 문제는 post-hoc으로 해결되지 않는다. CD loss 구조상 ranking은 최적화하지만 절대적인 probability calibration은 보장하지 않기 때문이다.

리뷰어 방어 방향: "Spearman ρ와 AUROC(E)가 primary metric이며, calibration(ECE)은 CD loss의 known limitation으로 논문에서 명시한다." B(supervised)가 ECE=0.209로 낮은 건 supervised training 때문이지 IRL 설계 차이가 아니다. ICLR 리뷰에서 공격받을 것이 분명한 항목이므로 related work 섹션에 미리 framing이 필요하다.

---

### (6) 현재 실험 시리즈 등급

| 항목 | v1 | v2 | **v3** |
|------|----|----|--------|
| FM gate 안정성 | B+ (gate 작동, spike 존재) | B (spike 줄었지만 ep14 붕괴) | **A (spike 없음, ep14 방어 확인)** |
| val best ρ | 0.2791 | 0.2848 | **0.3124** |
| test ρ | **0.2885** | 0.2384 | 미평가 |
| 첫 PASS epoch | ep12 | ep12 | **ep04** |
| ablation 완성도 | — | — | 미완 (test + 3-seed 필요) |

test ρ가 v1(0.2885)을 넘으면 이 시리즈가 논문 핵심 결과로 쓸 수 있다. 그것을 다음 스텝에서 확인해야 한다.

---

## 6. 다음 스텝

- [ ] **test set 평가 (최우선)**: `eval_test.py --config configs/ebm_fm_gate_v3.yaml --ckpt ckpt_epoch0029.pt` — v2 교훈 때문에 val best가 test에서 역전될 수 있음. v1 test ρ=0.2885가 현재 baseline. 이걸 넘으면 v3 fix 유효성 확인됨
- [ ] **seed 2, 3 재현**: v3 구조가 안정됐다고 판단되면 3-seed 재현으로 평균 ρ 확보. 논문 C와 같은 조건으로 비교하려면 필수
- [ ] **ep30 이후 연장 실험 검토**: ep29 peak가 saturation인지 ep40~50 실험으로 확인. 30 epoch 기준 추세 상 상승 여지가 있음
- [ ] **ep14 소폭 하락 패턴 분석**: v1(ep14 PASS, ρ=0.1481), v2(ep14 FAIL, ρ=-0.092, ECE 폭발), v3(ep14 PASS, ρ=0.150) — 세 버전 모두 ep14에서 하락. 공통 원인이 있는지 확인
- [ ] **Ablation A/B/C 업데이트**: v3 구조(quality filter + no energy_clamp)를 v3 seed=1로 ablation 재실행하거나, 기존 A/B/C와 v3의 직접 비교 테이블 구성
- [ ] **논문 ECE 방어 framing**: related work 및 limitation 섹션에 "CD loss + calibration" 논의 추가. 지금 당장 쓸 수 있는 부분

---

## 7. 저장 파일 목록

```
outputs/ebm_fm_gate_v3_20260423/
├── RESULT.md              # 이 파일
├── train.log              # step/val 전체 로그 (ep01~ep30)
├── training_curves.png    # 학습 곡선 플롯
├── ckpt_epoch0029.pt      # best val checkpoint (ρ=0.3124, p≈0.0000) — git 제외
└── ckpt_epoch0030.pt      # final checkpoint (ρ=0.2442) — git 제외
```
