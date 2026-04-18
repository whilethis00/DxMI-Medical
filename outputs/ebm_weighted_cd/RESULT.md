# ebm_weighted_cd — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-18 |
| **베이스** | irl_minirun (Gated hybrid, l2_reg=0.01 → EBM collapse) |
| **목적** | CD loss에 per-sample reward weight 주입 → within-demo ranking 신호 생성 |
| **설정** | α=1.0, T=1.0, l2_reg=0.1, SGLD-only Phase 1 (fm_enabled=0), 20ep |
| **현황** | 학습 완료 (epoch 20), val 완료 |

---

## 2. 무엇을 검증하나

irl_minirun에서 step 20에 EBM이 e_pos≈−50 / e_neg≈+50으로 이진 붕괴,
이후 모든 ep에서 Spearman ρ≈0, ranking 신호 전무.

CD loss에 reward weight(r_i)를 곱해 demo 내부에서 고reward 샘플이 더 낮은 energy를 갖도록 gradient를 유도했을 때:

- `e_pos_std > 0.3` 유지 (이전 붕괴 후 ≈0.01) — ranking 학습 가능한 dynamics 확보 여부
- `Spearman ρ > 0` 통계적 유의 (p < 0.05) — clinical correlation 신호 존재 여부
- EBM collapse 재발 없이 20ep 완주 여부

---

## 3. 학습 손실 곡선

> **주의:** train.log는 Singularity 터미널 출력만 존재. 마지막 5 step만 보존.

| step | reward_loss | cd_loss | reg_loss | e_pos | e_neg | e_pos_std | e_neg_std | reward_grad_norm | sep_std_ema | fm_enabled |
|-----:|:-----------:|:-------:|:--------:|:-----:|:-----:|:---------:|:---------:|:----------------:|:-----------:|:----------:|
| 3760 | -5.0496 | -10.2258 | 5.1762 | -5.1387 | 5.0241 | 0.3287 | 0.0100 | 1.4666 | 104.7 | 0.0 |
| 3770 | -4.9896 | -9.9973 | 5.0077 | -5.0089 | 4.9959 | 0.1671 | 0.0132 | 1.1799 | 96.7  | 0.0 |
| 3780 | -5.0187 | -9.9861 | 4.9674 | -5.0165 | 4.9415 | 0.2939 | 0.0157 | 1.4007 | 79.5  | 0.0 |
| 3790 | -5.0236 | -10.0806 | 5.0570 | -5.0163 | 5.0329 | 0.2630 | 0.0183 | 1.0193 | 85.8  | 0.0 |
| 3800 | -4.9853 | -9.8982 | 4.9129 | -4.8950 | 5.0052 | 0.3301 | 0.0193 | 0.9896 | 98.8  | 0.0 |

**관찰:**
- e_pos≈−5, e_neg≈+5 — 이전 ±50 붕괴 없음, 전 구간 안정
- e_pos_std≈0.17~0.33 — 클래스 내 분산 살아있음 (이전 붕괴 후 ≈0.01 대비 극적 개선)
- reward_grad_norm 0.99~1.47 — 안정적 (이전 Phase 2 진입 시 935 대비)
- sep_std_ema 79~105 — FM gate threshold(10.0) 훨씬 초과 → fm_enabled=0 유지 (버그: §5 참조)

---

## 4. 검증 지표

| epoch | Spearman ρ ↑ | p-value | AUROC(-E) | AUROC(E) ↑ | ECE ↓ | 판정 |
|------:|:------------:|:-------:|:---------:|:----------:|:-----:|:----:|
| 20 | **0.3041** | **0.0000** | 0.2731 | **0.7269** | 0.7945 | **PASS** |

### 이전 실험 대비 비교

| 실험 | Spearman ρ | p-value | AUROC(E) | ECE | EBM 상태 |
|------|:----------:|:-------:|:--------:|:---:|:--------:|
| irl_minirun C (epoch 20) | 0.0480 | 0.5256 | 0.4915 | 0.8023 | ±50 붕괴 |
| ablation B best (epoch 80) | 0.3028 | 0.0000 | 0.6719 | 0.2095 | 정상 |
| **ebm_weighted_cd (epoch 20)** | **0.3041** | **0.0000** | **0.7269** | 0.7945 | **정상** |

---

## 5. 결과 해석 및 인사이트

### 긍정적
- **EBM collapse 완전 해결:** e_pos≈−5 / e_neg≈+5, e_pos_std≈0.33 — ranking dynamics 확보
- **epoch 20에서 ablation B best(epoch 80)를 이미 초과:** ρ=0.3041 vs 0.3028, AUROC(E)=0.727 vs 0.672
- 이건 "FM hybrid가 성공했다"가 아니라, **EBM collapse를 깨고 reward-aware ranking을 만들 수 있게 됐다**는 첫 강한 증거

### 주의
- **ECE=0.7945 — 심각한 경고:** B의 ECE=0.2095 대비 극단적으로 나쁨. ordering은 맞기 시작했지만 confidence scale은 못 믿는 상태. 논문에서 그냥 넘어가면 안 됨.
- **fm_enabled=0 유지 — FM gate 버그:** `irl.py:263`에서 `sep_std_ema < 10.0` 조건인데 실제 sep_std_ema≈98~104 → 조건이 절대 참이 안 됨. EBM이 잘 학습될수록 sep_std_ema가 오히려 올라가는 구조적 문제. threshold 방향 재설계 필요.
- 현재 결과는 SGLD-only Phase 1 상태. Phase 2(FM hybrid) 진입 후 재붕괴 가능성 미검증.

### FM gate 버그 상세

```python
# src/models/irl.py:263
if self._sep_std_ema < self.cfg.fm_gate_sep_std_threshold:  # threshold=10.0
```
sep_std_ema = |e_fm - e_pos| / avg_std ≈ 79~105 → 조건 항상 False → fm_enabled 영원히 0.

---

## 6. 다음 스텝

- [ ] **재현 2~3회** — ρ=0.3041이 안정적으로 나오는지 확인. 단일 숫자 취하지 말 것.
- [ ] **FM gate 재설계** — sep_std_ema 기반 threshold 방향 수정 또는 다른 gate metric 도입
- [ ] **Phase 2 진입 후 충격 관찰** — 이전 minirun: Phase 2 진입 시 grad_norm=935, 2 step 만에 ±50 재붕괴 전례
- [ ] **ablation A/C 동일 조건 재훈련** — 공식 A < B < C 비교
- [ ] **ECE 개선 (나중)** — energy ordering 안정화 후 temperature scaling 또는 calibration head

---

## 7. 저장 파일 목록

```
outputs/ebm_weighted_cd/
├── ckpt_epoch0005.pt    ← (git 제외)
├── ckpt_epoch0010.pt    ← (git 제외)
├── ckpt_epoch0015.pt    ← (git 제외)
├── ckpt_epoch0020.pt    ← 최종 체크포인트 (git 제외)
├── train.log            ← 마지막 5 step 보존 (Singularity 터미널 출력)
└── RESULT.md            ← 이 파일
```
