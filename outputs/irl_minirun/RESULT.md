# irl_minirun (Gated hybrid — 1차 시도) — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-17 |
| **베이스** | ablation_C (irl_maxent) EBM collapse 원인 격리 후 수정 시도 |
| **목적** | Gated hybrid negative strategy 검증 — Phase 1(SGLD) → Phase 2(FM hybrid) 전환 |
| **설정** | l2_reg=0.01, fm_gate_sep_std_threshold=10.0, SGLD warm-up → FM hybrid, 20ep |
| **현황** | 학습 완료 (20ep), val 완료. 전 epoch FAIL. |

---

## 2. 무엇을 검증하나

ablation C에서 EBM이 step 20에 binary collapse → ranking 신호 전무.
Gated hybrid 전략(Phase 1 SGLD warm-up 후 FM 투입)이 collapse를 방지하는지:

- `e_pos_std > 0.5` ep200 이후 유지 — ranking dynamics 확보 여부
- `Spearman ρ > 0`, `p < 0.05` — clinical correlation 최소 신호
- FM gate 전환이 의도한 시점에 발생하는지

---

## 3. 학습 손실 곡선

**Phase 1 — SGLD warm-up (step 10 ~ 2800)**

| step | e_pos | e_neg | e_pos_std | e_neg_std | sep_std_ema |
|-----:|:-----:|:-----:|:---------:|:---------:|:-----------:|
| 10 | -103 | -12 | 3.53 | 3.70 | 15.9 |
| 20 | -47 | +54 | 0.71 | 0.21 | 6.3 |
| 200 | -50 | +50 | 0.023 | 0.014 | 134 |
| 2800 | -50 | +50 | 0.022 | 0.006 | 4.4 |

**Phase 2 — FM hybrid (step 3000~3800, fm_enabled=1)**

| step | e_pos | e_neg | e_pos_std | reward_grad_norm |
|-----:|:-----:|:-----:|:---------:|:----------------:|
| 3000 | +16 | +33 | 22.5 | **935** |
| 3200 | -52 | +49 | 0.41 | 32 |
| 3800 | -50 | +50 | 0.11 | 3.2 |

**관찰:**
- step 20에서 EBM 붕괴 완료. e_pos≈−50 / e_neg≈+50 이후 2800 step 변화 없음.
- Phase 2 진입 충격(grad_norm=935) → 2 step 만에 ±50 재붕괴.
- sep_std_ema 수치 불안정 (0/0 division): `sep = 0 / 0` → 4~387 무작위 진동. gate가 수치 artifact로 열림.

---

## 4. 검증 지표

| epoch | Spearman ρ ↑ | p-value | AUROC(E) | ECE ↓ | 판정 |
|------:|:------------:|:-------:|:--------:|:-----:|:----:|
| 5 | -0.0030 | 0.9689 | 0.4157 | 0.8023 | FAIL |
| 10 | 0.0564 | 0.4563 | 0.5169 | 0.8023 | FAIL |
| 15 | -0.0252 | 0.7393 | 0.4485 | 0.5300 | FAIL |
| 20 | 0.0480 | 0.5256 | 0.4915 | 0.8023 | FAIL |

---

## 5. 결과 해석 및 인사이트

### 주의
- **전 epoch FAIL.** Gated hybrid 구조 자체는 코드상 의도대로 동작하나, EBM이 Phase 1에서 이미 collapse → Phase 2 진입 무의미.
- **1차 병목:** EBM collapse가 FM보다 먼저. "FM gate 실패"가 아니라 EBM objective/dynamics 자체가 문제.
- **gate metric 버그:** `sep_std = sep / avg_std`에서 양쪽 모두 0에 가까울 때 수치 불안정 → `+ 1e-3` epsilon 필요.

### 근본 원인 (확정)
1. `l2_reg=0.01` → EBM gradient 균형점이 e_pos=−50 / e_neg=+50 → step 20 붕괴
2. gate metric 0/0 불안정 → Phase 2 진입이 FM quality 개선이 아닌 수치 노이즈에 의존
3. Phase 2 진입 후에도 FM 샘플을 즉시 이진 분리 → 재붕괴

---

## 6. 다음 스텝

- [x] gate metric epsilon 수정 (`avg_std + 1e-3`)
- [x] l2_reg 0.01 → 0.1 로 조정
- [x] ebm_weighted_cd 실험에서 collapse 돌파 확인 (ρ=0.3041, PASS)
- [ ] FM gate 재설계 (sep_std_ema 방향 버그 수정)

---

## 7. 저장 파일 목록

```
outputs/irl_minirun/
├── ckpt_epoch0005.pt ~ ckpt_epoch0020.pt  ← (git 제외)
└── RESULT.md
outputs/irl_minirun.log  ← 전체 step/val 로그
```
