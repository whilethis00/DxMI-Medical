# ablation_C (irl_maxent) — Result

## 1. 실험 메타

| 항목 | 내용 |
|------|------|
| **날짜** | 2026-04-05 |
| **베이스** | ablation_B (supervised_reward) |
| **목적** | MaxEnt IRL로 reward 역추론 → supervised B 대비 우위 여부 (핵심 ablation) |
| **설정** | MaxEnt IRL, Gated hybrid negative (FM), 200ep |
| **현황** | 학습 완료 (200ep), val 완료. 구버전 코드 (EBM collapse 수정 전) |

---

## 2. 무엇을 검증하나

B(supervised)가 ρ=0.3028로 clinical correlation을 달성.
MaxEnt IRL이 reward를 명시 없이 역추론했을 때 B를 넘는지:

- `Spearman ρ > 0.3028` — C > B 달성 여부 (논문 핵심 주장)
- `p < 0.05` — 통계적 유의성
- EBM dynamics 안정성

---

## 3. 학습 손실 곡선

| epoch | reward_loss | e_pos | e_neg | fm_loss |
|------:|:-----------:|:-----:|:-----:|:-------:|
| 10 | -5.0000 | -5.000 | +5.000 | 0.044 |
| 100 | -5.0000 | -5.000 | +5.000 | 0.054 |
| 200 | -5.0000 | -5.000 | +5.000 | 0.069 |

**관찰:** step 초반부터 e_pos≈−5, e_neg≈+5 고정 — EBM binary collapse 전 구간 지속. loss 자체는 수렴했으나 ranking 신호 없음.

> 상세 step 로그: `outputs/ablation_C.log`

---

## 4. 검증 지표

| epoch | Spearman ρ ↑ | p-value | AUROC | ECE ↓ | 판정 |
|------:|:------------:|:-------:|:-----:|:-----:|:----:|
| 10 | -0.0031 | 0.9675 | 0.5295 | 0.7955 | FAIL |
| 50 | 0.0759 | 0.3151 | 0.5078 | 0.7957 | FAIL |
| 100 | — | — | — | — | FAIL |
| 180 | 0.1071 | 0.1560 | 0.5128 | 0.7956 | FAIL |
| 200 | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

### B 대비 비교

| 지표 | B best (ep80) | C best (ep180) | 목표 달성? |
|------|:-------------:|:--------------:|:----------:|
| Spearman ρ | 0.3028 | 0.1071 | ❌ |
| p-value | 0.0000 | 0.1560 (FAIL) | ❌ |
| ECE | 0.2095 | 0.7956 | ❌ |

---

## 5. 결과 해석 및 인사이트

### 주의
- **전 epoch FAIL.** C < B — 논문 핵심 주장 미달성.
- **근본 원인: EBM collapse.** step 20에서 이미 binary separator로 붕괴. IRL reward signal 자체가 의미 없는 상태.
- ECE=0.7956로 calibration도 완전 망가짐 — A와 동일한 포화 패턴.

### 근본 원인 (확정)
1. `l2_reg=0.01`로 EBM이 step 20에 e_pos≈−50 / e_neg≈+50 이진 붕괴 완료
2. 이후 FM gate가 열려도 재붕괴 (Phase 2 진입 시 grad_norm=935 → 2 step 내 ±50 복귀)
3. 이 결과는 **구버전 코드 기준** — l2_reg, FM gate 수정 전

---

## 6. 다음 스텝

- [x] 구 C FAIL 원인 확정 → EBM collapse (l2_reg 과소)
- [x] ebm_weighted_cd에서 collapse 수정 후 재실험 → ρ=0.3041 달성
- [ ] 수정된 코드로 C 공식 재훈련 (ablation 비교용)

---

## 7. 저장 파일 목록

```
outputs/irl_maxent/
├── ckpt_epoch*.pt    ← epoch 10~200 (git 제외)
└── RESULT.md
outputs/ablation_C.log  ← 전체 step/val 로그 (구버전)
```
