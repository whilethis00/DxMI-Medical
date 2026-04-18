# ebm_weighted_cd 실험 결과

날짜: 2026-04-18
config: `configs/ebm_weighted_cd.yaml`

---

## 최종 val 지표 (epoch 20)

| 지표 | 값 | 판정 |
|------|-----|------|
| N | 177 | — |
| Spearman ρ | **0.3041** | **PASS** (p=0.0000) |
| AUROC(-E) | 0.2731 | — |
| AUROC(E) | **0.7269** | — |
| ECE | 0.7945 | — |

---

## 훈련 dynamics (step 3760~3800, 마지막 로그)

| step | e_pos | e_neg | e_pos_std | sep_std_ema | fm_enabled |
|------|-------|-------|-----------|-------------|------------|
| 3760 | -5.14 | +5.02 | 0.329 | 104.7 | 0.0 (Phase 1) |
| 3770 | -5.01 | +4.99 | 0.167 | 96.7  | 0.0 |
| 3780 | -5.02 | +4.94 | 0.294 | 79.5  | 0.0 |
| 3790 | -5.02 | +5.03 | 0.263 | 85.8  | 0.0 |
| 3800 | -4.90 | +5.01 | 0.330 | 98.8  | 0.0 |

---

## 이전 실험 대비 비교

| 실험 | Spearman ρ | p-value | AUROC(E) | EBM 상태 |
|------|------------|---------|----------|----------|
| irl_minirun (epoch 20) | 0.0480 | 0.5256 | 0.4915 | ±50 붕괴 |
| ablation B best (epoch 80) | 0.3028 | 0.0000 | 0.6719 | 정상 |
| **ebm_weighted_cd (epoch 20)** | **0.3041** | **0.0000** | **0.7269** | **정상** |

---

## 핵심 관찰

- EBM collapse 해결: e_pos≈−5, e_neg≈+5 (이전 ±50 붕괴 없음)
- e_pos_std≈0.3 유지 → 클래스 내 분산 살아있음 → ranking 학습 가능
- epoch 20 시점에서 ablation B best(epoch 80)를 이미 초과
- Phase 1(SGLD only, fm_enabled=0) 상태 — FM gate 아직 미진입
- ECE=0.7945 는 낮음 — FM Phase 2 진입 후 개선 예상

---

## 다음 단계

1. FM gate 조건 확인 (sep_std_ema가 threshold 이하로 내려오는 시점)
2. Phase 2(FM hybrid) 진입 후 behavior 관찰
3. 동일 조건으로 ablation A/C 재훈련하여 공식 비교
