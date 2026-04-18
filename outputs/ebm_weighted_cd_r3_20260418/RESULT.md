# ebm_weighted_cd_r3 실험 결과

날짜: 2026-04-19
config: `configs/ebm_weighted_cd_r3.yaml`

---

## 개요

Reward-weighted CD — reproduction run 3 (seed=3)

## 가설

*(작성 필요)*

## 실험 세팅

| 항목 | 값 |
|------|----|
| epochs | 20 |
| l2_reg | 0.1 |
| reward_cd_weight | 1.0 |
| reward_cd_temp | 1.0 |
| fm_gate_sep_std_threshold | 9999.0 |

## 결과

| epoch | 로그 |
|-------|------|
| epoch 1 | `[epoch 1] val: N=177 | Spearman ρ=0.1987 (p=0.0080, PASS) | AUROC(-E)=0.3870 | AUROC(E)=0.6130 | ECE=0.7978` |
| epoch 2 | `[epoch 2] val: N=177 | Spearman ρ=0.0839 (p=0.2667, FAIL) | AUROC(-E)=0.5193 | AUROC(E)=0.4807 | ECE=0.7960` |
| epoch 3 | `[epoch 3] val: N=177 | Spearman ρ=0.1552 (p=0.0392, PASS) | AUROC(-E)=0.3745 | AUROC(E)=0.6255 | ECE=0.7869` |
| epoch 4 | `[epoch 4] val: N=177 | Spearman ρ=0.1492 (p=0.0475, PASS) | AUROC(-E)=0.3772 | AUROC(E)=0.6228 | ECE=0.7952` |
| epoch 5 | `[epoch 5] val: N=177 | Spearman ρ=0.1790 (p=0.0171, PASS) | AUROC(-E)=0.3836 | AUROC(E)=0.6164 | ECE=0.7917` |
| epoch 6 | `[epoch 6] val: N=177 | Spearman ρ=0.1548 (p=0.0397, PASS) | AUROC(-E)=0.3865 | AUROC(E)=0.6135 | ECE=0.7939` |
| epoch 7 | `[epoch 7] val: N=177 | Spearman ρ=0.2437 (p=0.0011, PASS) | AUROC(-E)=0.3818 | AUROC(E)=0.6182 | ECE=0.7967` |
| epoch 8 | `[epoch 8] val: N=177 | Spearman ρ=0.1026 (p=0.1743, FAIL) | AUROC(-E)=0.3652 | AUROC(E)=0.6348 | ECE=0.7894` |
| epoch 9 | `[epoch 9] val: N=177 | Spearman ρ=0.1587 (p=0.0349, PASS) | AUROC(-E)=0.3262 | AUROC(E)=0.6738 | ECE=0.7962` |
| epoch 10 | `[epoch 10] val: N=177 | Spearman ρ=0.1193 (p=0.1139, FAIL) | AUROC(-E)=0.4462 | AUROC(E)=0.5538 | ECE=0.7924` |
| epoch 11 | `[epoch 11] val: N=177 | Spearman ρ=0.2679 (p=0.0003, PASS) | AUROC(-E)=0.3400 | AUROC(E)=0.6600 | ECE=0.7902` |
| epoch 12 | `[epoch 12] val: N=177 | Spearman ρ=0.2150 (p=0.0041, PASS) | AUROC(-E)=0.3966 | AUROC(E)=0.6034 | ECE=0.7944` |
| epoch 13 | `[epoch 13] val: N=177 | Spearman ρ=0.2500 (p=0.0008, PASS) | AUROC(-E)=0.3149 | AUROC(E)=0.6851 | ECE=0.7952` |
| epoch 14 | `[epoch 14] val: N=177 | Spearman ρ=0.2510 (p=0.0008, PASS) | AUROC(-E)=0.3592 | AUROC(E)=0.6408 | ECE=0.7944` |
| epoch 15 | `[epoch 15] val: N=177 | Spearman ρ=0.2548 (p=0.0006, PASS) | AUROC(-E)=0.2966 | AUROC(E)=0.7034 | ECE=0.7949` |
| epoch 16 | `[epoch 16] val: N=177 | Spearman ρ=0.2291 (p=0.0022, PASS) | AUROC(-E)=0.3311 | AUROC(E)=0.6689 | ECE=0.7953` |
| epoch 17 | `[epoch 17] val: N=177 | Spearman ρ=0.2844 (p=0.0001, PASS) | AUROC(-E)=0.3064 | AUROC(E)=0.6936 | ECE=0.7944` |
| epoch 18 | `[epoch 18] val: N=177 | Spearman ρ=0.2815 (p=0.0001, PASS) | AUROC(-E)=0.2852 | AUROC(E)=0.7148 | ECE=0.7951` |
| epoch 19 | `[epoch 19] val: N=177 | Spearman ρ=0.1568 (p=0.0372, PASS) | AUROC(-E)=0.3022 | AUROC(E)=0.6978 | ECE=0.7871` |
| epoch 20 | `[epoch 20] val: N=177 | Spearman ρ=0.2946 (p=0.0001, PASS) | AUROC(-E)=0.2926 | AUROC(E)=0.7074 | ECE=0.7934` |

**최종 val**: `[epoch 20] val: N=177 | Spearman ρ=0.2946 (p=0.0001, PASS) | AUROC(-E)=0.2926 | AUROC(E)=0.7074 | ECE=0.7934`

## 가설 달성 여부

*(작성 필요)*

## 인사이트

*(작성 필요)*

## 다음 계획

*(작성 필요)*
