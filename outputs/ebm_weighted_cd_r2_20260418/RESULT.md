# ebm_weighted_cd_r2 실험 결과

날짜: 2026-04-18
config: `configs/ebm_weighted_cd_r2.yaml`

---

## 개요

Reward-weighted CD — reproduction run 2 (seed=2)

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
| epoch 1 | `[epoch 1] val: N=177 | Spearman ρ=0.0088 (p=0.9070, FAIL) | AUROC(-E)=0.4070 | AUROC(E)=0.5930 | ECE=0.7957` |
| epoch 2 | `[epoch 2] val: N=177 | Spearman ρ=0.1232 (p=0.1024, FAIL) | AUROC(-E)=0.4913 | AUROC(E)=0.5087 | ECE=0.7925` |
| epoch 3 | `[epoch 3] val: N=177 | Spearman ρ=0.2529 (p=0.0007, PASS) | AUROC(-E)=0.3977 | AUROC(E)=0.6023 | ECE=0.7951` |
| epoch 4 | `[epoch 4] val: N=177 | Spearman ρ=0.1969 (p=0.0086, PASS) | AUROC(-E)=0.3970 | AUROC(E)=0.6030 | ECE=0.7867` |
| epoch 5 | `[epoch 5] val: N=177 | Spearman ρ=0.2176 (p=0.0036, PASS) | AUROC(-E)=0.4078 | AUROC(E)=0.5922 | ECE=0.7966` |
| epoch 6 | `[epoch 6] val: N=177 | Spearman ρ=0.1362 (p=0.0707, FAIL) | AUROC(-E)=0.4207 | AUROC(E)=0.5793 | ECE=0.7938` |
| epoch 7 | `[epoch 7] val: N=177 | Spearman ρ=0.0929 (p=0.2185, FAIL) | AUROC(-E)=0.3584 | AUROC(E)=0.6416 | ECE=0.7936` |
| epoch 8 | `[epoch 8] val: N=177 | Spearman ρ=0.2818 (p=0.0001, PASS) | AUROC(-E)=0.3935 | AUROC(E)=0.6065 | ECE=0.7953` |
| epoch 9 | `[epoch 9] val: N=177 | Spearman ρ=0.2743 (p=0.0002, PASS) | AUROC(-E)=0.3126 | AUROC(E)=0.6874 | ECE=0.7947` |
| epoch 10 | `[epoch 10] val: N=177 | Spearman ρ=0.2329 (p=0.0018, PASS) | AUROC(-E)=0.2782 | AUROC(E)=0.7218 | ECE=0.7956` |
| epoch 11 | `[epoch 11] val: N=177 | Spearman ρ=0.2674 (p=0.0003, PASS) | AUROC(-E)=0.2764 | AUROC(E)=0.7236 | ECE=0.7947` |
| epoch 12 | `[epoch 12] val: N=177 | Spearman ρ=0.2728 (p=0.0002, PASS) | AUROC(-E)=0.3324 | AUROC(E)=0.6676 | ECE=0.7951` |
| epoch 13 | `[epoch 13] val: N=177 | Spearman ρ=0.2135 (p=0.0043, PASS) | AUROC(-E)=0.2489 | AUROC(E)=0.7511 | ECE=0.7939` |
| epoch 14 | `[epoch 14] val: N=177 | Spearman ρ=0.2105 (p=0.0049, PASS) | AUROC(-E)=0.3477 | AUROC(E)=0.6523 | ECE=0.7897` |
| epoch 15 | `[epoch 15] val: N=177 | Spearman ρ=0.2888 (p=0.0001, PASS) | AUROC(-E)=0.2723 | AUROC(E)=0.7277 | ECE=0.7934` |
| epoch 16 | `[epoch 16] val: N=177 | Spearman ρ=0.2189 (p=0.0034, PASS) | AUROC(-E)=0.2947 | AUROC(E)=0.7053 | ECE=0.7948` |
| epoch 17 | `[epoch 17] val: N=177 | Spearman ρ=0.2902 (p=0.0001, PASS) | AUROC(-E)=0.3129 | AUROC(E)=0.6871 | ECE=0.7937` |
| epoch 18 | `[epoch 18] val: N=177 | Spearman ρ=0.2368 (p=0.0015, PASS) | AUROC(-E)=0.2900 | AUROC(E)=0.7100 | ECE=0.7945` |
| epoch 19 | `[epoch 19] val: N=177 | Spearman ρ=0.2828 (p=0.0001, PASS) | AUROC(-E)=0.3104 | AUROC(E)=0.6896 | ECE=0.7947` |
| epoch 20 | `[epoch 20] val: N=177 | Spearman ρ=0.2650 (p=0.0004, PASS) | AUROC(-E)=0.3233 | AUROC(E)=0.6767 | ECE=0.7936` |

**최종 val**: `[epoch 20] val: N=177 | Spearman ρ=0.2650 (p=0.0004, PASS) | AUROC(-E)=0.3233 | AUROC(E)=0.6767 | ECE=0.7936`

## 가설 달성 여부

*(작성 필요)*

## 인사이트

*(작성 필요)*

## 다음 계획

*(작성 필요)*
