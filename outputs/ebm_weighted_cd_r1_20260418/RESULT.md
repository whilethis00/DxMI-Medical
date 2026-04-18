# ebm_weighted_cd_r1 실험 결과

날짜: 2026-04-18
config: `configs/ebm_weighted_cd_r1.yaml`

---

## 개요

Reward-weighted CD — reproduction run 1 (seed=1)

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
| epoch 1 | `[epoch 1] val: N=177 | Spearman ρ=-0.1316 (p=0.0808, FAIL) | AUROC(-E)=0.4628 | AUROC(E)=0.5372 | ECE=0.7885` |
| epoch 2 | `[epoch 2] val: N=177 | Spearman ρ=0.0029 (p=0.9691, FAIL) | AUROC(-E)=0.4879 | AUROC(E)=0.5121 | ECE=0.7933` |
| epoch 3 | `[epoch 3] val: N=177 | Spearman ρ=0.0296 (p=0.6956, FAIL) | AUROC(-E)=0.4307 | AUROC(E)=0.5693 | ECE=0.7914` |
| epoch 4 | `[epoch 4] val: N=177 | Spearman ρ=0.1603 (p=0.0330, PASS) | AUROC(-E)=0.4411 | AUROC(E)=0.5589 | ECE=0.7929` |
| epoch 5 | `[epoch 5] val: N=177 | Spearman ρ=0.2342 (p=0.0017, PASS) | AUROC(-E)=0.4053 | AUROC(E)=0.5947 | ECE=0.7924` |
| epoch 6 | `[epoch 6] val: N=177 | Spearman ρ=0.1759 (p=0.0192, PASS) | AUROC(-E)=0.3292 | AUROC(E)=0.6708 | ECE=0.7934` |
| epoch 7 | `[epoch 7] val: N=177 | Spearman ρ=0.1963 (p=0.0088, PASS) | AUROC(-E)=0.4076 | AUROC(E)=0.5924 | ECE=0.7962` |
| epoch 8 | `[epoch 8] val: N=177 | Spearman ρ=0.1400 (p=0.0631, FAIL) | AUROC(-E)=0.4196 | AUROC(E)=0.5804 | ECE=0.7963` |
| epoch 9 | `[epoch 9] val: N=177 | Spearman ρ=0.0349 (p=0.6449, FAIL) | AUROC(-E)=0.4464 | AUROC(E)=0.5536 | ECE=0.7932` |
| epoch 10 | `[epoch 10] val: N=177 | Spearman ρ=0.1934 (p=0.0099, PASS) | AUROC(-E)=0.3537 | AUROC(E)=0.6463 | ECE=0.7966` |
| epoch 11 | `[epoch 11] val: N=177 | Spearman ρ=0.2181 (p=0.0035, PASS) | AUROC(-E)=0.3810 | AUROC(E)=0.6190 | ECE=0.7945` |
| epoch 12 | `[epoch 12] val: N=177 | Spearman ρ=0.2198 (p=0.0033, PASS) | AUROC(-E)=0.4069 | AUROC(E)=0.5931 | ECE=0.7929` |
| epoch 13 | `[epoch 13] val: N=177 | Spearman ρ=0.1254 (p=0.0962, FAIL) | AUROC(-E)=0.3066 | AUROC(E)=0.6934 | ECE=0.7930` |
| epoch 14 | `[epoch 14] val: N=177 | Spearman ρ=0.2364 (p=0.0015, PASS) | AUROC(-E)=0.4215 | AUROC(E)=0.5785 | ECE=0.7937` |
| epoch 15 | `[epoch 15] val: N=177 | Spearman ρ=0.1450 (p=0.0541, FAIL) | AUROC(-E)=0.3736 | AUROC(E)=0.6264 | ECE=0.7982` |
| epoch 16 | `[epoch 16] val: N=177 | Spearman ρ=0.1918 (p=0.0105, PASS) | AUROC(-E)=0.3184 | AUROC(E)=0.6816 | ECE=0.7930` |
| epoch 17 | `[epoch 17] val: N=177 | Spearman ρ=0.2298 (p=0.0021, PASS) | AUROC(-E)=0.3911 | AUROC(E)=0.6089 | ECE=0.7939` |
| epoch 18 | `[epoch 18] val: N=177 | Spearman ρ=0.2100 (p=0.0050, PASS) | AUROC(-E)=0.3513 | AUROC(E)=0.6487 | ECE=0.7941` |
| epoch 19 | `[epoch 19] val: N=177 | Spearman ρ=0.2285 (p=0.0022, PASS) | AUROC(-E)=0.3295 | AUROC(E)=0.6705 | ECE=0.7927` |
| epoch 20 | `[epoch 20] val: N=177 | Spearman ρ=0.2233 (p=0.0028, PASS) | AUROC(-E)=0.3347 | AUROC(E)=0.6653 | ECE=0.7933` |

**최종 val**: `[epoch 20] val: N=177 | Spearman ρ=0.2233 (p=0.0028, PASS) | AUROC(-E)=0.3347 | AUROC(E)=0.6653 | ECE=0.7933`

## 가설 달성 여부

*(작성 필요)*

## 인사이트

*(작성 필요)*

## 다음 계획

*(작성 필요)*
