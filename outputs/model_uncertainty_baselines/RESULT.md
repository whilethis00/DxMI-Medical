# model_uncertainty_baselines — Result

## 1. 실험 목적

필수 실험 블록 1: **모델 불확실성 vs. 임상적 모호성**.

악성도 분류 모델의 표준 uncertainty score가 전문가 불일치(`malignancy_var`)가 큰 사례를 안정적으로 복원하는지 확인한다.

핵심 질문:

> 악성도 분류 모델이 잘 맞는다고 해서, 그 모델의 uncertainty가 expert disagreement를 설명하는가?

## 2. 설정

| 항목 | 내용 |
|---|---|
| config | `configs/model_uncertainty_baselines.yaml` |
| test set size | 178 |
| classifier seeds | 1, 2, 3 |
| high-disagreement 기준 | test set `malignancy_var` top quartile |
| high-disagreement cutoff | `malignancy_var >= 0.75` |
| high-disagreement base rate | 0.2753 |
| malignancy label | `malignancy_mean >= 3.0` |

실행 명령:

```bash
/home/introai26/miniconda3/envs/dxmi_medical/bin/python scripts/uncertainty_baselines.py \
  --config configs/model_uncertainty_baselines.yaml
```

## 3. 결과

| score | rho | p-value | AUROC(high disagreement) | malignancy AUROC | top5 enrichment | top10 enrichment | top20 enrichment |
|---|---:|---:|---:|---:|---:|---:|---:|
| predictive entropy | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| margin uncertainty | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| MC entropy | -0.0726 | 0.3358 | 0.4392 | 0.7694 | 1.6145 | 1.6145 | 1.0091 |
| MC variance | -0.0230 | 0.7604 | 0.4455 | 0.7694 | 0.4036 | 0.6054 | 0.8073 |
| ensemble entropy | -0.0293 | 0.6977 | 0.4637 | 0.7694 | 1.6145 | 1.0091 | 0.9082 |
| ensemble variance | 0.0491 | 0.5151 | 0.5037 | 0.7694 | 0.8073 | 0.8073 | 1.2109 |

## 4. 해석

분류 모델은 악성도 판별 자체는 의미 있게 학습했다.

- `malignancy AUROC = 0.7694`

하지만 표준 uncertainty score들은 전문가 불일치를 복원하지 못했다.

- 모든 Spearman correlation이 약하고 통계적으로 유의하지 않다.
- predictive entropy와 margin uncertainty는 오히려 음의 상관을 보인다.
- MC-dropout variance는 거의 0에 가까운 상관이다: `rho = -0.0230`, `p = 0.7604`.
- ensemble variance가 sign 기준으로는 가장 낫지만, high-disagreement AUROC는 `0.5037`로 random 수준이다.

결론:

> 악성도 분류는 가능하지만, 분류 모델의 표준 uncertainty는 expert disagreement로 드러나는 clinical ambiguity를 설명하지 못한다.

## 5. 논문용 요약

A malignancy classifier achieved reasonable diagnostic discrimination on the test set (`AUROC = 0.7694`), but its standard uncertainty estimates failed to recover expert disagreement. Predictive entropy, margin uncertainty, MC-dropout variance, and ensemble variance all showed weak, non-significant association with `malignancy_var` (`|rho| <= 0.0983`, all `p > 0.19`). The strongest high-disagreement detector, ensemble variance, was effectively random (`AUROC = 0.5037`). These results suggest that model uncertainty and clinical ambiguity are empirically distinct in this setting.

## 6. 산출물

- `outputs/model_uncertainty_baselines/results_test.csv`
- `outputs/model_uncertainty_baselines/results_test.md`
- `outputs/model_uncertainty_baselines/predictions_test.csv`
- `outputs/model_uncertainty_baselines/seed*/ckpt_best.pt` (git ignored)
