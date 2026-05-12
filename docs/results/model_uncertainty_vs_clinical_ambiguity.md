# Model Uncertainty vs. Clinical Ambiguity

This result file tracks experiment block 1.

## Claim Under Test

Standard malignancy-classification uncertainty should not reliably recover cases with high expert disagreement (`malignancy_var`). If this holds, it supports the distinction between model uncertainty and clinical ambiguity.

## Command

```bash
python scripts/uncertainty_baselines.py --config configs/model_uncertainty_baselines.yaml
```

CPU smoke test:

```bash
python scripts/uncertainty_baselines.py --config configs/model_uncertainty_baselines.yaml \
    --device cpu --epochs 1 --max-train-batches 1 --max-eval-batches 1 --base-ch 4 --mc-samples 2
```

## Metrics

- `rho`: Spearman correlation between uncertainty score and `malignancy_var`.
- `auroc_high_disagreement`: AUROC for detecting top-quantile high-disagreement cases.
- `top*_enrichment`: fraction of high-disagreement cases in the highest uncertainty subset divided by the base high-disagreement rate.
- `malignancy_auroc`: sanity check that the classifier learned malignancy classification.

## Results

Completed on the test split with 3 classifier seeds.

| score | rho | p-value | AUROC(high disagreement) | malignancy AUROC | top5 enrichment | top10 enrichment | top20 enrichment |
|---|---:|---:|---:|---:|---:|---:|---:|
| predictive entropy | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| margin uncertainty | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| MC entropy | -0.0726 | 0.3358 | 0.4392 | 0.7694 | 1.6145 | 1.6145 | 1.0091 |
| MC variance | -0.0230 | 0.7604 | 0.4455 | 0.7694 | 0.4036 | 0.6054 | 0.8073 |
| ensemble entropy | -0.0293 | 0.6977 | 0.4637 | 0.7694 | 1.6145 | 1.0091 | 0.9082 |
| ensemble variance | 0.0491 | 0.5151 | 0.5037 | 0.7694 | 0.8073 | 0.8073 | 1.2109 |

High-disagreement definition: top quartile by `malignancy_var`.

Test set:

- `n = 178`
- high-disagreement cutoff: `malignancy_var >= 0.75`
- high-disagreement base rate: `0.2753`

## Interpretation

The classifier learned a meaningful malignancy decision boundary: `malignancy_auroc = 0.7694`.

However, standard classifier uncertainty scores did not recover expert disagreement:

- all Spearman correlations were weak and non-significant;
- entropy and margin uncertainty were negatively correlated with disagreement;
- MC-dropout variance had near-zero correlation (`rho = -0.0230`, `p = 0.7604`);
- ensemble variance was the best by sign, but still near random (`rho = 0.0491`, `p = 0.5151`, AUROC = `0.5037`).

This supports the core distinction:

> A model can learn malignancy classification while its uncertainty does not capture clinical ambiguity revealed by expert disagreement.

## Paper-Ready Summary

A malignancy classifier achieved reasonable diagnostic discrimination on the test set (`AUROC = 0.7694`), but its standard uncertainty estimates failed to recover expert disagreement. Predictive entropy, margin uncertainty, MC-dropout variance, and ensemble variance all showed weak, non-significant association with `malignancy_var` (`|rho| <= 0.0983`, all `p > 0.19`). The strongest high-disagreement detector, ensemble variance, was effectively random (`AUROC = 0.5037`). These results suggest that model uncertainty and clinical ambiguity are empirically distinct in this setting.

## Output Files

- `outputs/model_uncertainty_baselines/results_test.csv`
- `outputs/model_uncertainty_baselines/results_test.md`
- `outputs/model_uncertainty_baselines/predictions_test.csv`
- `outputs/model_uncertainty_baselines/seed*/ckpt_best.pt`
