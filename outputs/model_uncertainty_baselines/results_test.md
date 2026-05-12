# Uncertainty Baseline Results (test)

| score | rho | p_value | auroc_high_disagreement | malignancy_auroc | top5_enrichment | top10_enrichment | top20_enrichment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| predictive_entropy | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| margin_uncertainty | -0.0983 | 0.1918 | 0.4309 | 0.7694 | 1.6145 | 1.2109 | 1.0091 |
| mc_entropy | -0.0726 | 0.3358 | 0.4392 | 0.7694 | 1.6145 | 1.6145 | 1.0091 |
| mc_variance | -0.0230 | 0.7604 | 0.4455 | 0.7694 | 0.4036 | 0.6054 | 0.8073 |
| ensemble_entropy | -0.0293 | 0.6977 | 0.4637 | 0.7694 | 1.6145 | 1.0091 | 0.9082 |
| ensemble_variance | 0.0491 | 0.5151 | 0.5037 | 0.7694 | 0.8073 | 0.8073 | 1.2109 |

Interpretation guide:

- Low rho / AUROC near 0.5 means standard model uncertainty does not recover expert disagreement.
- `malignancy_auroc` is a sanity check for classification learning, not the clinical ambiguity result.
