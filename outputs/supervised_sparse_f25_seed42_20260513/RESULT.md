# supervised_sparse_f25_seed42 실험 결과

날짜: 2026-05-13
config: `configs/generated/sparse_labels/supervised_f25_seed42.yaml`

---

## 개요

EBM supervised on GT malignancy variance — Ablation B | sparse labels: 25% seed=42

## 가설

*(작성 필요)*

## 실험 세팅

| 항목 | 값 |
|------|----|
| epochs | 200 |
| l2_reg | 0.1 |
| reward_cd_weight | N/A |
| reward_cd_temp | N/A |
| fm_gate_sep_std_threshold | ? |

## 결과

| epoch | rho | p | AUROC(E) | ECE | N | avg_loss | 판정 |
|-------|-----|---|----------|-----|---|----------|------|
| 50 | 0.3535 | 0.0000 | 0.7103 | 0.2225 | 177 | 0.1017 | best |
| 200 | 0.2656 | 0.0004 | 0.6609 | 0.2121 | 177 | 0.0344 | PASS |

**최종 val**: `rho=0.2656 (p=0.0004), AUROC(E)=0.6609, ECE=0.2121, avg_loss=0.0344`

**Best val**: `rho=0.3535 @ ep50`

## 가설 달성 여부

PASS 기준은 만족했지만, 최고 성능은 ep50에서 나온 뒤 후반으로 갈수록 rho가 낮아졌다.

## 인사이트

25% sparse supervised 설정에서는 loss가 계속 낮아져도 validation rho는 함께 개선되지 않았다. 최종 ep200은 ep50 대비 rho가 0.0879 낮고 AUROC(E)도 0.0494 낮아, early stopping 기준으로는 ep50 checkpoint가 더 적합하다.

## 다음 계획

ep50 checkpoint 기준으로 test 평가를 수행하고, 같은 seed의 50%/100% sparse supervised 결과와 rho, AUROC(E), ECE를 비교한다.
