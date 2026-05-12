# supervised_reward 실험 결과

날짜: 2026-05-12
config: `configs/supervised_reward.yaml`

---

## 개요

EBM supervised on GT malignancy variance — Ablation B

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

### Validation

Best validation checkpoint is epoch 80.

| checkpoint | Spearman rho | p-value | AUROC(E) | ECE | status |
|---|---:|---:|---:|---:|---|
| ep80 | 0.3028 | <0.001 | 0.6719 | 0.2095 | PASS |

Final epoch validation:

| checkpoint | Spearman rho | p-value | AUROC(E) | ECE | status |
|---|---:|---:|---:|---:|---|
| ep200 | 0.2455 | 0.0010 | 0.6739 | 0.1989 | PASS |

### Test

Evaluated with `ckpt_epoch0080.pt`, selected by best validation rho.

| checkpoint | Spearman rho | p-value | AUROC(E) | ECE | status |
|---|---:|---:|---:|---:|---|
| ep80 | 0.2083 | 0.0053 | 0.6461 | 0.2318 | PASS |

Raw test output:

```text
[TEST] supervised_reward | ckpt=ckpt_epoch0080.pt | N=178 | Spearman rho=0.2083 (p=0.0053, PASS) | AUROC(-E)=0.3539 | AUROC(E)=0.6461 | ECE=0.2318
```

## 가설 달성 여부

부분 달성.

Direct supervised variance regression은 expert disagreement를 유의하게 예측한다. 따라서 이후 IRL 계열은 이 baseline보다 강하거나, sparse/noisy/preference supervision에서 더 견고하다는 식으로 정당화해야 한다.

## 인사이트

- Strong direct-supervision baseline은 약하지 않다: test rho 0.2083, p=0.0053.
- 1번 실험의 classifier uncertainty baseline과 달리, `malignancy_var` 직접 지도는 disagreement signal을 실제로 학습한다.
- 이 baseline은 리뷰어의 "그냥 variance regression 하면 되지 않나?" 반론에 대한 기준선으로 사용해야 한다.
- 기존 FM gate v1 test rho 0.2885는 이 supervised baseline보다 높다. 다만 v3 seed1 test rho 0.2585는 supervised baseline보다 높지만 v1보다는 낮다.

## 다음 계획

- IRL/FM 계열과 supervised baseline을 같은 test split에서 표로 정리한다.
- 3번 실험: disagreement label fraction 100/50/25/10%에서 supervised regression과 IRL을 비교한다.
