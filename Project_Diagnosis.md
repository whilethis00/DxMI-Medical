# 프로젝트 냉정 진단 (2026-04-17)

## 한 줄 진단

**현재 가장 큰 리스크는 방법론 명분이 아니라, 측정이 맞는지 모르고 있다는 것이다.**

올바른 점검 순서:

1. **측정이 맞는가** — AUROC convention 검증
2. **학습 신호가 살아 있는가** — gradient path, energy saturation
3. **그 다음에야** 방법론 명분 문제

지금 결과는 IRL의 이론적 패배를 보여주는 게 아니다. **평가 정의와 reward/policy 학습 다이나믹이 깨진 상태일 가능성을 먼저 시사한다.**

---

## Ablation 결과 — 현재 숫자는 신뢰 보류

| 실험 | Spearman ρ | p-value | AUROC | ECE | 판정 |
|------|------------|---------|-------|-----|------|
| A (No IRL) | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B (Supervised reward) | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C (MaxEnt IRL) | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

### AUROC가 지금 이 숫자 그대로면 모순이다

B에서 Spearman ρ = +0.245 (energy ↑ = disagreement ↑)인데 AUROC = 0.33이다.

energy가 disagreement와 양의 상관이면, `score = energy`로 AUROC를 계산하면 0.67이 나와야 한다. 0.33이 나왔다는 건 **`score = -energy`로 계산했다는 증거**다.

이걸 뒤집으면:
- B: AUROC ≈ **0.67** — Spearman과 일관됨, 실제로 강한 결과
- C: AUROC ≈ **0.42** — Spearman과 일관됨, 전 항목 실패
- A: AUROC ≈ **0.53** — 거의 random

즉 **현재 AUROC 표는 읽을 수 없다.** sign이 통일됐는지 확인 전까지 AUROC 숫자로 어떤 해석도 하지 말 것.

확인해야 하는 것 세 가지:
- AUROC에 넣은 score가 `energy`인지 `-energy`인지
- positive label이 `high disagreement=1`인지 `low disagreement=1`인지
- binarization threshold 정의

`energy`와 `-energy` 둘 다로 재계산하고, Spearman 방향과 일치하는 쪽을 convention으로 고정한다.

---

## C 실패의 4가지 가능한 경로

C가 망하는 원인을 단일 원인으로 확정하면 안 된다. 아직 4개 경로가 열려 있다.

### 1. 평가 버그
AUROC sign/label 정의 오류. 위에서 서술.

### 2. Reward model saturation
`e_pos ≈ -5`, `e_neg ≈ 5`로 빠르게 포화. energy가 ranking function이 아니라 binary separator로 작동하면 policy improvement 자체가 불가능해진다. 이 상태에서 IRL을 돌리면 C는 반드시 A처럼 실패한다.

### 3. Gradient path bug (Apr 14 수정)
`_policy_sample()`, `_freeze_ebm_params()` 추가가 의도한 대로 작동하지 않을 수 있다. reward_loss가 VF에 grad를 보내는지, EBM freeze 시 leakage가 없는지 아직 검증 안 됐다.

### 4. FM policy sample이 trivially easy negative (핵심)

Apr 14에 SGLD negative → FM policy sample로 교체했다. **이게 포화가 계속 나는 새로운 원인일 수 있다.**

학습 초반 FM은 아직 수렴 안 됐다. FM rollout 결과는 Gaussian noise에 가까운 패치다. EBM 입장에서 real nodule patch vs. noise patch는 trivially easy separation이다. 즉, negative quality 자체가 너무 약해서 EBM이 calibration 대신 margin만 키우는 방향으로 최적화될 수 있다.

SGLD negative는 적어도 real sample을 perturbation한 것이었다. FM policy sample은 early training일수록 hard negative가 아니라 trivial negative가 된다.

**이 문제는 grad path를 완전히 고쳐도, l2_reg를 조정해도 포화가 안 잡힐 수 있다는 뜻이다.** 별도 경로로 봐야 한다.

---

## Saturation 진단: 어떻게 원인을 좁히는가

포화의 원인은 `l2_reg` 하나가 아닐 가능성이 높다. 따라서 변수를 한꺼번에 sweep하는 게 아니라, **로그로 원인을 먼저 좁힌 뒤 해당 변수만 건드리는 게 맞다.**

### 필수 로그 항목

- `e_pos`, `e_neg` 평균
- `e_pos_std`, `e_neg_std` — 분산이 죽어 있는지 확인
- **pos/neg energy histogram + overlap** — 분산이 살아 있어도 두 분포가 너무 쉽게 갈라지면 ranking은 죽는다. std만으로는 부족하고 histogram overlap을 같이 봐야 한다
- batch 내 rank correlation
- epoch별 val Spearman trajectory
- policy sample energy distribution (FM rollout 결과물의 energy 분포)

### 원인 좁히기 순서

1. histogram overlap이 죽어 있으면 → negative quality 문제 (FM trivial negative 가설)
2. std는 살아 있는데 ranking이 죽었으면 → regularization / temperature 문제
3. ranking도 있는데 metric이 망했으면 → evaluation bug

이 분기에 따라 건드릴 변수가 달라진다. 원인 파악 없이 전체 sweep은 실험이 아니라 도박이다.

---

## 실행 순서

**AUROC 재계산과 grad smoke test는 병렬로 진행한다.** 서로 독립적이다.

Saturation mini-run은 grad path 검증 이후여야 한다. Apr 14 수정이 graph를 바꿨기 때문에, 깨진 graph 위에서 돌린 포화 진단은 의미가 없다.

### Step 1 (병렬)

**AUROC 재계산**
- `score = energy` vs `score = -energy` 둘 다 계산
- positive label 정의 명시
- Spearman 방향과 일치하는 convention으로 고정
- 결과: 기존 표 재해석 또는 폐기

**Grad path smoke test** (CPU 2-step)
- reward update 시: EBM에 grad 있음, VF에 grad 없어야 하는가/있어야 하는가 (설계 의도 먼저 명확히)
- policy loss 시: VF에 grad 흐름, EBM에 grad 없어야 함
- `_policy_sample(detach=True/False)`가 기대한 graph connectivity를 실제로 보장하는가
- "grad가 흐른다"가 아니라 **누구에게, 어느 loss에서, 어느 경로로**가 기준

### Step 2

**Saturation mini-run** (Step 1 완료 후)
- 위 로그 항목 전부 기록
- histogram overlap으로 원인 좁히기
- 합격 기준: `e_pos/e_neg`가 극단값으로 즉시 안 감, val Spearman이 epoch 초반 이후에도 의미 있게 움직임, energy histogram이 즉시 두 덩어리로 찢어지지 않음

### Step 3

**Full retrain** (Step 1, 2 완료 후)

---

## 하지 말 것

- `l2_reg` 하나 줄이고 full retrain → 원인 분해 없이 운에 맡기는 것
- C 실패를 "IRL이 이론적으로 불필요하다"로 읽는 것 → 아직 evaluation bug와 grad bug, trivial negative 문제가 열려 있다
- 플랜 B를 지금 설계하는 것 → 정상 동작하는 실험 한 번도 안 해보고 항복 플랜 짜는 것

방법론 명분 문제(IRL의 필요성)는 없어지는 게 아니다. 하지만 지금은 순서가 아니다. **계측기 교정 단계가 먼저다.** Step 1~3 완료 후에도 C < B면 그때 정면으로 본다.

---

## 방법론 명분 — 나중에 볼 문제

지금은 미뤄도 되지만 완전히 잊으면 안 된다.

리뷰어가 찌를 포인트:
> "임상 disagreement proxy가 이미 있는데 왜 굳이 IRL을 쓰죠?"

이 질문에 답하려면 C > B로 끝나거나, 아니면 IRL이 빛나는 조건을 설계해야 한다.

예시:
- annotator labels 일부만 사용했을 때 (label efficiency)
- noisy reward proxy에서 robustness 비교
- pairwise preference만 줬을 때

**full supervision setting에서 B가 강한 건 당연하다.** IRL의 존재 이유가 full supervision이 주어졌을 때 B를 이기는 것이라면, 그 명분은 처음부터 약하다. IRL이 빛나는 판을 설계하지 않으면 B를 이기는 구현이 완성돼도 의미가 없다.

---

## 프로젝트 강점 — 변하지 않은 것

- 문제 정의: "모델 uncertainty"가 아니라 "의사 disagreement" 추정
- 데이터: LIDC-IDRI 4인 독립 판독, inter-reader variance 직접 사용 가능
- 논문 구조: A/B/C ablation + clinical correlation 이중 증명

**스토리 골격은 살아 있다.**

---

## 최종 판정

| 항목 | 평가 |
|------|------|
| 아이디어 | 강함 |
| 데이터 선택 | 적절 |
| 문제 정의 | 매력적 |
| 실험적 증거 | 아직 신뢰 불가 (측정 검증 전) |
| 핵심 claim | 미입증 — 단, 실험 자체가 아직 정상 상태가 아님 |
| 가장 큰 즉각적 위험 | AUROC 해석 오류 + energy saturation |
| 그 다음 위험 | FM trivial negative로 인한 saturation 재발 |
| 방법론 명분 위험 | 실재하나 지금 순서 아님 |

**지금 이 프로젝트는 죽지 않았다. 하지만 지금 숫자로는 아무것도 결론 내릴 수 없다.**

Step 1~2를 완료하고 나서야 "IRL이 작동하는가"를 처음으로 제대로 물을 수 있다.

---

*작성: 2026-04-17*
