# 프로젝트 냉정 진단 (2026-04-17)

## 한 줄 진단

**현재 가장 큰 리스크는 구현 버그가 아니라, 방법론 명분이 무너질 수 있다는 점이다.**

현재 결과가 말하는 것:
- **A(No IRL)는 죽었다**
- **B(Supervised reward)는 이미 의미 있는 signal을 잡는다**
- **C(MaxEnt IRL)는 아직 B를 못 이긴다**

이 상태에서 리뷰어는 이렇게 말한다:
> "임상 disagreement proxy가 이미 있는데 왜 굳이 IRL을 쓰죠?"

이 질문에 못 버티면 reject다.

---

## 현재 프로젝트 실제 상태

### 강점

- 문제 정의가 선명하다: "모델 uncertainty"가 아니라 "의사 disagreement"를 추정하겠다는 것
- 데이터 선택이 적절하다: LIDC-IDRI 4인 독립 판독 → inter-reader variance 직접 사용 가능
- 논문 구조가 깔끔하다: A/B/C ablation + clinical correlation 이중 증명

**스토리 골격은 살아 있다.**

### 불편한 진실

실험 결과는 아직 가설을 지지하지 않는다.

B가 통과하고 C가 실패했다는 건 — "IRL이 임상 uncertainty를 더 잘 학습한다"는 주장이 현재 **실험적으로 반박된 상태**라는 뜻이다.

지금 데이터만 보면, 핵심인 IRL이 불필요할 가능성이 더 높다.

---

## Ablation 결과 재해석

| 실험 | Spearman ρ | p-value | AUROC | ECE | 판정 |
|------|------------|---------|-------|-----|------|
| A (No IRL) | -0.0814 | 0.2814 | 0.4758 | 0.7956 | FAIL |
| B (Supervised reward) | 0.2455 | 0.0010 | 0.3261 | 0.1989 | PASS |
| C (MaxEnt IRL) | 0.0778 | 0.3034 | 0.5812 | 0.7955 | FAIL |

### A

완전 실패. 하지만 출발점으로 쓸 수 있다 — baseline이 임상 disagreement를 못 잡는다는 것을 보여줌.

### B

ρ=0.2455, p=0.0010. 엄청 중요한 결과다.
"annotator variance를 uncertainty signal로 쓰는 것 자체는 의미 있다"는 건 입증됐다.

### C

AUROC만 A/B보다 높고, 핵심 clinical metric은 실패.
가장 위험하다. 논문의 중심 명제가 **clinical disagreement alignment**이기 때문에,
여기서 실패하면 IRL 논문이 아니라 그냥 불안정한 학습 실험이 된다.

---

## 포화 패턴 해석: `e_pos ≈ -5`, `e_neg ≈ 5`

거의 확실히 경고등이다. 가능성은 셋 중 하나다.

### 1. Contrastive signal이 너무 쉬워서 energy가 binary saturation으로 무너짐

demo와 negative가 너무 쉽게 분리되면, EBM이 calibration 대신 margin만 키운다. 그러면 rank correlation이 망가질 수 있다.

### 2. Negative sampling이 임상적으로 의미 없는 방향으로 가고 있음

SGLD negative든 FM policy sample이든, negative가 "hard but clinically relevant"하지 않으면
energy는 disagreement를 배우는 게 아니라 단순 density boundary를 배운다.

### 3. Regularization/temperature/scale 문제

극단 포화면 temperature, gradient clipping, energy centering, L2 regularization, step size 중 하나 이상이 잘못 맞물렸을 가능성이 높다.

**지금 문제는 단순히 "성능이 낮다"가 아니라, energy가 ranking function으로 작동하지 않고 있다는 것이다.**

---

## 가장 위험한 가정 5개

### 1. "Radiologist variance = clinical uncertainty ground truth proxy"가 충분히 강하다는 가정

완전한 ground truth가 아니다. 리뷰어는 바로 물을 것이다:
- 판독자 variance는 uncertainty인가, 아니면 단순 noise인가?
- reader skill variation, annotation habit, class imbalance 영향은?
- malignancy score 1~5 variance가 truly calibrated clinical uncertainty인가?

→ 논문에서는 반드시 "proxy"라고 낮춰 말해야 하고, 보완 분석이 필요하다.

### 2. "IRL이 supervised reward보다 원리적으로 더 맞다"는 가정

현재 결과가 부정하고 있다.

IRL은 보통 reward가 직접 없거나, sparse하거나, preference만 있을 때 빛난다.
LIDC에서는 이미 reward proxy가 있다. 리뷰어는 이렇게 찌를 수 있다:

> "reward proxy가 있는데 왜 더 어렵고 불안정한 IRL을 쓰나요?"

이 질문에 답하려면 C가 B보다 확실히 낫거나, 최소한 **label efficiency / robustness / missing supervision**에서 강해야 한다.

### 3. "Energy map"이라고 부를 수 있다는 가정

현재 reward 정의는 case/nodule-level score variance다:

```
r(x) = -Var_i[malignancy_score_i(x)]
```

그런데 말하는 산출물은 "uncertainty map"이다. 여기서 논리 점프가 있다:
- 입력: CT volume
- supervision: lesion-level or case-level disagreement
- 출력: spatial map

리뷰어: "어떤 supervision으로 voxel/region-level energy map의 spatial validity를 보장하나요?"

→ map claim이 과장일 수 있다. "uncertainty-aware score/field"로 낮추는 게 더 안전하다. 정말 map을 주장하려면 localization sanity check가 필요하다.

### 4. "Flow Matching 교체가 중립적이다"는 가정

Apr 14 수정 내용 — SGLD negatives → FM policy samples, EBM freeze + policy gradient flow 조정 — 은 단순 백본 교체가 아니다. 사실상 **학습 게임의 구조를 바꾸고 있다.**

실패 원인이 "IRL 아이디어 자체"인지, 아니면 "DxMI → FM 재구성 과정의 optimization failure"인지 아직 분리되지 않았다.

### 5. "Clinical correlation + ablation이면 충분하다"는 가정

거의 충분하지만, 아직 아니다. ICLR급으로 가려면 최소한 하나는 더 있어야 한다:
- robustness analysis
- calibration analysis 해석 정교화
- label efficiency
- qualitative case studies with failure modes
- uncertainty localization sanity check

지금 표만으로는 "works/doesn't work" 수준이지, "왜 중요한가"까지 못 간다.

---

## 이번 주에 진짜 해야 할 일

순서가 중요하다. 지금은 재실험부터 돌리면 안 된다.

### 1. 학습이 논리적으로 맞는가 먼저 증명해라 (smoke test)

CPU 2-step smoke test는 그냥 체크리스트가 아니라 **최우선**이다.

검증해야 할 것:
- `reward_loss.backward()` 시 `VelocityField`에 실제 gradient가 흐르는가
- EBM freeze 시 원치 않는 grad leakage가 없는가
- demo sample vs policy sample energy ordering이 초반부터 어떻게 변하는가
- `reward_grad_norm`, `policy_grad_norm`, `fm_sample_energy` 로그가 시간에 따라 일관적인가

여기서 하나라도 이상하면 GPU 시간 쓰는 건 낭비다.

### 2. 포화 억제를 독립 과제로 잡아라

지금은 성능 이전에 dynamics 복구가 먼저다.

필수 로그:
- `e_pos`, `e_neg` 평균뿐 아니라 분산/분위수
- energy histogram over train/val
- case별 energy rank stability
- epoch별 Spearman curve
- reward distribution before/after normalization

봐야 하는 건 절대값이 아니라 **랭킹이 살아 있는지**다.

### 3. C vs B 정면승부만 보지 말고, 승부 조건을 바꿔라

지금 같은 full supervision setting에서 B가 강하면 당연하다. IRL의 장점을 보여줄 수 있는 판을 설계해야 한다.

예시:
- annotator labels 일부만 사용했을 때
- coarse disagreement bin만 줬을 때
- pairwise preference만 줬을 때
- noisy reward proxy에서 robustness 비교
- out-of-split generalization

**IRL이 빛나는 조건을 설계하지 않으면 B를 못 이길 수도 있다. 그건 구현 실패가 아니라 실험 설계 실패다.**

### 4. Claim을 낮출 준비를 해라

지금 상태에서 "direct expression of radiologist confusion regions" 같은 문구는 위험하다.

현재 안전한 표현:
- clinically aligned uncertainty score
- disagreement-aware energy function
- proxy for inter-reader uncertainty

논문은 멋있는 문장보다 살아남는 문장이 중요하다.

### 5. 실패 시 플랜 B를 미리 설계해라

만약 2주 안에 C가 B를 못 넘으면, 프로젝트의 중심 질문을 바꿔야 한다.

예시:
- "IRL beats supervised"가 아니라 "IRL-style energy learning provides a principled framework, but supervised reward is a strong upper baseline"
- 혹은 "Clinical disagreement estimation with energy-based generative modeling"으로 축 이동

**IRL superiority를 메인 claim으로 걸고 죽지 마라. 지금 실험은 그 claim을 아직 지지하지 않는다.**

---

## 합리적인 우선순위

### 이번 주 (Apr W3)

- [ ] Apr 14 수정분 코드 리뷰/커밋
- [ ] smoke test로 gradient path 완전 검증
- [ ] energy saturation 원인 분해
- [ ] A/C 소규모 재학습으로 dynamics만 확인

### 그 다음

- [ ] full retrain 전에 hyperparameter sweep
- [ ] 특히 `temperature`, `l2_reg`, negative sampling hardness, reward normalization 재점검
- [ ] B/C 비교를 full supervision 외 setting까지 확장

### 5월 안에

- [ ] "C가 왜 B보다 나아야 하는가"를 실험적으로 재정의
- [ ] 아니면 claim을 수정

---

## 프로젝트 성공 조건 재정의

성공 조건은 C가 무조건 최고가 되는 게 아니다.

**1. C > B를 명확히 보인다** → 원래 플랜 유지

**2. 왜 B가 강하고 C가 실패하는지 메커니즘까지 설명한다** → 실패 분석 자체를 contribution으로 일부 흡수

지금 가장 나쁜 상태는 C가 B보다 못한데도 하이퍼파라미터 더 돌리면서 희망회로만 돌리는 것이다.

---

## 최종 판정

| 항목 | 평가 |
|------|------|
| 아이디어 | 강함 |
| 데이터 선택 | 적절 |
| 문제 정의 | 매력적 |
| 실험적 증거 | 부족 |
| 핵심 claim | 미입증 |
| 가장 큰 위험 | IRL의 필요성 붕괴 |

**지금 이 프로젝트는 죽지 않았다. 하지만 아직 ICLR 2027 submission candidate도 아니다.**

지금 해야 할 일은 더 많이 돌리는 게 아니라:

1. C가 왜 망했는지 구조적으로 밝히고
2. IRL이 이 setting에서 정말 필요한지 증명 가능한 판을 다시 설계하는 것

---

*작성: 2026-04-17*
