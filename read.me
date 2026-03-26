# DxMI Medical — 프로젝트 파악 요약

## 프로젝트 개요

**정식 명칭:** DxMI Medical
**타겟 학회:** ICLR 2027
**서브미션 마감:** 2026년 9월 말
**베이스 논문:** DxMI (NeurIPS 2024) — Maximum Entropy IRL + EBM 기반 Diffusion 모델

**한 줄 핵심:**
영상의학과 의사의 진단 행동을 IRL(역강화학습)로 역추론해서, "의사가 실제로 헷갈리는 영역"을 에너지 함수로 직접 표현하는 임상 기반 의료 불확실성 추정 모델.

---

## 왜 이 연구가 필요한가

### 기존 방법의 한계

| 방법 | 문제점 |
|---|---|
| MC Dropout | 느리고, 임상 불확실성과 무관 |
| Deep Ensemble | 비용 크고, 모델 간 통계적 분산만 측정 |
| Diffusion-based uncertainty | 학습 데이터 이탈 정도만 봄, 임상 의미 없음 |

### 핵심 문제 재정의

- 기존: **"모델이 얼마나 불확실한가?"**
- 이 연구: **"영상의학과 의사가 이 케이스를 보고 얼마나 헷갈릴까?"**

이 두 질문의 답은 다르다. 모델이 확신하는 케이스도 의사 간 의견이 갈릴 수 있고, 반대도 가능하다.

---

## 핵심 아이디어 — IRL로 reward 역추론

```
Expert trajectory  =  영상의학과 의사의 진단 행동 (어떤 케이스에서 판독 의견이 갈리는가)
IRL로 추론한 reward  =  임상적으로 의미있는 불확실성 신호
학습된 energy function  =  "의사가 헷갈리는 영역"의 직접적 표현
```

**왜 직접 supervised로 안 하는가?**
"이 이미지가 얼마나 임상적으로 불확실한가"를 수식으로 미리 정의하는 것 자체가 불가능하다. IRL은 reward를 모른 채, 의사의 행동 패턴으로부터 reward를 역추론한다.

---

## 방법론 구조

```
CT / MRI 볼륨
      ↓
Flow Matching backbone (OT-CFM, 원본 DxMI의 DDPM 교체)
      ↓  ←────────── EBM이 reward 신호 제공
Energy-Based Model (EBM)
      ↓
임상 기반 uncertainty map
(energy 높음 = 의사 간 disagreement 높은 영역)
```

### 세 가지 핵심 구성 요소

**1. Flow Matching backbone (OT-CFM)**
- 원본 DxMI의 DDPM을 OT-CFM으로 교체
- trajectory가 직선에 가까워 step 수 감소, 학습 안정성 향상
- novelty는 reward 설계에 집중

**2. EBM — 임상 에너지 함수**

$$q(x) = \frac{1}{Z} \exp(-E_\theta(x) / \tau)$$

에너지의 의미 재정의:

$$E_\theta(x) \approx \text{Var}_{i \in [N]}[\text{score}_i(x)]$$

- energy 높음 → 의사 간 판독 분산 큼 → 임상적으로 불확실한 케이스

**3. MaxEnt IRL reward**

$$r(x) = -\text{Var}_{i=1}^{4}[\text{malignancy\_score}_i(x)]$$

- LIDC-IDRI: 4명의 영상의학과 의사가 독립적으로 매긴 악성도 점수(1~5)의 분산 활용
- reward 높음 = 의사들이 일치 = 임상적으로 확실한 케이스

---

## 데이터셋 — LIDC-IDRI

| 항목 | 내용 |
|---|---|
| 모달리티 | CT |
| 케이스 수 | 1,018명 |
| 병변 수 | 2,635개 폐결절 |
| 어노테이션 | 영상의학과 의사 4명이 독립적으로 악성도 점수(1~5) 매김 |
| 공개 여부 | 공개 데이터셋, IRB 불필요 |

LIDC-IDRI를 선택한 이유: 4명의 독립 판독 → inter-reader variance 직접 계산 가능 → 이 variance가 "임상적 불확실성"의 ground truth proxy.

---

## 합격을 결정하는 두 숫자

### 1. IRL Ablation (방법론 타당성)

| 버전 | 설명 | 예상 성능 |
|---|---|---|
| A | EBM만, IRL 없음 (원본 DxMI) | 최하 |
| B | Supervised reward (variance 직접 loss 주입) | 중간 |
| C (ours) | MaxEnt IRL로 reward 역추론 | 최상 |

C > B > A, 특히 **C vs B 차이**가 핵심. C와 B가 비슷하면 "IRL 불필요" → reject.

### 2. Clinical Correlation (임상 유효성)

```
우리 모델의 energy(x) 높은 케이스
          ↕
LIDC annotator 4명의 variance 높은 케이스
→ Spearman ρ 계산 + p-value (p < 0.05 필수)
```

**둘 다 나와야 한다.**
- ablation만 → "수치상 낫긴 한데 의료에 왜 필요한지 모르겠다" → reject
- clinical corr만 → "결과는 좋은데 IRL이 그 원인인지 모르겠다" → reject

---

## 구현 로드맵

| 기간 | 작업 |
|---|---|
| 2026년 4월 | LIDC-IDRI 전처리 + DxMI 원본 코드 재현 |
| 2026년 5월 | Reward 함수 v1 설계 + EBM 연결 실험 |
| 2026년 6월 | Flow matching backbone 교체 + Ablation 설계 |
| 2026년 7월 | Ablation A/B/C 완료 + Clinical correlation 분석 |
| 2026년 8월 | 논문 초안 작성 |
| 2026년 9월 말 | ICLR 2027 제출 |

**Fallback:** ICLR reject → AAAI 2027 (10월) → CVPR 2027 (11월)

---

## 진행 상황 체크리스트

- [ ] LIDC-IDRI 다운로드 + 전처리 스크립트
- [ ] DxMI 원본 코드 재현 (2D synthetic 결과 확인)
- [ ] Annotator variance → scalar reward 계산
- [ ] Flow matching (OT-CFM) backbone 교체
- [ ] EBM ↔ reward 연결 코드
- [ ] Ablation 버전 A / B / C 실험
- [ ] Clinical correlation (Spearman ρ) 분석
- [ ] 논문 작성

---

## 베이스 논문

```bibtex
@inproceedings{yoon2024dxmi,
  title     = {Maximum Entropy Inverse Reinforcement Learning of
               Diffusion Models with Energy-Based Models},
  author    = {Yoon, Sangwoong and Hwang, Himchan and Kwon, Dohyun
               and Noh, Yung-Kyun and Park, Frank C.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
```

GitHub: https://github.com/swyoon/Diffusion-by-MaxEntIRL

---

## 연간 플랜 내 위치

3개 프로젝트 연간 플랜 중 하나:
- H-NEMESIS (AAAI 2026)
- **DxMI Medical (ICLR 2027)** ← 현재 프로젝트
- latentVLA (ICRA 2027)
