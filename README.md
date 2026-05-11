# 전문가 불일치로부터 임상적 모호성 학습하기

이 저장소는 의료영상에서 **임상적 모호성 학습(clinical ambiguity learning)**을 연구합니다.

이 프로젝트는 이전에 `DxMI Medical`이라는 이름으로 정리되어 있었지만, 현재 연구 방향은 더 명확해졌습니다. 목표는 또 하나의 일반적인 불확실성 추정기를 만드는 것이 아닙니다. 전문가의 불일치, 판단, 선호에서 드러나는 잠재적인 임상적 모호성을 학습하는 것이 목표입니다.

목표 학회: ICLR 2027.

## 핵심 주장

대부분의 불확실성 방법은 모델 중심의 질문을 던집니다.

> 모델은 얼마나 불확실한가?

하지만 임상 의사결정에는 종종 다른 질문이 필요합니다.

> 이 사례는 전문 임상의에게 본질적으로 모호한가?

모델 불확실성과 임상적 모호성은 같은 대상이 아닙니다. 모델은 영상의학과 전문의들이 서로 다른 판단을 내리는 사례에서도 높은 확신을 보일 수 있고, 반대로 전문가들이 빠르게 합의할 사례에서도 불확실해할 수 있습니다. 이 프로젝트는 전문가 불일치를 버려야 할 라벨 노이즈로 보지 않고, **임상적 모호성**이라는 잠재 속성을 관측할 수 있게 해주는 신호로 다룹니다.

## 해결하려는 문제

우리는 의료영상에서 **임상적 모호성**을 학습하고자 합니다.

목표는 단순한 병변 분류, 분포 밖 탐지, 일반적인 불확실성 추정이 아닙니다. 목표는 어떤 의료 사례가 시각적 근거 자체의 경계성, 혼재성, 임상적 모호성 때문에 전문 임상의들 사이에서 일관되지 않게 해석될 가능성이 높은지를 추정하는 것입니다.

LIDC-IDRI에서는 네 명의 영상의학과 전문의가 폐 결절의 악성도를 독립적으로 평가합니다. 이들의 불일치는 평균내어 제거해야 할 방해 요소로 취급하지 않습니다. 임상 해석이 불안정해지는 지점을 드러내는 측정 가능한 신호로 사용합니다.

문제 정의는 다음과 같습니다.

> 의료영상과 부분적인 전문가 판단 신호가 주어졌을 때, 유능한 전문가들이 해당 사례에 대해 서로 다른 판단을 내릴 가능성, 즉 잠재적인 임상적 모호성을 반영하는 점수를 학습한다.

## 연구 질문

주요 연구 질문은 다음과 같습니다.

> 임상적 모호성을 지도 라벨이 아니라 잠재 보상으로 학습할 수 있는가?

운영적 질문은 다음과 같습니다.

> 모델 불확실성이나 직접적인 분산 회귀에만 의존하지 않고, 전문가 불일치 패턴, 판단, 선호로부터 임상적으로 의미 있는 모호성을 학습할 수 있는가?

이 프레이밍이 중요한 이유는 LIDC-IDRI가 실제 진단 궤적을 포함하지 않기 때문입니다. 이 데이터셋은 여러 전문가의 악성도 평정을 제공합니다. 따라서 이 연구의 주장은 완전한 전문가 행동을 관찰했다는 것이 아닙니다. 다중 전문가 판단이 잠재적인 임상적 모호성 구조를 드러낸다는 것입니다.

## 기존 연구가 충분하지 않은 이유

기존 불확실성 방법들은 대체로 다른 질문에 답합니다.

- MC Dropout / Ensembles: 모델 파라미터나 예측이 얼마나 불안정한가?
- Bayesian / epistemic uncertainty: 데이터가 모델을 충분히 뒷받침하지 못하는 영역은 어디인가?
- Aleatoric uncertainty: 라벨이 얼마나 noisy한가?
- Diffusion uncertainty: 샘플이 학습된 데이터 manifold에서 얼마나 떨어져 있는가?
- Supervised disagreement prediction: 주석자 분산을 직접 회귀할 수 있는가?

이 프로젝트가 다루는 문제는 다릅니다.

> 어떤 사례가 전문가들 사이의 불일치를 유발할 만큼 임상적으로 모호한가?

가장 강한 리뷰어 반론은 다음과 같을 가능성이 큽니다.

> annotator variance가 있는데, 왜 그냥 회귀하지 않는가?

이에 대한 답은 "IRL이 더 복잡하다"가 될 수 없습니다. 답은 경험적이고 개념적이어야 합니다.

- 불일치는 잠재적인 모호성 자체가 아니라 그 proxy입니다.
- 실제 환경에서는 완전한 다중 전문가 불일치 라벨이 희소하거나, noisy하거나, 누락되어 있거나, 선호 형태로만 제공될 수 있습니다.
- 보상 추론 formulation은 supervision이 약하거나, 부분적이거나, noisy하거나, preference-based일 때 가장 유용해야 합니다.

따라서 gap은 단순히 "기존 방법의 성능이 낮다"가 아닙니다. gap은 개념적입니다.

> 기존 방법들은 대개 모델 불확실성, 라벨 노이즈, 직접적인 불일치 타깃을 모델링합니다. 전문가 판단을 통해 드러나는 잠재 구조로서의 임상적 모호성을 명시적으로 모델링하지 않습니다.

## 방법론적 관점

이 방법의 핵심은 단순히 "EBM + Flow Matching + IRL"이 아닙니다. 이들은 구현 구성 요소입니다.

논문 수준의 방법은 다음과 같습니다.

> 전문가 불일치 패턴으로부터 잠재적인 임상적 모호성 보상을 추론하고, energy-based generative policy를 사용해 임상적으로 모호한 영역을 식별한다.

이는 연구 질문에 직접 답합니다.

- 모호성이 직접 관측되는 값이 아니라 잠재적인 값이라면, 회귀만 사용하는 대신 보상 추론을 사용합니다.
- 불일치가 불완전한 proxy라면, noisy label에서 calibrated probability를 강제로 맞추기보다 사례를 모호성 기준으로 rank하는 energy landscape를 학습합니다.
- supervision이 희소하거나, noisy하거나, preference-based라면, 더 약한 전문가 신호를 사용할 수 있는 IRL/preference-learning 관점을 사용합니다.
- 모호성 경계가 중요하다면, Flow Matching을 그 경계 부근의 사례를 탐색하는 differentiable policy로 사용합니다.

구성 요소:

1. **전문가 판단 모델링**
   LIDC-IDRI는 네 명의 독립적인 악성도 평정을 제공합니다. 이들의 불일치는 완벽한 ground-truth label이 아니라 잠재적인 임상적 모호성을 드러내는 관측 가능한 proxy로 사용됩니다.

2. **잠재 임상 보상 추론**
   전문가 합의가 높은 사례는 더 높은 certainty reward를 받아야 합니다. 전문가 불일치가 큰 사례는 더 낮은 reward 또는 더 높은 energy를 유도해야 합니다. MaxEnt IRL을 사용해 이 잠재 보상 구조를 추론합니다.

3. **Energy-Based 임상적 모호성 점수**
   EBM energy는 임상적 모호성 점수로 해석됩니다. 높은 energy는 높은 전문가 불일치와 정렬되어야 합니다.

4. **모호성 경계 정책으로서의 Flow Matching**
   Flow Matching은 단순히 DDPM을 빠르게 대체하는 방법으로 제시하지 않습니다. 임상적으로 확실한 사례와 모호한 사례 사이의 경계를 탐색하는 differentiable policy로 사용합니다.

## 강한 논문을 위해 필요한 실험

annotator variance와의 상관만으로는 충분하지 않습니다. 논문은 문제 정의를 입증하고 IRL 사용을 정당화하는 실험을 필요로 합니다.

필수 실험 블록:

1. **모델 불확실성 vs. 임상적 모호성**
   표준 불확실성 baseline이 high-disagreement 사례를 안정적으로 복원하지 못한다는 점을 보여야 합니다.

2. **강한 supervised variance regression**
   강한 직접 지도 baseline과 비교해야 합니다. 약한 supervised baseline은 설득력이 없습니다.

3. **제한된 전문가 라벨**
   불일치 라벨의 100%, 50%, 25%, 10%만 사용해 학습합니다. IRL은 direct regression보다 더 완만하게 성능이 저하되어야 합니다.

4. **Preference-only supervision**
   "사례 A가 사례 B보다 더 모호하다"와 같은 pairwise statement를 사용합니다. 보상 추론의 동기가 가장 명확해지는 설정입니다.

5. **누락되거나 noisy한 전문가**
   annotator를 제거하거나 rating noise를 주입합니다. 완전한 4-reader variance가 신뢰하기 어려울 때도 방법이 견고해야 합니다.

6. **정성적 임상 근거**
   high-energy 사례는 불분명한 margin, 혼재된 texture, 경계성 악성도 cue, 전문가 판단을 나눌 수 있는 morphology 등 임상적으로 그럴듯한 모호성과 대응되어야 합니다.

## 현재 실험 상태

최신 기술 라인은 `ebm_fm_gate_v3`입니다. 이 버전은 FM negative가 demo energy 영역으로 들어가는 것을 막기 위해 FM sample quality filter를 추가합니다.

현재까지 알려진 결과:

| Experiment | Split | Spearman rho | AUROC(E) | ECE | Status |
|---|---:|---:|---:|---:|---|
| Supervised reward B | val | 0.3028 | 0.6719 | 0.2095 | strong direct-supervision baseline |
| SGLD-only C baseline | test | 0.239 +/- 0.010 | - | ~0.79 | useful but calibration poor |
| FM gate v1 seed1 | test | 0.2885 | 0.7214 | 0.8127 | current test champion among FM-gate runs |
| FM gate v2 seed1 | test | 0.2383 | 0.7096 | 0.7885 | quality issue / generalization drop |
| FM gate v3 seed1 | val | 0.3124 | 0.7130 | 0.7934 | best-val improves over B on val |
| FM gate v3 seed1 | test | 0.2585 | 0.6946 | 0.8118 | passes, but below v1 seed1 |
| FM gate v3 seed2 | val | 0.3229 | 0.7398 | 0.7866 | test pending |
| FM gate v3 seed3 | val | 0.3044 | 0.7491 | 0.7933 | test pending |

해석:

- v3는 validation run에서 quality-filter 아이디어를 검증하지만, seed1 test는 v1을 넘지 못합니다.
- seed2와 seed3는 v3에 대해 어떤 claim을 하기 전에 test evaluation이 필요합니다.
- IRL/CD 기반 variant에서는 ECE가 여전히 좋지 않습니다. 현재 주된 근거는 ranking metric입니다.
- 다음으로 연구상 중요한 실험은 작은 v3 tweak이 아닙니다. sparse/noisy/preference expert-supervision 설정입니다.

주요 결과 파일:

- [FM gate v3 seed1 result](outputs/ebm_fm_gate_v3_20260423/RESULT.md)
- [April week 3 progress](docs/progress/Apr_W3.md)
- [Project diagnosis](docs/planning/Project_Diagnosis.md)

## 저장소 구조

```text
configs/       YAML experiment configs
data/          LIDC-IDRI raw, processed, and split files
docs/          project notes, plans, operations, and result summaries
notebooks/     exploratory analysis
outputs/       experiment logs, figures, and checkpoints
papers/        reference papers
scripts/       training, evaluation, data, and diagnostic entrypoints
src/           reusable package code
```

중요 문서:

- [Docs index](docs/README.md)
- [Project diagnosis](docs/planning/Project_Diagnosis.md)
- [April week 3 progress](docs/progress/Apr_W3.md)
- [Code fix priority](docs/planning/Code_Fix_Priority.md)

## 환경

- Conda env: `dxmi_medical`
- Python: `/home/introai26/miniconda3/envs/dxmi_medical/bin/python`

## 자주 쓰는 명령어

실험 학습:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/<experiment>.yaml
```

best validation checkpoint를 test split에서 평가:

```bash
python scripts/eval_test.py \
    --config configs/ebm_fm_gate_v3.yaml \
    --ckpt outputs/ebm_fm_gate_v3_20260423/ckpt_best_val.pt
```

현재 v3 3-seed test evaluation 실행:

```bash
bash scripts/eval_v3_3seed.sh
```

## 참고문헌

Base paper:

```bibtex
@inproceedings{yoon2024dxmi,
  title     = {Maximum Entropy Inverse Reinforcement Learning of
               Diffusion Models with Energy-Based Models},
  author    = {Yoon, Sangwoong and Hwang, Himchan and Kwon, Dohyun
               and Noh, Yung-Kyun and Park, Frank C.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

Local PDF: [papers/Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models.pdf](<papers/Maximum Entropy Inverse Reinforcement Learning of Diffusion Models with Energy-Based Models.pdf>)
