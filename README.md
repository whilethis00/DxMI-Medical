# Learning Clinical Ambiguity from Expert Disagreement

This repository studies **clinical ambiguity learning** in medical imaging.

The project was previously organized as `DxMI Medical`, but the research direction is now sharper: the goal is not to build another generic uncertainty estimator. The goal is to learn the latent clinical ambiguity that is revealed by expert disagreement, judgments, and preferences.

Target venue: ICLR 2027.

## Core Thesis

Most uncertainty methods ask a model-centric question:

> How uncertain is the model?

Clinical decision-making often needs a different question:

> Is this case intrinsically ambiguous to expert clinicians?

Model uncertainty and clinical ambiguity are not the same object. A model can be confident on cases where radiologists disagree, and a model can be uncertain on cases where experts would quickly agree. This project treats expert disagreement not as disposable label noise, but as an observable proxy for a latent property: **clinical ambiguity**.

## Problem We Want to Solve

We want to learn **clinical ambiguity** in medical imaging.

The target is not simply lesion classification, out-of-distribution detection, or generic uncertainty estimation. The target is to estimate whether a medical case is likely to be interpreted inconsistently by expert clinicians because the visual evidence itself is borderline, mixed, or clinically ambiguous.

In LIDC-IDRI, four radiologists independently rate pulmonary nodule malignancy. Their disagreement is not treated as a nuisance to average away. It is treated as a measurable signal that reveals where clinical interpretation becomes unstable.

The problem statement is:

> Given medical images and partial expert judgment signals, learn a score that reflects latent clinical ambiguity: the likelihood that competent experts would disagree on the case.

## Research Question

Primary research question:

> Can clinical ambiguity be learned as a latent reward, rather than as a supervised label?

Operational version:

> Can we learn clinically meaningful ambiguity from expert disagreement patterns, judgments, or preferences, rather than from model uncertainty or direct variance regression alone?

This framing is important because LIDC-IDRI does not contain real diagnostic trajectories. It contains multiple expert malignancy ratings. Therefore the claim is not that we observe full expert behavior. The claim is that multi-expert judgments reveal a latent clinical ambiguity structure.

## Why Existing Work Is Insufficient

Existing uncertainty methods usually answer different questions:

- MC Dropout / Ensembles: how unstable are model parameters or predictions?
- Bayesian / epistemic uncertainty: where is the model undersupported by data?
- Aleatoric uncertainty: where is the label noisy?
- Diffusion uncertainty: how far is a sample from the learned data manifold?
- Supervised disagreement prediction: can annotator variance be regressed directly?

The proposed problem is different:

> Which cases are clinically ambiguous enough that experts are likely to disagree?

The strongest reviewer objection is expected to be:

> If annotator variance is available, why not just regress it?

The answer cannot be "IRL is more complex." The answer must be empirical and conceptual:

- Disagreement is only a proxy for latent ambiguity, not the ambiguity itself.
- In realistic settings, full multi-expert disagreement labels are sparse, noisy, missing, or only available as preferences.
- A reward-inference formulation should be most useful when supervision is weak, partial, noisy, or preference-based.

The gap is therefore not just "existing methods perform worse." The gap is conceptual:

> Existing methods usually model model uncertainty, label noise, or direct disagreement targets. They do not explicitly model clinical ambiguity as a latent structure revealed through expert judgments.

## Methodological View

The method is not primarily "EBM + Flow Matching + IRL." Those are implementation components.

The paper-level method is:

> Infer a latent clinical ambiguity reward from expert disagreement patterns, then use an energy-based generative policy to identify clinically ambiguous regions.

This directly answers the research question:

- If ambiguity is latent rather than directly observed, use reward inference rather than only regression.
- If disagreement is an imperfect proxy, learn an energy landscape that ranks cases by ambiguity instead of forcing calibrated probabilities from noisy labels.
- If supervision is sparse, noisy, or preference-based, use an IRL/preference-learning view that can consume weaker expert signals.
- If the ambiguity boundary matters, use Flow Matching as a differentiable policy to explore cases near that boundary.

Components:

1. **Expert Judgment Modeling**
   LIDC-IDRI provides four independent malignancy ratings. Their disagreement is used as an observable proxy for latent clinical ambiguity, not as a perfect ground-truth label.

2. **Latent Clinical Reward Inference**
   Cases with expert consensus should receive higher certainty reward. Cases with expert disagreement should induce lower reward or higher energy. MaxEnt IRL is used to infer this latent reward structure.

3. **Energy-Based Clinical Ambiguity Score**
   EBM energy is interpreted as a clinical ambiguity score. Higher energy should align with higher expert disagreement.

4. **Flow Matching as an Ambiguity-Boundary Policy**
   Flow Matching is not presented as merely a faster DDPM replacement. It is a differentiable policy for exploring the boundary between clinically certain and ambiguous cases.

## Experiments Required for a Strong Paper

Correlation with annotator variance is not enough. The paper needs experiments that prove the problem definition and justify IRL.

Required experimental blocks:

1. **Model uncertainty vs. clinical ambiguity**
   Show that standard uncertainty baselines do not reliably recover high-disagreement cases.

2. **Strong supervised variance regression**
   Compare against a strong direct-supervision baseline. A weak supervised baseline will not be credible.

3. **Limited expert labels**
   Train with 100%, 50%, 25%, and 10% of disagreement labels. IRL should degrade more gracefully than direct regression.

4. **Preference-only supervision**
   Use pairwise statements such as "case A is more ambiguous than case B." This is where reward inference has the clearest motivation.

5. **Missing/noisy experts**
   Remove annotators or inject rating noise. The method should remain robust when full 4-reader variance is unreliable.

6. **Qualitative clinical evidence**
   High-energy cases should correspond to clinically plausible ambiguity: unclear margins, mixed texture, borderline malignancy cues, or morphology that would plausibly split expert judgments.

## Current Experimental Status

The latest technical line is `ebm_fm_gate_v3`, which adds an FM sample quality filter to prevent FM negatives from entering the demo energy region.

Known results so far:

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

Interpretation:

- v3 validates the quality-filter idea on validation runs, but seed1 test does not beat v1.
- seed2 and seed3 need test evaluation before making any claim about v3.
- ECE remains poor for IRL/CD-based variants; ranking metrics are currently the primary evidence.
- The next research-critical experiments are not more small v3 tweaks. They are sparse/noisy/preference expert-supervision settings.

Key result files:

- [FM gate v3 seed1 result](outputs/ebm_fm_gate_v3_20260423/RESULT.md)
- [April week 3 progress](docs/progress/Apr_W3.md)
- [Project diagnosis](docs/planning/Project_Diagnosis.md)

## Repository Layout

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

Important docs:

- [Docs index](docs/README.md)
- [Project diagnosis](docs/planning/Project_Diagnosis.md)
- [April week 3 progress](docs/progress/Apr_W3.md)
- [Code fix priority](docs/planning/Code_Fix_Priority.md)

## Environment

- Conda env: `dxmi_medical`
- Python: `/home/introai26/miniconda3/envs/dxmi_medical/bin/python`

## Common Commands

Train an experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py \
    --config configs/<experiment>.yaml
```

Evaluate best validation checkpoints on the test split:

```bash
python scripts/eval_test.py \
    --config configs/ebm_fm_gate_v3.yaml \
    --ckpt outputs/ebm_fm_gate_v3_20260423/ckpt_best_val.pt
```

Run the current v3 3-seed test evaluation:

```bash
bash scripts/eval_v3_3seed.sh
```

## Reference

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
