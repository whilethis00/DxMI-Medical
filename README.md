# DxMI Medical

Clinical ambiguity estimation for medical imaging with MaxEnt IRL, EBM, and Flow Matching.

The project targets ICLR 2027. The core question is whether clinically meaningful ambiguity can be learned from expert diagnostic behavior rather than from model-centric uncertainty alone.

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

- [Project diagnosis](docs/planning/Project_Diagnosis.md)
- [April week 3 progress](docs/progress/Apr_W3.md)
- [Code fix priority](docs/planning/Code_Fix_Priority.md)
- [Docs index](docs/README.md)

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

## Research Direction

Most uncertainty methods answer a model-centric question: how uncertain is the model? This project asks a clinical question instead: which cases are intrinsically ambiguous to expert radiologists?

Current framing:

- Expert disagreement is not treated as disposable label noise.
- LIDC-IDRI multi-reader malignancy scores provide a proxy for clinical ambiguity.
- EBM energy is interpreted as a clinical ambiguity score.
- MaxEnt IRL is used to infer a latent clinical reward landscape from expert diagnostic behavior.
- Flow Matching acts as a differentiable policy for sampling and exploring clinically ambiguous regions.

## Current Experimental Status

The latest main line is `ebm_fm_gate_v3`.

- Seed 1 has validation and test results documented in [outputs/ebm_fm_gate_v3_20260423/RESULT.md](outputs/ebm_fm_gate_v3_20260423/RESULT.md).
- Seed 2 and seed 3 training logs exist, but their `RESULT.md` files still need full result writeups.
- The next critical step is the v3 3-seed test evaluation and comparison against v1/v2, supervised reward, and SGLD-only C baselines.

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
