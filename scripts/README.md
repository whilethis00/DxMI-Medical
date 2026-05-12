# Scripts

Executable entrypoints are intentionally kept directly under `scripts/` so existing `torchrun`, PBS, and result-log commands remain stable.

## Training

- `train.py`: main training entrypoint
- `run_train.sh`: local ablation launcher
- `run_supervised_reward.sh`: block-2 supervised variance-regression baseline launcher
- `run_train_pbs.sh`: PBS launcher
- `run_repro.sh`: repeated seed launcher
- `run_sweep.sh`: collapse sweep launcher
- `run_v3_seed2.sh`, `run_v3_seed3.sh`: PBS launchers for v3 seed replication

## Evaluation

- `eval_test.py`: one-time test evaluation for best-val checkpoints
- `uncertainty_baselines.py`: malignancy-classifier uncertainty baselines for model uncertainty vs. clinical ambiguity
- `eval_v3.sh`: v1/v2/v3 comparison
- `eval_v3_3seed.sh`: v3 3-seed test evaluation
- `reeval_checkpoints.py`: historical checkpoint reevaluation
- `temperature_scaling.py`: post-hoc calibration check

## Data

- `download_lidc.py`: LIDC-IDRI download helper
- `preprocess_lidc.py`: preprocessing pipeline
- `verify_preprocess.py`: preprocessing verification and EDA figures

## Diagnostics

- `smoke_test.py`: gradient path smoke test
- `diagnose_c.py`: C checkpoint separation/collapse diagnosis
- `plot_training.py`: plot training curves for an output directory
