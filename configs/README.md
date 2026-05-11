# Configs

Experiment configs are grouped by research line through filename prefixes.

## Baselines

- `ebm_baseline.yaml`: no-IRL EBM baseline
- `supervised_reward.yaml`: supervised reward baseline
- `irl_maxent.yaml`: original MaxEnt IRL baseline
- `irl_minirun.yaml`: small sanity run

## Weighted CD / SGLD-Only C

- `ebm_weighted_cd.yaml`
- `ebm_weighted_cd_r1.yaml`
- `ebm_weighted_cd_r2.yaml`
- `ebm_weighted_cd_r3.yaml`

## FM Gate Experiments

- `ebm_fm_gate_v1.yaml`
- `ebm_fm_gate_v2.yaml`
- `ebm_fm_gate_v3.yaml`
- `ebm_fm_gate_v3_s2.yaml`
- `ebm_fm_gate_v3_s3.yaml`

The `logging.output_dir` value is intentionally seed-specific for replicated runs so `ckpt_best_val.pt` can be discovered reliably.

## Diagnostics

- `ebm_collapse_sweep.yaml`
