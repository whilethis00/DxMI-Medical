"""
학습 진입점.
사용법:
  python scripts/train.py --config configs/ebm_baseline.yaml       # Ablation A (no IRL)
  python scripts/train.py --config configs/irl_maxent.yaml          # Ablation C (MaxEnt IRL)
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import make_dataloaders
from src.models.ebm import EBM, contrastive_divergence_loss
from src.models.flow_matching import VelocityField, ot_cfm_loss
from src.models.irl import MaxEntIRL, IRLConfig


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ── 학습 모드 ──────────────────────────────────────────────────────────────────

def train_ebm_only(cfg: dict, device: torch.device):
    """Ablation A: EBM만 contrastive divergence로 학습 (no IRL)."""
    from src.models.ebm import EBM
    from src.models.irl import ReplayBuffer

    loaders = make_dataloaders(
        cfg["data"]["splits_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    opt = torch.optim.Adam(
        ebm.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["epochs"]
    )

    replay = ReplayBuffer(
        max_size=cfg["ebm"]["replay_buffer_size"],
        shape=(1, 48, 48, 48),
    )

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]

    global_step = 0
    for epoch in range(cfg["training"]["epochs"]):
        ebm.train()
        epoch_loss = 0.0

        for batch in loaders["train"]:
            x = batch["patch"].to(device)

            x_init = replay.sample(x.size(0), cfg["ebm"]["replay_prob"], device)
            x_neg  = ebm.sample_langevin(
                x_init,
                n_steps    = cfg["ebm"]["sgld_steps"],
                step_size  = cfg["ebm"]["sgld_step_size"],
                noise_scale= cfg["ebm"]["sgld_noise_scale"],
            )
            replay.push(x_neg)

            opt.zero_grad()
            loss, metrics = contrastive_divergence_loss(
                ebm, x, x_neg, l2_reg=cfg["training"]["l2_reg"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), cfg["training"]["grad_clip"])
            opt.step()

            epoch_loss += metrics["loss"]
            global_step += 1

            if global_step % log_interval == 0:
                print(f"[step {global_step}] loss={metrics['loss']:.4f} "
                      f"e_pos={metrics['e_pos']:.3f} e_neg={metrics['e_neg']:.3f}")

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": ebm.state_dict(), "opt": opt.state_dict()},
                out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
            )
            print(f"[epoch {epoch+1}] avg_loss={epoch_loss/len(loaders['train']):.4f}")


def train_irl(cfg: dict, device: torch.device):
    """Ablation C: MaxEnt IRL (EBM reward + Flow Matching policy)."""
    loaders = make_dataloaders(
        cfg["data"]["splits_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    vf  = VelocityField(base_ch=cfg["model"]["base_ch"]).to(device)

    irl_cfg = IRLConfig(
        reward_lr         = cfg["training"]["reward_lr"],
        fm_lr             = cfg["training"]["fm_lr"],
        sgld_steps        = cfg["ebm"]["sgld_steps"],
        sgld_step_size    = cfg["ebm"]["sgld_step_size"],
        sgld_noise_scale  = cfg["ebm"]["sgld_noise_scale"],
        replay_buffer_size= cfg["ebm"]["replay_buffer_size"],
        replay_prob       = cfg["ebm"]["replay_prob"],
        l2_reg            = cfg["training"]["l2_reg"],
        grad_clip         = cfg["training"]["grad_clip"],
    )
    irl = MaxEntIRL(ebm, vf, irl_cfg, device)

    out_dir = Path(cfg["logging"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_interval = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]

    global_step = 0
    for epoch in range(cfg["training"]["epochs"]):
        ebm.train(); vf.train()

        for batch in loaders["train"]:
            x = batch["patch"].to(device)
            metrics = irl.step(x)
            global_step += 1

            if global_step % log_interval == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[step {global_step}] {parts}")

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "ebm":   ebm.state_dict(),
                    "vf":    vf.state_dict(),
                },
                out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
            )
            print(f"[epoch {epoch+1}] checkpoint saved")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] device: {device}")

    experiment_type = cfg.get("experiment", {}).get("type", "ebm_only")
    if experiment_type == "irl":
        train_irl(cfg, device)
    else:
        train_ebm_only(cfg, device)


if __name__ == "__main__":
    main()
