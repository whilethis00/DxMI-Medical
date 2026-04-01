"""
학습 진입점.
단일 GPU:
  python scripts/train.py --config configs/ebm_baseline.yaml

DDP (다중 GPU):
  torchrun --nproc_per_node=4 scripts/train.py --config configs/ebm_baseline.yaml
  CUDA_VISIBLE_DEVICES=0,3,5,7 torchrun --nproc_per_node=4 scripts/train.py --config configs/ebm_baseline.yaml
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist
import numpy as np
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import LIDCDataset
from src.models.ebm import EBM, contrastive_divergence_loss
from src.models.flow_matching import VelocityField, ot_cfm_loss
from src.models.irl import MaxEntIRL, IRLConfig


# ── DDP 유틸 ──────────────────────────────────────────────────────────────────

def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_ddp() else 0

def is_main() -> bool:
    return rank() == 0

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    if is_ddp():
        dist.destroy_process_group()


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed + rank())
    np.random.seed(seed + rank())
    torch.cuda.manual_seed_all(seed + rank())


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state: dict, path: Path):
    if is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def make_ddp_loaders(cfg: dict, splits_dir: str) -> dict:
    """DDP-aware DataLoader 생성 (DistributedSampler 적용)."""
    splits_dir = Path(splits_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            continue
        ds = LIDCDataset(csv, augment=(split == "train"))
        sampler = DistributedSampler(ds, shuffle=(split == "train")) if is_ddp() else None
        loaders[split] = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=(sampler is None and split == "train"),
            sampler=sampler,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


# ── 학습 모드 ──────────────────────────────────────────────────────────────────

def train_ebm_only(cfg: dict, device: torch.device):
    """Ablation A: EBM contrastive divergence (no IRL). DDP 지원."""
    from src.models.irl import ReplayBuffer

    loaders = make_ddp_loaders(cfg, cfg["data"]["splits_dir"])

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    if is_ddp():
        ebm = DDP(ebm, device_ids=[device.index])

    raw_ebm = ebm.module if is_ddp() else ebm

    opt = torch.optim.Adam(
        ebm.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["epochs"]
    )

    # ReplayBuffer는 각 프로세스가 독립적으로 유지
    replay = ReplayBuffer(
        max_size=cfg["ebm"]["replay_buffer_size"] // max(1, dist.get_world_size() if is_ddp() else 1),
        shape=(1, 48, 48, 48),
    )

    out_dir = Path(cfg["logging"]["output_dir"])
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    global_step   = 0

    for epoch in range(cfg["training"]["epochs"]):
        if is_ddp():
            loaders["train"].sampler.set_epoch(epoch)
        ebm.train()
        epoch_loss = 0.0

        for batch in loaders["train"]:
            x = batch["patch"].to(device)

            x_init = replay.sample(x.size(0), cfg["ebm"]["replay_prob"], device)
            x_neg  = raw_ebm.sample_langevin(
                x_init,
                n_steps    = cfg["ebm"]["sgld_steps"],
                step_size  = cfg["ebm"]["sgld_step_size"],
                noise_scale= cfg["ebm"]["sgld_noise_scale"],
            )
            replay.push(x_neg)

            opt.zero_grad()
            loss, metrics = contrastive_divergence_loss(
                raw_ebm, x, x_neg, l2_reg=cfg["training"]["l2_reg"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ebm.parameters(), cfg["training"]["grad_clip"])
            opt.step()

            epoch_loss += metrics["loss"]
            global_step += 1

            if is_main() and global_step % log_interval == 0:
                print(f"[rank0|step {global_step}] loss={metrics['loss']:.4f} "
                      f"e_pos={metrics['e_pos']:.3f} e_neg={metrics['e_neg']:.3f}",
                      flush=True)

        scheduler.step()

        if is_main() and (epoch + 1) % save_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "model": raw_ebm.state_dict(), "opt": opt.state_dict()},
                out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
            )
            print(f"[epoch {epoch+1}] avg_loss={epoch_loss/len(loaders['train']):.4f}", flush=True)

    cleanup_ddp()


def train_irl(cfg: dict, device: torch.device):
    """Ablation C: MaxEnt IRL. DDP 지원."""
    loaders = make_ddp_loaders(cfg, cfg["data"]["splits_dir"])

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    vf  = VelocityField(base_ch=cfg["model"]["base_ch"]).to(device)

    if is_ddp():
        ebm = DDP(ebm, device_ids=[device.index])
        vf  = DDP(vf,  device_ids=[device.index])

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
    irl = MaxEntIRL(
        ebm.module if is_ddp() else ebm,
        vf.module  if is_ddp() else vf,
        irl_cfg, device
    )

    out_dir = Path(cfg["logging"]["output_dir"])
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    global_step   = 0

    for epoch in range(cfg["training"]["epochs"]):
        if is_ddp():
            loaders["train"].sampler.set_epoch(epoch)
        ebm.train(); vf.train()

        for batch in loaders["train"]:
            x = batch["patch"].to(device)
            metrics = irl.step(x)
            global_step += 1

            if is_main() and global_step % log_interval == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"[rank0|step {global_step}] {parts}", flush=True)

        if is_main() and (epoch + 1) % save_interval == 0:
            raw_ebm = ebm.module if is_ddp() else ebm
            raw_vf  = vf.module  if is_ddp() else vf
            save_checkpoint(
                {"epoch": epoch, "ebm": raw_ebm.state_dict(), "vf": raw_vf.state_dict()},
                out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
            )
            print(f"[epoch {epoch+1}] checkpoint saved", flush=True)

    cleanup_ddp()


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # DDP 여부는 torchrun이 설정한 환경변수로 자동 감지
    if "LOCAL_RANK" in os.environ:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_main():
        world = dist.get_world_size() if is_ddp() else 1
        print(f"[INFO] device={device}, world_size={world}", flush=True)

    cfg = load_config(args.config)
    set_seed(args.seed)

    experiment_type = cfg.get("experiment", {}).get("type", "ebm_only")
    if experiment_type == "irl":
        train_irl(cfg, device)
    else:
        train_ebm_only(cfg, device)


if __name__ == "__main__":
    main()
