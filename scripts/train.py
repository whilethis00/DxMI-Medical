"""
학습 진입점.

단일 GPU:
  python scripts/train.py --config configs/ebm_baseline.yaml

단일 GPU 재시작:
  python scripts/train.py --config configs/ebm_baseline.yaml --resume outputs/ebm_baseline/ckpt_epoch0100.pt

DDP (다중 GPU):
  torchrun --nproc_per_node=2 scripts/train.py --config configs/ebm_baseline.yaml
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py --config configs/ebm_baseline.yaml --resume ...
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from src.evaluation.metrics import evaluate


# ── DDP 유틸 ──────────────────────────────────────────────────────────────────

def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_ddp() else 0

def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1

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


# ── 체크포인트 ─────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path):
    if is_main():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)


def load_checkpoint(path: str, raw_model, opt=None, scheduler=None) -> int:
    """체크포인트 로드. 재시작 epoch 번호 반환."""
    ckpt = torch.load(path, map_location="cpu")
    # 키 이름이 'model'이거나 'ebm'이거나 (IRL 체크포인트 호환)
    model_state = ckpt.get("model") or ckpt.get("ebm")
    raw_model.load_state_dict(model_state)
    if opt and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    if is_main():
        print(f"[INFO] Resumed from {path} (next epoch: {start_epoch})", flush=True)
    return start_epoch


def load_irl_checkpoint(path: str, raw_ebm, raw_vf) -> int:
    """IRL 체크포인트 로드. 재시작 epoch 번호 반환."""
    ckpt = torch.load(path, map_location="cpu")
    raw_ebm.load_state_dict(ckpt["ebm"])
    raw_vf.load_state_dict(ckpt["vf"])
    start_epoch = ckpt.get("epoch", 0) + 1
    if is_main():
        print(f"[INFO] Resumed IRL from {path} (next epoch: {start_epoch})", flush=True)
    return start_epoch


# ── 스케줄러 (warmup + cosine) ─────────────────────────────────────────────────

def make_scheduler(opt, cfg):
    warmup = cfg["training"].get("warmup_epochs", 0)
    total  = cfg["training"]["epochs"]
    if warmup > 0:
        return torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, total - warmup)
                ),
            ],
            milestones=[warmup],
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)


# ── Early Stopping ─────────────────────────────────────────────────────────────

class EarlyStopper:
    """Spearman ρ 기준 patience-based early stopping."""

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val: float, epoch: int) -> bool:
        """개선되면 False, patience 초과하면 True 반환."""
        if val > self.best + self.min_delta:
            self.best = val
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        return self.counter >= self.patience


def should_stop_ddp(stop_on_rank0: bool, device) -> bool:
    """rank0의 stop 여부를 전체 rank에 broadcast."""
    if not is_ddp():
        return stop_on_rank0
    t = torch.tensor(int(stop_on_rank0), device=device)
    dist.broadcast(t, src=0)
    return bool(t.item())


# ── DataLoader ─────────────────────────────────────────────────────────────────

def make_ddp_loaders(cfg: dict, splits_dir: str) -> dict:
    """학습용: DistributedSampler / val·test: 단순 로더 (rank0에서만 평가)."""
    splits_dir = Path(splits_dir)
    loaders = {}
    for split in ["train", "val", "test"]:
        csv = splits_dir / f"{split}.csv"
        if not csv.exists():
            continue
        ds = LIDCDataset(csv, augment=(split == "train"))
        if split == "train" and is_ddp():
            sampler = DistributedSampler(ds, shuffle=True)
        else:
            sampler = None
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


# ── Validation (rank0 only) ────────────────────────────────────────────────────

def run_val(raw_ebm, val_loader, device):
    """rank0에서만 evaluation 실행. 다른 rank는 None 반환."""
    if not is_main() or val_loader is None:
        return None
    return evaluate(raw_ebm, val_loader, device)


# ── Ablation A: EBM only (No IRL) ─────────────────────────────────────────────

def train_ebm_only(cfg: dict, device: torch.device, resume_path: str = None):
    """Ablation A: EBM contrastive divergence, no IRL. DDP + early stopping + resume 지원."""
    from src.models.irl import ReplayBuffer

    loaders = make_ddp_loaders(cfg, cfg["data"]["splits_dir"])

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    if is_ddp():
        ebm = DDP(ebm, device_ids=[device.index])
    raw_ebm = ebm.module if is_ddp() else ebm

    opt = torch.optim.Adam(
        raw_ebm.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = make_scheduler(opt, cfg)

    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, raw_ebm, opt, scheduler)

    replay = ReplayBuffer(
        max_size=cfg["ebm"]["replay_buffer_size"] // world_size(),
        shape=(1, 48, 48, 48),
    )

    out_dir       = Path(cfg["logging"]["output_dir"])
    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    patience      = cfg["training"].get("early_stop_patience", 0)
    stopper       = EarlyStopper(patience) if patience > 0 else None

    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if is_ddp():
            loaders["train"].sampler.set_epoch(epoch)
        ebm.train()
        epoch_loss = 0.0

        for batch in loaders["train"]:
            x = batch["patch"].to(device)

            x_init = replay.sample(x.size(0), cfg["ebm"]["replay_prob"], device)
            x_neg  = raw_ebm.sample_langevin(
                x_init,
                n_steps     = cfg["ebm"]["sgld_steps"],
                step_size   = cfg["ebm"]["sgld_step_size"],
                noise_scale = cfg["ebm"]["sgld_noise_scale"],
            )
            replay.push(x_neg)

            opt.zero_grad()
            # DDP 래퍼를 통한 forward → backward 시 gradient allreduce 보장
            loss, metrics = contrastive_divergence_loss(
                ebm, x, x_neg, l2_reg=cfg["training"]["l2_reg"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_ebm.parameters(), cfg["training"]["grad_clip"])
            opt.step()

            epoch_loss  += metrics["loss"]
            global_step += 1

            if is_main() and global_step % log_interval == 0:
                print(
                    f"[rank0|step {global_step}] loss={metrics['loss']:.4f} "
                    f"cd={metrics['cd_loss']:.4f} reg={metrics['reg_loss']:.4f} "
                    f"e_pos={metrics['e_pos']:.3f} e_neg={metrics['e_neg']:.3f} "
                    f"e_pos_std={metrics['e_pos_std']:.3f} e_neg_std={metrics['e_neg_std']:.3f} "
                    f"x_neg[min/mean/max]={metrics['x_neg_min']:.3f}/"
                    f"{metrics['x_neg_mean']:.3f}/{metrics['x_neg_max']:.3f}",
                    flush=True,
                )

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # Validation
            ebm.eval()
            val_result = run_val(raw_ebm, loaders.get("val"), device)
            ebm.train()
            stop_flag = False
            if stopper is not None and is_main() and val_result is not None:
                stop_flag = stopper.step(val_result.rho, epoch + 1)

            if is_main():
                avg_loss = epoch_loss / len(loaders["train"])
                print(f"[epoch {epoch+1}] avg_loss={avg_loss:.4f}", flush=True)
                if val_result:
                    print(f"[epoch {epoch+1}] val: {val_result}", flush=True)
                    if stopper is not None:
                        print(
                            f"[epoch {epoch+1}] best_val_epoch={stopper.best_epoch} "
                            f"best_rho={stopper.best:.4f} patience={stopper.counter}/{stopper.patience}",
                            flush=True,
                        )
                save_checkpoint(
                    {
                        "epoch":     epoch,
                        "model":     raw_ebm.state_dict(),
                        "opt":       opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
                )

            # Early stopping (all ranks must reach this together for DDP broadcast)
            if stopper is not None:
                if should_stop_ddp(stop_flag, device):
                    if is_main():
                        print(
                            f"[INFO] Early stop at epoch {epoch+1} "
                            f"(best ρ={stopper.best:.4f} @ epoch {stopper.best_epoch})",
                            flush=True,
                        )
                    break

    cleanup_ddp()


# ── Ablation B: Supervised Reward ─────────────────────────────────────────────

def train_supervised(cfg: dict, device: torch.device, resume_path: str = None):
    """
    Ablation B: EBM을 GT malignancy_var로 직접 지도 학습.
    Loss: MSE(E(x), malignancy_var)  + λ||E||² (energy regularization)
    """
    loaders = make_ddp_loaders(cfg, cfg["data"]["splits_dir"])

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    if is_ddp():
        ebm = DDP(ebm, device_ids=[device.index])
    raw_ebm = ebm.module if is_ddp() else ebm

    opt = torch.optim.Adam(
        raw_ebm.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = make_scheduler(opt, cfg)

    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint(resume_path, raw_ebm, opt, scheduler)

    out_dir       = Path(cfg["logging"]["output_dir"])
    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    patience      = cfg["training"].get("early_stop_patience", 0)
    stopper       = EarlyStopper(patience) if patience > 0 else None

    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if is_ddp():
            loaders["train"].sampler.set_epoch(epoch)
        ebm.train()
        epoch_loss = 0.0

        for batch in loaders["train"]:
            x       = batch["patch"].to(device)
            mal_var = batch["malignancy_var"].to(device)  # GT 불확실성 레이블

            opt.zero_grad()
            energy = ebm(x)  # (B,)
            # MSE: 높은 disagreement → 높은 energy 학습
            mse_loss = F.mse_loss(energy, mal_var)
            reg_loss = cfg["training"]["l2_reg"] * (energy ** 2).mean()
            loss     = mse_loss + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_ebm.parameters(), cfg["training"]["grad_clip"])
            opt.step()

            epoch_loss  += loss.item()
            global_step += 1

            if is_main() and global_step % log_interval == 0:
                print(
                    f"[rank0|step {global_step}] loss={loss.item():.4f} "
                    f"mse={mse_loss.item():.4f}",
                    flush=True,
                )

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            ebm.eval()
            val_result = run_val(raw_ebm, loaders.get("val"), device)
            ebm.train()
            stop_flag = False
            if stopper is not None and is_main() and val_result is not None:
                stop_flag = stopper.step(val_result.rho, epoch + 1)

            if is_main():
                avg_loss = epoch_loss / len(loaders["train"])
                print(f"[epoch {epoch+1}] avg_loss={avg_loss:.4f}", flush=True)
                if val_result:
                    print(f"[epoch {epoch+1}] val: {val_result}", flush=True)
                    if stopper is not None:
                        print(
                            f"[epoch {epoch+1}] best_val_epoch={stopper.best_epoch} "
                            f"best_rho={stopper.best:.4f} patience={stopper.counter}/{stopper.patience}",
                            flush=True,
                        )
                save_checkpoint(
                    {
                        "epoch":     epoch,
                        "model":     raw_ebm.state_dict(),
                        "opt":       opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
                )

            if stopper is not None:
                if should_stop_ddp(stop_flag, device):
                    if is_main():
                        print(
                            f"[INFO] Early stop at epoch {epoch+1} "
                            f"(best ρ={stopper.best:.4f} @ epoch {stopper.best_epoch})",
                            flush=True,
                        )
                    break

    cleanup_ddp()


# ── Ablation C: MaxEnt IRL (제안 모델) ──────────────────────────────────────────

def train_irl(cfg: dict, device: torch.device, resume_path: str = None):
    """Ablation C: MaxEnt IRL. DDP + early stopping + resume 지원."""
    loaders = make_ddp_loaders(cfg, cfg["data"]["splits_dir"])

    ebm = EBM(base_ch=cfg["model"]["base_ch"]).to(device)
    vf  = VelocityField(base_ch=cfg["model"]["base_ch"]).to(device)

    if is_ddp():
        ebm = DDP(ebm, device_ids=[device.index])
        vf  = DDP(vf,  device_ids=[device.index])

    irl_cfg = IRLConfig(
        reward_lr          = cfg["training"]["reward_lr"],
        fm_lr              = cfg["training"]["fm_lr"],
        sgld_steps         = cfg["ebm"]["sgld_steps"],
        sgld_step_size     = cfg["ebm"]["sgld_step_size"],
        sgld_noise_scale   = cfg["ebm"]["sgld_noise_scale"],
        replay_buffer_size = cfg["ebm"]["replay_buffer_size"] // world_size(),
        replay_prob        = cfg["ebm"]["replay_prob"],
        l2_reg             = cfg["training"]["l2_reg"],
        energy_clamp       = cfg["training"].get("energy_clamp", None),
        grad_clip          = cfg["training"]["grad_clip"],
        reward_steps_per_iter = cfg["training"].get("reward_steps_per_iter", 5),
        fm_steps_per_iter     = cfg["training"].get("fm_steps_per_iter", 10),
        policy_sample_steps   = cfg["training"].get("policy_sample_steps", 32),
        policy_grad_steps     = cfg["training"].get("policy_grad_steps", 8),
        reward_weight         = cfg["training"].get("reward_weight", 0.1),
        sgld_permanent_ratio  = cfg["training"].get("sgld_permanent_ratio", 0.2),
        fm_gate_sep_std_threshold = cfg["training"].get("fm_gate_sep_std_threshold", 10.0),
        fm_gate_consecutive   = cfg["training"].get("fm_gate_consecutive", 3),
        fm_gate_check_interval= cfg["training"].get("fm_gate_check_interval", 50),
        sep_std_ema_alpha     = cfg["training"].get("sep_std_ema_alpha", 0.1),
        reward_cd_weight      = cfg["training"].get("reward_cd_weight", 0.0),
        reward_cd_temp        = cfg["training"].get("reward_cd_temp", 1.0),
    )
    # DDP 래퍼를 그대로 전달 → MaxEntIRL 내부에서 gradient sync 보장
    irl = MaxEntIRL(ebm, vf, irl_cfg, device)

    raw_ebm = irl._raw_ebm

    start_epoch = 0
    if resume_path:
        start_epoch = load_irl_checkpoint(resume_path, raw_ebm, irl._raw_vf)

    out_dir       = Path(cfg["logging"]["output_dir"])
    log_interval  = cfg["logging"]["log_interval"]
    save_interval = cfg["logging"]["save_interval"]
    patience      = cfg["training"].get("early_stop_patience", 0)
    stopper       = EarlyStopper(patience) if patience > 0 else None

    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    _log_file = out_dir / "train.log"

    def _log(msg: str):
        if not is_main():
            return
        print(msg, flush=True)
        with open(_log_file, "a") as f:
            f.write(msg + "\n")

    global_step = 0

    for epoch in range(start_epoch, cfg["training"]["epochs"]):
        if is_ddp():
            loaders["train"].sampler.set_epoch(epoch)
        ebm.train()
        vf.train()

        for batch in loaders["train"]:
            x      = batch["patch"].to(device)
            reward = batch["reward"].to(device) if "reward" in batch else None
            metrics = irl.step(x, reward=reward)
            global_step += 1

            if is_main() and global_step % log_interval == 0:
                parts = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                _log(f"[rank0|step {global_step}] {parts}")

        if (epoch + 1) % save_interval == 0:
            ebm.eval()
            val_result = run_val(raw_ebm, loaders.get("val"), device)
            ebm.train()
            stop_flag = False
            if stopper is not None and is_main() and val_result is not None:
                stop_flag = stopper.step(val_result.rho, epoch + 1)

            if is_main():
                _log(f"[epoch {epoch+1}] checkpoint saved")
                if val_result:
                    _log(f"[epoch {epoch+1}] val: {val_result}")
                    if stopper is not None:
                        _log(
                            f"[epoch {epoch+1}] best_val_epoch={stopper.best_epoch} "
                            f"best_rho={stopper.best:.4f} patience={stopper.counter}/{stopper.patience}"
                        )
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "ebm":   raw_ebm.state_dict(),
                        "vf":    irl._raw_vf.state_dict(),
                    },
                    out_dir / f"ckpt_epoch{epoch+1:04d}.pt",
                )

            if stopper is not None:
                if should_stop_ddp(stop_flag, device):
                    if is_main():
                        print(
                            f"[INFO] Early stop at epoch {epoch+1} "
                            f"(best ρ={stopper.best:.4f} @ epoch {stopper.best_epoch})",
                            flush=True,
                        )
                    break

    cleanup_ddp()


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="체크포인트 경로 (예: outputs/ebm_baseline/ckpt_epoch0100.pt)",
    )
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        local_rank = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if is_main():
        print(f"[INFO] device={device}, world_size={world_size()}", flush=True)

    cfg = load_config(args.config)
    set_seed(args.seed)

    exp_type = cfg.get("experiment", {}).get("type", "ebm_only")
    if exp_type == "irl":
        train_irl(cfg, device, resume_path=args.resume)
    elif exp_type == "supervised":
        train_supervised(cfg, device, resume_path=args.resume)
    else:
        train_ebm_only(cfg, device, resume_path=args.resume)


if __name__ == "__main__":
    main()
