"""
Energy-Based Model (EBM) for 3D CT nodule patches.
- 3D ResNet-like backbone (48³ input)
- Scalar energy output: high energy = high annotator disagreement
- Loss: contrastive divergence (SGLD negative samples)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 3D ResNet 블록 ──────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.res  = ResBlock3D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.conv(x))


# ── EBM ────────────────────────────────────────────────────────────────────────

class EBM(nn.Module):
    """
    Input : (B, 1, 48, 48, 48) float32
    Output: (B,) scalar energy

    Architecture:
        48³ → DownBlock(1→32) → 24³
            → DownBlock(32→64) → 12³
            → DownBlock(64→128) → 6³
            → DownBlock(128→256) → 3³
            → GlobalAvgPool → 256
            → Linear(256 → 64) → SiLU → Linear(64 → 1)
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        ch = base_ch
        self.encoder = nn.Sequential(
            DownBlock3D(1,      ch),       # 48 → 24
            DownBlock3D(ch,     ch * 2),   # 24 → 12
            DownBlock3D(ch * 2, ch * 4),   # 12 → 6
            DownBlock3D(ch * 4, ch * 8),   # 6  → 3
        )
        hidden = ch * 8
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns energy E(x), shape (B,)."""
        feat = self.encoder(x)
        return self.head(feat).squeeze(-1)

    # ── SGLD negative sampling ────────────────────────────────────────────────

    @torch.no_grad()
    def sample_langevin(
        self,
        x_init: torch.Tensor,
        n_steps: int = 60,
        step_size: float = 10.0,
        noise_scale: float = 0.005,
    ) -> torch.Tensor:
        """
        Stochastic Gradient Langevin Dynamics (SGLD) starting from x_init.
        Used to generate negative samples for contrastive divergence.
        """
        x = x_init.clone().requires_grad_(True)
        for _ in range(n_steps):
            with torch.enable_grad():
                energy = self.forward(x).sum()
                grad = torch.autograd.grad(energy, x)[0]
            noise = torch.randn_like(x) * noise_scale
            x = (x - step_size * grad + noise).clamp(0.0, 1.0).detach().requires_grad_(True)
        return x.detach()


# ── 손실 함수 ──────────────────────────────────────────────────────────────────

def contrastive_divergence_loss(
    model: EBM,
    x_pos: torch.Tensor,
    x_neg: torch.Tensor,
    l2_reg: float = 1.0,
    energy_clamp: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    CD 손실: E(x+) - E(x-)  (minimize → lower energy for real data)

    Args:
        x_pos: (B, 1, 48, 48, 48) real patches
        x_neg: (B, 1, 48, 48, 48) SGLD-sampled negative patches
        l2_reg: energy regularization weight — equilibrium |E| = 1/(2*l2_reg).
                l2_reg=0.01 → ±50, l2_reg=0.1 → ±5, l2_reg=0.5 → ±1
        energy_clamp: if set, clamp |E| to [-energy_clamp, energy_clamp] before loss.
                      use only after l2_reg tuning fails to prevent numeric explosion.

    Returns:
        loss, metrics dict
    """
    e_pos = model(x_pos)
    e_neg = model(x_neg)

    if energy_clamp is not None:
        e_pos = e_pos.clamp(-energy_clamp, energy_clamp)
        e_neg = e_neg.clamp(-energy_clamp, energy_clamp)

    cd_loss  = (e_pos - e_neg).mean()
    reg_loss = l2_reg * (e_pos ** 2 + e_neg ** 2).mean()
    loss     = cd_loss + reg_loss

    metrics = {
        "loss":     loss.item(),
        "cd_loss":  cd_loss.item(),
        "reg_loss": reg_loss.item(),
        "e_pos":    e_pos.mean().item(),
        "e_neg":    e_neg.mean().item(),
        "e_pos_std": e_pos.std(unbiased=False).item(),
        "e_neg_std": e_neg.std(unbiased=False).item(),
        "x_neg_min": x_neg.min().item(),
        "x_neg_max": x_neg.max().item(),
        "x_neg_mean": x_neg.mean().item(),
    }
    return loss, metrics
