"""
Flow Matching (OT-CFM) for 3D CT nodule patches.

DxMI 원본의 DDPM 대신 Optimal Transport Conditional Flow Matching을 사용.
- Lipman et al. (2022) "Flow Matching for Generative Modeling"
- Tong et al. (2023) "Improving and Generalizing Flow Matching" (OT-CFM)

핵심: x_t = (1-t)*x_0 + t*x_1 으로 선형 보간 경로 구성 (OT plan 사용)
      velocity field v_θ(x_t, t) 학습 → dx/dt = v_θ(x_t, t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdyn.core import NeuralODE


# ── 시간 임베딩 ─────────────────────────────────────────────────────────────────

class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) float in [0,1] → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── 3D UNet-style velocity field ────────────────────────────────────────────────

class ResBlock3DTime(nn.Module):
    """ResBlock conditioned on time embedding."""
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # time conditioning: broadcast over spatial dims
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class VelocityField(nn.Module):
    """
    v_θ(x_t, t): (B, 1, 48, 48, 48) × (B,) → (B, 1, 48, 48, 48)

    U-Net: encoder (downsample) + decoder (upsample) with skip connections.
    Channels: 1 → 32 → 64 → 128 → 64 → 32 → 1
    """

    def __init__(self, base_ch: int = 32, time_dim: int = 128):
        super().__init__()
        ch = base_ch
        self.time_emb = SinusoidalTimeEmb(time_dim)

        # Encoder
        self.enc1 = nn.Conv3d(1, ch, 3, padding=1)
        self.enc1_res = ResBlock3DTime(ch, time_dim)
        self.down1 = nn.Conv3d(ch, ch * 2, 4, stride=2, padding=1)   # 48→24

        self.enc2_res = ResBlock3DTime(ch * 2, time_dim)
        self.down2 = nn.Conv3d(ch * 2, ch * 4, 4, stride=2, padding=1)  # 24→12

        self.enc3_res = ResBlock3DTime(ch * 4, time_dim)
        self.down3 = nn.Conv3d(ch * 4, ch * 4, 4, stride=2, padding=1)  # 12→6

        # Bottleneck
        self.mid1 = ResBlock3DTime(ch * 4, time_dim)
        self.mid2 = ResBlock3DTime(ch * 4, time_dim)

        # Decoder
        self.up3 = nn.ConvTranspose3d(ch * 4, ch * 4, 4, stride=2, padding=1)  # 6→12
        self.dec3_res = ResBlock3DTime(ch * 4 + ch * 4, time_dim)
        self.dec3_proj = nn.Conv3d(ch * 4 + ch * 4, ch * 4, 1)

        self.up2 = nn.ConvTranspose3d(ch * 4, ch * 2, 4, stride=2, padding=1)  # 12→24
        self.dec2_res = ResBlock3DTime(ch * 2 + ch * 2, time_dim)
        self.dec2_proj = nn.Conv3d(ch * 2 + ch * 2, ch * 2, 1)

        self.up1 = nn.ConvTranspose3d(ch * 2, ch, 4, stride=2, padding=1)  # 24→48
        self.dec1_res = ResBlock3DTime(ch + ch, time_dim)
        self.dec1_proj = nn.Conv3d(ch + ch, ch, 1)

        self.out = nn.Conv3d(ch, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)  # (B, time_dim)

        # Encoder
        h1 = self.enc1_res(self.enc1(x), t_emb)          # (B, ch, 48, 48, 48)
        h2 = self.enc2_res(self.down1(h1), t_emb)         # (B, 2ch, 24, 24, 24)
        h3 = self.enc3_res(self.down2(h2), t_emb)         # (B, 4ch, 12, 12, 12)
        hb = self.down3(h3)                                # (B, 4ch, 6, 6, 6)

        # Bottleneck
        hb = self.mid2(self.mid1(hb, t_emb), t_emb)

        # Decoder
        d3 = self.up3(hb)                                  # (B, 4ch, 12, 12, 12)
        d3 = self.dec3_proj(self.dec3_res(torch.cat([d3, h3], dim=1), t_emb))

        d2 = self.up2(d3)                                  # (B, 2ch, 24, 24, 24)
        d2 = self.dec2_proj(self.dec2_res(torch.cat([d2, h2], dim=1), t_emb))

        d1 = self.up1(d2)                                  # (B, ch, 48, 48, 48)
        d1 = self.dec1_proj(self.dec1_res(torch.cat([d1, h1], dim=1), t_emb))

        return self.out(d1)                                 # (B, 1, 48, 48, 48)


# ── OT-CFM 손실 ─────────────────────────────────────────────────────────────────

def ot_cfm_loss(
    model: VelocityField,
    x1: torch.Tensor,
    sigma_min: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    OT-CFM (minibatch OT approximation via random pairing).

    x0 ~ N(0, I),  x1: real data
    t ~ Uniform[0, 1]
    x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1  (Gaussian path)
    target velocity: u_t = x1 - (1 - sigma_min) * x0

    Loss: ||v_θ(x_t, t) - u_t||²
    """
    B = x1.size(0)
    device = x1.device

    x0 = torch.randn_like(x1)
    t  = torch.rand(B, device=device)

    t_broad = t.view(B, 1, 1, 1, 1)
    x_t   = (1 - (1 - sigma_min) * t_broad) * x0 + t_broad * x1
    u_t   = x1 - (1 - sigma_min) * x0          # target velocity

    v_pred = model(x_t, t)
    loss   = F.mse_loss(v_pred, u_t)

    return loss, {"fm_loss": loss.item()}


# ── 샘플링 (ODE 적분) ───────────────────────────────────────────────────────────

def rollout(
    velocity_field: VelocityField,
    x: torch.Tensor,
    n_steps: int = 32,
    t_start: float = 0.0,
    t_end: float = 1.0,
) -> torch.Tensor:
    """
    Euler rollout of the learned velocity field.

    Unlike `sample()`, this helper keeps autograd intact so it can be used
    inside policy loss terms.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")

    dt = (t_end - t_start) / n_steps
    B = x.size(0)

    for i in range(n_steps):
        t_val = t_start + i * dt
        t = torch.full((B,), t_val, device=x.device, dtype=x.dtype)
        v = velocity_field(x, t)
        x = (x + v * dt).clamp(0.0, 1.0)

    return x

class FlowMatchingWrapper(nn.Module):
    """
    torchdyn NeuralODE와 호환되는 래퍼.
    augmented_dynamics=False, t를 외부에서 주입.
    """

    def __init__(self, velocity_field: VelocityField):
        super().__init__()
        self.vf = velocity_field
        self._t = None  # set before ODE solve

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """torchdyn convention: forward(t_scalar, x_batch)"""
        t_batch = t.expand(x.size(0))
        return self.vf(x, t_batch)


@torch.no_grad()
def sample(
    velocity_field: VelocityField,
    n_samples: int,
    device: torch.device,
    n_steps: int = 100,
) -> torch.Tensor:
    """
    x0 ~ N(0, I) → x1 via Euler integration of v_θ.

    Returns: (n_samples, 1, 48, 48, 48) float32
    """
    velocity_field.eval()
    x = torch.randn(n_samples, 1, 48, 48, 48, device=device)
    return rollout(velocity_field, x, n_steps=n_steps).clamp(0.0, 1.0)


# ── EBM + Flow Matching 결합 인터페이스 ──────────────────────────────────────────

class EBMGuidedFlowMatching(nn.Module):
    """
    Flow Matching 샘플링 시 EBM energy gradient로 가이드.

    샘플링 루프:
      x_{t+dt} = x_t + dt * [v_θ(x_t, t) - λ * ∇_x E_φ(x_t)]

    낮은 energy(= 낮은 annotator disagreement) 방향으로 샘플 유도.
    """

    def __init__(self, velocity_field: VelocityField, ebm: nn.Module):
        super().__init__()
        self.vf  = velocity_field
        self.ebm = ebm

    def sample_guided(
        self,
        n_samples: int,
        device: torch.device,
        guidance_scale: float = 1.0,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """EBM-guided flow matching 샘플링."""
        x = torch.randn(n_samples, 1, 48, 48, 48, device=device)
        dt = 1.0 / n_steps

        self.vf.eval()
        self.ebm.eval()

        for i in range(n_steps):
            t = torch.full((n_samples,), i * dt, device=device)

            # flow velocity
            with torch.no_grad():
                v = self.vf(x, t)

            # EBM gradient (∇_x E): lower energy = more certain nodule
            x_grad = x.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = self.ebm(x_grad).sum()
                grad_e = torch.autograd.grad(energy, x_grad)[0]

            x = (x + dt * (v - guidance_scale * grad_e)).clamp(0.0, 1.0).detach()

        return x
