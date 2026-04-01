"""
Maximum Entropy IRL reward update loop.
- Demonstration: real nodule patches (reward = -Var(malignancy))
- MaxEnt IRL: reward_θ(x) ≈ EBM energy E_φ(x) (sign flipped)
- Reward gradient: ∇_θ L = E_demo[∇_θ r] - E_model[∇_θ r]
  → In EBM terms: gradient of energy w.r.t. φ
  → demo: decrease E on real data, model: increase E on generated samples
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable


@dataclass
class IRLConfig:
    # Reward (EBM) update
    reward_lr: float = 1e-4
    reward_steps_per_iter: int = 5       # gradient steps on reward per IRL iter

    # SGLD (negative sample generation for EBM)
    sgld_steps: int = 60
    sgld_step_size: float = 10.0
    sgld_noise_scale: float = 0.005
    replay_buffer_size: int = 10000
    replay_prob: float = 0.95

    # Flow Matching update
    fm_lr: float = 1e-4
    fm_steps_per_iter: int = 10          # gradient steps on FM per IRL iter

    # Regularization
    l2_reg: float = 1.0
    grad_clip: float = 1.0


class ReplayBuffer:
    """
    SGLD replay buffer for EBM training stability.
    Randomly returns stored or fresh noise samples.
    """

    def __init__(self, max_size: int, shape: tuple):
        self.max_size = max_size
        self.shape    = shape
        self.buffer   = torch.FloatTensor(max_size, *shape).uniform_(0, 1)
        self.size     = 0

    def sample(self, batch_size: int, replay_prob: float, device: torch.device) -> torch.Tensor:
        """Returns (batch_size, *shape) init samples for SGLD."""
        n_replay = int(batch_size * replay_prob)
        n_fresh  = batch_size - n_replay

        idx = torch.randint(0, max(self.size, 1), (n_replay,))
        replay = self.buffer[idx].to(device)
        fresh  = torch.rand(n_fresh, *self.shape, device=device)
        return torch.cat([replay, fresh], dim=0)

    def push(self, samples: torch.Tensor):
        """Add SGLD results back into the buffer."""
        samples = samples.detach().cpu()
        n = samples.size(0)
        idx = torch.randint(0, self.max_size, (n,))
        self.buffer[idx] = samples
        self.size = min(self.size + n, self.max_size)


class MaxEntIRL:
    """
    MaxEnt IRL training loop.

    reward_fn  = EBM (E_φ): lower energy = higher reward
    policy     = Flow Matching (VelocityField): generates samples

    IRL iteration:
      1. 데모(실제 결절) vs 모델(FM 샘플) 대비로 EBM reward 업데이트
      2. 업데이트된 reward를 이용해 FM velocity field 업데이트
    """

    def __init__(
        self,
        ebm: nn.Module,
        velocity_field: nn.Module,
        config: IRLConfig,
        device: torch.device,
    ):
        self.ebm   = ebm.to(device)
        self.vf    = velocity_field.to(device)
        self.cfg   = config
        self.device = device

        self.reward_opt = torch.optim.Adam(ebm.parameters(), lr=config.reward_lr)
        self.fm_opt     = torch.optim.Adam(velocity_field.parameters(), lr=config.fm_lr)

        # Replay buffer for SGLD
        self.replay = ReplayBuffer(
            max_size=config.replay_buffer_size,
            shape=(1, 48, 48, 48),
        )

    # ── EBM reward 업데이트 ────────────────────────────────────────────────────

    def update_reward(self, x_demo: torch.Tensor) -> dict:
        """
        MaxEnt IRL reward gradient:
          ∇_φ L = E_demo[∇_φ E_φ(x)] - E_model[∇_φ E_φ(x̃)]

        EBM loss: E(x_demo) - E(x_neg)  + λ||E||²
        """
        from src.models.ebm import contrastive_divergence_loss

        total_metrics = {"reward_loss": 0.0, "e_pos": 0.0, "e_neg": 0.0}

        for _ in range(self.cfg.reward_steps_per_iter):
            # SGLD negative samples
            x_init = self.replay.sample(x_demo.size(0), self.cfg.replay_prob, self.device)
            x_neg  = self.ebm.sample_langevin(
                x_init,
                n_steps    = self.cfg.sgld_steps,
                step_size  = self.cfg.sgld_step_size,
                noise_scale= self.cfg.sgld_noise_scale,
            )
            self.replay.push(x_neg)

            self.reward_opt.zero_grad()
            loss, metrics = contrastive_divergence_loss(
                self.ebm, x_demo, x_neg, l2_reg=self.cfg.l2_reg
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.ebm.parameters(), self.cfg.grad_clip)
            self.reward_opt.step()

            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0.0) / self.cfg.reward_steps_per_iter

        return total_metrics

    # ── FM policy 업데이트 ──────────────────────────────────────────────────────

    def update_policy(self, x_demo: torch.Tensor) -> dict:
        """
        Flow Matching 업데이트:
          - OT-CFM loss: 기본 생성 목표
          - IRL reward shaping: EBM energy를 낮추는 방향으로 추가 gradient
            (reward = -E_φ(x_1), x_1 = FM 샘플)
        """
        from src.models.flow_matching import ot_cfm_loss

        total_fm_loss = 0.0

        for _ in range(self.cfg.fm_steps_per_iter):
            self.fm_opt.zero_grad()

            # OT-CFM flow matching loss
            fm_loss, _ = ot_cfm_loss(self.vf, x_demo)

            # IRL reward shaping: 생성 샘플의 EBM energy 최소화
            # x_1 근사: t=0.99에서 FM 샘플 (빠른 근사)
            B = x_demo.size(0)
            x0 = torch.randn_like(x_demo)
            t  = torch.full((B,), 0.99, device=self.device)
            x_approx = (1 - t.view(B, 1, 1, 1, 1)) * x0 + t.view(B, 1, 1, 1, 1) * x_demo
            reward_loss = self.ebm(x_approx).mean()  # minimize energy → maximize reward

            loss = fm_loss + 0.1 * reward_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.vf.parameters(), self.cfg.grad_clip)
            self.fm_opt.step()

            total_fm_loss += fm_loss.item() / self.cfg.fm_steps_per_iter

        return {"fm_loss": total_fm_loss}

    # ── IRL 루프 1 iteration ────────────────────────────────────────────────────

    def step(self, x_demo: torch.Tensor) -> dict:
        """
        IRL 1 iteration = reward update + policy update.

        Args:
            x_demo: (B, 1, 48, 48, 48) demonstration patches

        Returns:
            metrics dict
        """
        reward_metrics = self.update_reward(x_demo)
        fm_metrics     = self.update_policy(x_demo)
        return {**reward_metrics, **fm_metrics}
