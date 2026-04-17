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
from contextlib import contextmanager
from typing import Iterable


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
    policy_sample_steps: int = 32        # detached FM rollout for reward update
    policy_grad_steps: int = 8           # differentiable FM rollout for policy loss
    reward_weight: float = 0.1

    # Regularization
    l2_reg: float = 1.0               # equilibrium |E| = 1/(2*l2_reg): 0.01→±50, 0.1→±5, 0.5→±1
    energy_clamp: float | None = None  # hard clamp after l2_reg tuning; None = off
    grad_clip: float = 1.0

    # Reward-weighted CD (옵션 A)
    # reward_cd_weight > 0이면 uniform CD 대신 softmax(reward/T) 가중 CD 사용
    # T가 작을수록 high-reward 샘플에 집중; T→∞이면 uniform CD와 동일
    reward_cd_weight: float = 0.0     # 0 = off (기존 uniform CD), 1 = full weighted CD
    reward_cd_temp: float = 1.0       # softmax temperature for reward weighting

    # Gated hybrid negative strategy
    # Phase 1 (warm-up): SGLD-only negatives until FM earns its place
    # Phase 2 (hybrid): FM negatives mixed in, SGLD permanently kept at sgld_permanent_ratio
    sgld_permanent_ratio: float = 0.2      # SGLD 비율 — warm-up 후에도 끝까지 유지
    fm_gate_sep_std_threshold: float = 10.0  # sep/std(EMA)가 이 값 아래로 내려와야 FM 투입
    fm_gate_consecutive: int = 3           # 연속 N회 통과해야 전환 (noise 방지)
    fm_gate_check_interval: int = 50       # N reward steps마다 gate 체크
    sep_std_ema_alpha: float = 0.1         # sep/std EMA 스무딩 계수


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

    DDP 지원: ebm/velocity_field에 DDP 래퍼를 그대로 넘겨도 됨.
    내부에서 _raw_ebm/_raw_vf (원본 모듈)를 통해 SGLD 수행,
    DDP 래퍼를 통한 forward로 gradient allreduce 보장.
    """

    def __init__(
        self,
        ebm: nn.Module,
        velocity_field: nn.Module,
        config: IRLConfig,
        device: torch.device,
    ):
        self.ebm  = ebm            # DDP 래퍼 또는 raw 모듈
        self.vf   = velocity_field
        self.cfg  = config
        self.device = device

        # SGLD 및 grad_norm clip용 raw 모듈 (DDP.module 또는 그대로)
        self._raw_ebm = getattr(ebm, "module", ebm)
        self._raw_vf  = getattr(velocity_field, "module", velocity_field)

        self.reward_opt = torch.optim.Adam(self._raw_ebm.parameters(), lr=config.reward_lr)
        self.fm_opt     = torch.optim.Adam(self._raw_vf.parameters(),  lr=config.fm_lr)

        # Replay buffer for SGLD
        self.replay = ReplayBuffer(
            max_size=config.replay_buffer_size,
            shape=(1, 48, 48, 48),
        )

        # Gated hybrid negative state
        self._reward_step_count: int = 0       # 누적 reward step 수
        self._gate_pass_count: int   = 0       # 연속 gate 통과 횟수
        self._fm_enabled: bool       = False   # FM negative 투입 여부
        self._sep_std_ema: float     = float("inf")  # sep/std EMA (초기값=무한대)

    def _policy_sample(self, batch_size: int, detach: bool, n_steps: int) -> torch.Tensor:
        """
        Roll out the current FM policy from Gaussian noise.
        """
        from src.models.flow_matching import rollout

        x0 = torch.randn(batch_size, 1, 48, 48, 48, device=self.device)
        vf = self._raw_vf if detach else self.vf

        if detach:
            with torch.no_grad():
                return rollout(vf, x0, n_steps=n_steps).detach()
        return rollout(vf, x0, n_steps=n_steps)

    @contextmanager
    def _freeze_ebm_params(self):
        """
        Disable EBM parameter grads while still allowing gradients to flow
        from the reward scalar back to the policy samples.
        """
        prev = [p.requires_grad for p in self._raw_ebm.parameters()]
        try:
            for p in self._raw_ebm.parameters():
                p.requires_grad_(False)
            yield
        finally:
            for p, flag in zip(self._raw_ebm.parameters(), prev):
                p.requires_grad_(flag)

    @staticmethod
    def _grad_norm(
        loss: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
    ) -> float:
        """
        Compute gradient norm without consuming the graph so individual loss
        terms can be inspected before the actual backward pass.
        """
        params = [p for p in params if p.requires_grad]
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        sq_norm = torch.zeros((), device=loss.device)
        for grad in grads:
            if grad is not None:
                sq_norm = sq_norm + grad.pow(2).sum()
        return sq_norm.sqrt().item()

    # ── negative 샘플링 (gated hybrid) ───────────────────────────────────────

    def _sample_negatives(self, x_demo: torch.Tensor) -> tuple[torch.Tensor, str]:
        """
        Gated hybrid negative strategy:

        Phase 1 (warm-up, _fm_enabled=False):
          SGLD-only — real sample perturbation, hard negative 보장

        Phase 2 (hybrid, _fm_enabled=True):
          sgld_permanent_ratio 만큼 SGLD, 나머지 FM policy sample 혼합
          SGLD는 끝까지 유지해 hard negative baseline을 보존

        Returns:
            x_neg: negative samples
            source: 'sgld' | 'hybrid' (로깅용)
        """
        B = x_demo.size(0)

        if not self._fm_enabled:
            # Phase 1: SGLD only
            x_init = self.replay.sample(B, self.cfg.replay_prob, self.device)
            x_neg  = self._raw_ebm.sample_langevin(
                x_init,
                n_steps     = self.cfg.sgld_steps,
                step_size   = self.cfg.sgld_step_size,
                noise_scale = self.cfg.sgld_noise_scale,
            )
            self.replay.push(x_neg)
            return x_neg, "sgld"

        # Phase 2: SGLD + FM hybrid
        n_sgld = max(1, int(B * self.cfg.sgld_permanent_ratio))
        n_fm   = B - n_sgld

        x_init = self.replay.sample(n_sgld, self.cfg.replay_prob, self.device)
        x_sgld = self._raw_ebm.sample_langevin(
            x_init,
            n_steps     = self.cfg.sgld_steps,
            step_size   = self.cfg.sgld_step_size,
            noise_scale = self.cfg.sgld_noise_scale,
        )
        self.replay.push(x_sgld)

        x_fm  = self._policy_sample(n_fm, detach=True, n_steps=self.cfg.policy_sample_steps)
        x_neg = torch.cat([x_sgld, x_fm], dim=0)
        return x_neg, "hybrid"

    def _check_fm_gate(self, e_pos: torch.Tensor, e_neg_fm: torch.Tensor) -> bool:
        """
        FM negative 투입 자격 gate.

        sep/std EMA가 threshold 아래로 fm_gate_consecutive회 연속 통과하면 전환.
        배치 noise 방지를 위해 EMA 스무딩 후 fm_gate_check_interval마다만 체크.

        Returns True if gate just opened (transition event).
        """
        if self._fm_enabled:
            return False  # 이미 열림

        # sep/std 계산 (현재 배치 기준)
        sep     = (e_pos.mean() - e_neg_fm.mean()).abs().item()
        avg_std = (e_pos.std(unbiased=False).item() + e_neg_fm.std(unbiased=False).item()) / 2 + 1e-3
        sep_std = sep / avg_std

        # EMA 업데이트
        alpha = self.cfg.sep_std_ema_alpha
        if self._sep_std_ema == float("inf"):
            self._sep_std_ema = sep_std
        else:
            self._sep_std_ema = alpha * sep_std + (1 - alpha) * self._sep_std_ema

        # interval마다 체크
        if self._reward_step_count % self.cfg.fm_gate_check_interval != 0:
            return False

        if self._sep_std_ema < self.cfg.fm_gate_sep_std_threshold:
            self._gate_pass_count += 1
        else:
            self._gate_pass_count = 0  # 연속 실패 시 리셋

        if self._gate_pass_count >= self.cfg.fm_gate_consecutive:
            self._fm_enabled = True
            return True  # transition event

        return False

    # ── EBM reward 업데이트 ────────────────────────────────────────────────────

    def update_reward(self, x_demo: torch.Tensor, reward: torch.Tensor | None = None) -> dict:
        """
        MaxEnt IRL reward gradient:
          ∇_φ L = E_demo[∇_φ E_φ(x)] - E_model[∇_φ E_φ(x̃)]

        EBM loss (uniform):         mean(E(x_demo)) - mean(E(x_neg)) + λ||E||²
        EBM loss (reward-weighted): (w · E(x_demo)).sum() - mean(E(x_neg)) + λ||E||²
          where w = softmax(reward / T),  reward = -Var(malignancy)

        Negative strategy: gated hybrid (SGLD warm-up → SGLD+FM after gate)
        """
        total_metrics = {
            "reward_loss": 0.0,
            "cd_loss": 0.0,
            "reg_loss": 0.0,
            "e_pos": 0.0,
            "e_neg": 0.0,
            "e_pos_std": 0.0,
            "e_neg_std": 0.0,
            "reward_grad_norm": 0.0,
            "fm_sample_energy": 0.0,
            "sep_std_ema": 0.0,
            "fm_enabled": 0.0,   # 0 or 1 — phase 추적용
        }

        for _ in range(self.cfg.reward_steps_per_iter):
            self._reward_step_count += 1

            # negative 샘플링 (gated hybrid)
            x_neg, neg_source = self._sample_negatives(x_demo)

            # gate 체크 (SGLD phase일 때 FM quality를 별도로 측정)
            gate_opened = False
            fm_energy_for_log = float("nan")
            if not self._fm_enabled:
                x_fm_probe = self._policy_sample(
                    min(4, x_demo.size(0)),
                    detach=True,
                    n_steps=self.cfg.policy_sample_steps,
                )
                with torch.no_grad():
                    e_pos_probe = self.ebm(x_demo[:x_fm_probe.size(0)])
                    e_fm_probe  = self.ebm(x_fm_probe)
                fm_energy_for_log = e_fm_probe.mean().item()
                gate_opened = self._check_fm_gate(e_pos_probe, e_fm_probe)
                if gate_opened:
                    # 방금 전환됨 — 이번 step부터 hybrid 적용
                    x_neg, neg_source = self._sample_negatives(x_demo)

            self.reward_opt.zero_grad()
            e_pos = self.ebm(x_demo)
            e_neg = self.ebm(x_neg)

            if self.cfg.energy_clamp is not None:
                e_pos = e_pos.clamp(-self.cfg.energy_clamp, self.cfg.energy_clamp)
                e_neg = e_neg.clamp(-self.cfg.energy_clamp, self.cfg.energy_clamp)

            # CD loss: uniform 또는 reward-weighted
            if self.cfg.reward_cd_weight > 0.0 and reward is not None:
                w = torch.softmax(reward / self.cfg.reward_cd_temp, dim=0)
                cd_pos  = (w * e_pos).sum()
                cd_loss = (1.0 - self.cfg.reward_cd_weight) * e_pos.mean() \
                        + self.cfg.reward_cd_weight * cd_pos \
                        - e_neg.mean()
            else:
                cd_loss = (e_pos - e_neg).mean()

            reg_loss = self.cfg.l2_reg * (e_pos ** 2 + e_neg ** 2).mean()
            loss     = cd_loss + reg_loss
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._raw_ebm.parameters(), self.cfg.grad_clip)
            self.reward_opt.step()

            step_metrics = {
                "reward_loss":   loss.item(),
                "cd_loss":       cd_loss.item(),
                "reg_loss":      reg_loss.item(),
                "e_pos":         e_pos.mean().item(),
                "e_neg":         e_neg.mean().item(),
                "e_pos_std":     e_pos.std(unbiased=False).item(),
                "e_neg_std":     e_neg.std(unbiased=False).item(),
                "reward_grad_norm": float(grad_norm),
                "fm_sample_energy": fm_energy_for_log,
                "sep_std_ema":   self._sep_std_ema if self._sep_std_ema != float("inf") else -1.0,
                "fm_enabled":    float(self._fm_enabled),
            }
            for k in total_metrics:
                total_metrics[k] += step_metrics[k] / self.cfg.reward_steps_per_iter

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

        total_metrics = {
            "fm_loss": 0.0,
            "policy_energy": 0.0,
            "policy_grad_norm": 0.0,
            "policy_term_grad_norm": 0.0,
            "reward_term_grad_norm": 0.0,
        }

        for _ in range(self.cfg.fm_steps_per_iter):
            self.fm_opt.zero_grad()

            # OT-CFM flow matching loss — DDP 래퍼 통해 gradient sync
            fm_loss, _ = ot_cfm_loss(self.vf, x_demo)

            # Reward must be attached to actual policy samples so the velocity
            # field receives a usable gradient signal.
            x_policy = self._policy_sample(
                x_demo.size(0), detach=False, n_steps=self.cfg.policy_grad_steps
            )
            with self._freeze_ebm_params():
                reward_loss = self._raw_ebm(x_policy).mean()

            policy_term_grad_norm = self._grad_norm(fm_loss, self._raw_vf.parameters())
            reward_term_grad_norm = self._grad_norm(
                self.cfg.reward_weight * reward_loss,
                self._raw_vf.parameters(),
            )
            loss = fm_loss + self.cfg.reward_weight * reward_loss
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self._raw_vf.parameters(), self.cfg.grad_clip)
            self.fm_opt.step()

            step_metrics = {
                "fm_loss": fm_loss.item(),
                "policy_energy": reward_loss.item(),
                "policy_grad_norm": float(grad_norm),
                "policy_term_grad_norm": policy_term_grad_norm,
                "reward_term_grad_norm": reward_term_grad_norm,
            }
            for k in total_metrics:
                total_metrics[k] += step_metrics[k] / self.cfg.fm_steps_per_iter

        return total_metrics

    # ── IRL 루프 1 iteration ────────────────────────────────────────────────────

    def step(self, x_demo: torch.Tensor, reward: torch.Tensor | None = None) -> dict:
        """
        IRL 1 iteration = reward update + policy update.

        Args:
            x_demo: (B, 1, 48, 48, 48) demonstration patches
            reward: (B,) per-sample reward, e.g. -Var(malignancy). None = uniform CD.

        Returns:
            metrics dict
        """
        reward_metrics = self.update_reward(x_demo, reward=reward)
        fm_metrics     = self.update_policy(x_demo)
        return {**reward_metrics, **fm_metrics}
