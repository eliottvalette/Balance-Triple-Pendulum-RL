from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F


class PendulumActorPolicy(nn.Module):
    """Stochastic scalar-action policy used by PPO."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        max_action: float = 0.5,
        initial_log_std: float = -1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        if not math.isfinite(max_action) or max_action <= 0.0:
            raise ValueError("max_action must be finite and positive")
        if not math.isfinite(initial_log_std):
            raise ValueError("initial_log_std must be finite")

        self.max_action = float(max_action)
        self.mean_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), float(initial_log_std)))
        self._init_actor_output()

    def _init_actor_output(self) -> None:
        final_linear = self.mean_network[-1]
        if not isinstance(final_linear, nn.Linear):
            raise TypeError("expected final mean layer to be Linear")
        nn.init.orthogonal_(final_linear.weight, gain=0.01)
        nn.init.zeros_(final_linear.bias)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        raw_mean = self.mean_network(state)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(raw_mean)
        return raw_mean, std

    def _distribution(self, state: Tensor) -> Normal:
        mean, std = self(state)
        return Normal(mean, std)

    def _squash(self, raw_action: Tensor) -> Tensor:
        return self.max_action * torch.tanh(raw_action)

    def _inverse_squash(self, action: Tensor) -> Tensor:
        normalized_action = action / self.max_action
        epsilon = torch.finfo(normalized_action.dtype).eps
        # Only stabilise atanh at exact floating-point boundaries. Executed
        # actions are produced by tanh and are never hard-clamped here.
        normalized_action = normalized_action.clamp(-1.0 + epsilon, 1.0 - epsilon)
        return torch.atanh(normalized_action)

    def _squashed_log_prob(self, distribution: Normal, raw_action: Tensor) -> Tensor:
        log_tanh_derivative = 2.0 * (
            math.log(2.0) - raw_action - F.softplus(-2.0 * raw_action)
        )
        log_abs_det_jacobian = math.log(self.max_action) + log_tanh_derivative
        return (distribution.log_prob(raw_action) - log_abs_det_jacobian).sum(dim=-1)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self._distribution(state)
        raw_action = distribution.sample()
        action = self._squash(raw_action)
        log_prob = self._squashed_log_prob(distribution, raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self._distribution(state)
        raw_action = self._inverse_squash(action)
        log_prob = self._squashed_log_prob(distribution, raw_action)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic(self, state: Tensor) -> Tensor:
        raw_mean, _std = self(state)
        return self._squash(raw_mean)


class PendulumValueCritic(nn.Module):
    """State-value function V(s) used for GAE and PPO value regression."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.value_network(state).squeeze(-1)
