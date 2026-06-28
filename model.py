from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal


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
        mean = self.mean_network(state).clamp(-self.max_action, self.max_action)
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return mean, std

    def _distribution(self, state: Tensor) -> Normal:
        mean, std = self(state)
        return Normal(mean, std)

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self._distribution(state)
        action = distribution.sample().clamp(-self.max_action, self.max_action)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        distribution = self._distribution(state)
        bounded_action = action.clamp(-self.max_action, self.max_action)
        log_prob = distribution.log_prob(bounded_action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic(self, state: Tensor) -> Tensor:
        mean, _std = self(state)
        return mean


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
