import torch
import torch.nn as nn


class TriplePendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=512, max_action=0.5):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class TriplePendulumCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = self._build_q_network(input_dim, hidden_dim)
        self.q2 = self._build_q_network(input_dim, hidden_dim)

    @staticmethod
    def _build_q_network(input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.q1(state_action), self.q2(state_action)

    def q1_value(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.q1(state_action)
