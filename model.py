import torch
import torch.nn as nn


class PendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=512, max_action=0.5):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class PendulumCritic(nn.Module):
    def __init__(self, state_dim, action_dim=1, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.critic_head_a = self._build_value_network(input_dim, hidden_dim)
        self.critic_head_b = self._build_value_network(input_dim, hidden_dim)

    @staticmethod
    def _build_value_network(input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.critic_head_a(state_action), self.critic_head_b(state_action)

    def primary_value(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        return self.critic_head_a(state_action)
