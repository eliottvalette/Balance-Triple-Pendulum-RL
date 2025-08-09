# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class TriplePendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(TriplePendulumActor, self).__init__()
        
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.seq_1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

    def forward(self, state):
        """
        Args
        ----
        state : (batch, state_dim)  — entrée normalisée

        Returns
        -------
        action :   (batch, 8)        ∈ {-1,0,1}
        probs :    (batch, 8, 3)     pour debug éventuel
        """
        x_1 = self.seq_1(state)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)
        action_logits  = self.action_head(x_3)                          # (B, 24)

        # soft‑max par artic. (dim=-1)
        probs   = F.softmax(action_logits, dim=-1)

        return probs

class TriplePendulumCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(TriplePendulumCritic, self).__init__()
        
        super().__init__()

        self.seq_1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.seq_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.seq_3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )

        self.V_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_tensor):
        """
        Here V(s) estimates the value of the state, it's an estimation of how much the situation is favorable (in terms of future expected rewards)
        """
        x_1 = self.seq_1(input_tensor)
        x_2 = self.seq_2(x_1)
        x_3 = self.seq_3(x_2)
        
        V = self.V_head(x_3)

        return V