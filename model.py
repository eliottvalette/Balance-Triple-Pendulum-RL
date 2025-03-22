import torch
import torch.nn as nn
import torch.nn.functional as F

class TriplePendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TriplePendulumActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output between -1 and 1

class TriplePendulumCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TriplePendulumCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) 