import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class TriplePendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TriplePendulumActor, self).__init__()
        
        # Add layer normalization for input
        self.input_norm = nn.LayerNorm(state_dim)
        
        # Initialize with orthogonal weights and zero bias for first layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        
        # Layer normalization after first layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Initialize with orthogonal weights and zero bias for second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        
        # Layer normalization after second layer
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Initialize with orthogonal weights and zero bias for output layer
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)  # Smaller gain for output layer
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, x):
        # Normalize input
        x = self.input_norm(x)
        
        # First layer with normalization
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # Second layer with normalization
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        
        # Output layer with tanh
        x = self.fc3(x)
        return torch.tanh(x)  # Output between -1 and 1

class TriplePendulumCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TriplePendulumCritic, self).__init__()
        
        # Add layer normalization for input
        self.input_norm = nn.LayerNorm(state_dim + action_dim)
        
        # Initialize with orthogonal weights and zero bias for first layer
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        
        # Layer normalization after first layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Initialize with orthogonal weights and zero bias for second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        
        # Layer normalization after second layer
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Initialize with orthogonal weights and zero bias for output layer
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.fc3.weight, gain=1)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        
        # Normalize input
        x = self.input_norm(x)
        
        # First layer with normalization
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        
        # Second layer with normalization
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        
        # Output layer
        x = self.fc3(x)
        return x 