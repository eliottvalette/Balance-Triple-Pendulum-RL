import torch
import torch.nn as nn
import torch.nn.functional as F
import random as rd

class TriplePendulumActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(TriplePendulumActor, self).__init__()
        
        # Add layer normalization for input
        self.input_norm = nn.LayerNorm(state_dim)
        
        # First layer
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Third layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc3.weight, gain=1.414)
        nn.init.constant_(self.fc3.bias, 0)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Fourth layer
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc4.weight, gain=1.414)
        nn.init.constant_(self.fc4.bias, 0)
        self.norm4 = nn.LayerNorm(hidden_dim)
        
        # Fifth layer
        self.fc5 = nn.Linear(hidden_dim + state_dim, action_dim * 4)
        nn.init.orthogonal_(self.fc5.weight, gain=1.414)
        nn.init.constant_(self.fc5.bias, 0)
        self.norm5 = nn.LayerNorm(action_dim * 4)
        
        # Output layer
        self.fc_out = nn.Linear(action_dim * 4, action_dim)
        nn.init.orthogonal_(self.fc_out.weight, gain=0.01)
        nn.init.constant_(self.fc_out.bias, 0)
        
    def forward(self, x):
        # Ensure input has correct shape
        if len(x.shape) > 2:
            x = x.squeeze(-1)
            
        saved_input = x

        # Normalize input
        x = self.input_norm(x)
        
        # First layer
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)
        
        # Fourth layer
        x = self.fc4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)
        
        # Fifth layer (with skip connection)
        # Ensure both tensors have the same number of dimensions
        if len(x.shape) != len(saved_input.shape):
            if len(x.shape) > len(saved_input.shape):
                saved_input = saved_input.unsqueeze(-1)
            else:
                x = x.unsqueeze(-1)
        
        x = torch.cat([x, saved_input], dim=-1)
        x = self.fc5(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)
        
        # Output layer
        x = self.fc_out(x)
        return torch.tanh(x)

class TriplePendulumCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(TriplePendulumCritic, self).__init__()
        
        # Add layer normalization for input
        self.input_norm = nn.LayerNorm(state_dim + action_dim)
        
        # First layer
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        nn.init.orthogonal_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.orthogonal_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Third layer
        self.fc3 = nn.Linear(hidden_dim, action_dim * 32)
        nn.init.orthogonal_(self.fc3.weight, gain=1.414)
        nn.init.constant_(self.fc3.bias, 0)
        self.norm3 = nn.LayerNorm(action_dim * 32)

        # Fourth layer
        self.fc4 = nn.Linear(action_dim * 32 + state_dim, action_dim * 8)
        nn.init.orthogonal_(self.fc4.weight, gain=1.414)
        nn.init.constant_(self.fc4.bias, 0)
        self.norm4 = nn.LayerNorm(action_dim * 8)
        
        # Output layer
        self.fc_out = nn.Linear(action_dim * 8, 1)
        nn.init.orthogonal_(self.fc_out.weight, gain=1)
        nn.init.constant_(self.fc_out.bias, 0)
                
    def forward(self, state, action):
        # Ensure inputs have correct shape
        if len(state.shape) > 2:
            state = state.squeeze(-1)
        if len(action.shape) > 2:
            action = action.squeeze(-1)
        
        saved_input = state
        x = torch.cat([state, action], dim=1)
        
        # Normalize input
        x = self.input_norm(x)
        
        # First layer
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        # Third layer
        x = self.fc3(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        # Fourth layer (with skip connection)
        x = torch.cat([x, saved_input], dim=1)
        x = self.fc4(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)
        
        # Output layer
        x = self.fc_out(x)
        return x 