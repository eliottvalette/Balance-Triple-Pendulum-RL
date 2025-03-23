import gym
import torch
import numpy as np
from env import TriplePendulumEnv
from model import TriplePendulumActor, TriplePendulumCritic
from reward import RewardManager
from metrics import MetricsTracker
import torch.nn.functional as F
import os
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class TriplePendulumTrainer:
    def __init__(self, config):
        self.config = config
        self.env = TriplePendulumEnv(render_mode=None)  # Disable rendering during training
        self.reward_manager = RewardManager()
        
        # Initialize models
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.actor = TriplePendulumActor(state_dim, action_dim)
        self.critic = TriplePendulumCritic(state_dim, action_dim)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        self.total_steps = 0
        self.max_steps = 300  # Maximum steps per episode
        
        # Exploration parameters
        self.noise_scale = 0.5  # Initial noise scale
        self.noise_decay = 0.9999  # Decay rate
        self.min_noise = 0.01  # Minimum noise
        self.epsilon = 1.0  # Initial random action probability
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.min_epsilon = 0.1  # Minimum epsilon
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config['buffer_capacity'])
        
        # Reward normalization
        self.reward_scale = 1.0
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.001  # For running statistics update
        
        # Créer le dossier pour sauvegarder les résultats
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def normalize_reward(self, reward):
        # Update running statistics
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * abs(reward - self.reward_running_mean)
        
        # Avoid division by zero
        std = max(self.reward_running_std, 1e-6)
        
        # Normalize and scale
        normalized_reward = (reward - self.reward_running_mean) / std
        return normalized_reward * self.reward_scale

    def collect_trajectory(self):
        state, _ = self.env.reset()
        done = False
        trajectory = []
        episode_reward = 0
        reward_components = None
        num_steps = 0
        
        while not done and num_steps < self.max_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Epsilon-greedy exploration
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-1, 1)  # Random action
            else:
                with torch.no_grad():
                    action = self.actor(state_tensor).squeeze().numpy()
                    # Add exploration noise with decay
                    noise = np.random.normal(0, self.noise_scale)
                    action = np.clip(action + noise, -1, 1)
            
            # Scale action to environment range and ensure it's an array
            scaled_action = np.array([action * self.env.force_mag])
            
            # Take step in environment
            next_state, terminated = self.env.step(scaled_action)
            
            # Calculate custom reward and components
            custom_reward, upright_reward, x_penalty, x_dot_penalty, non_alignement_penalty, acceleration_penalty, energy_penalty = self.reward_manager.calculate_reward(next_state, terminated)
            reward_components = self.reward_manager.get_reward_components(next_state)
            
            # Normalize reward
            normalized_reward = self.normalize_reward(custom_reward)
            
            # Store transition in replay buffer with normalized reward
            self.memory.push(state, action, normalized_reward, next_state, terminated)
            
            trajectory.append((state, action, custom_reward, next_state, terminated))
            episode_reward += custom_reward
            state = next_state
            self.total_steps += 1
            num_steps += 1
            
        # Decay exploration parameters
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
        return trajectory, episode_reward, reward_components
    
    def update_networks(self):
        if len(self.memory) < self.config['batch_size']:
            return {"critic_loss": 0, "actor_loss": 0}  # Not enough samples yet

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Update critic
        current_q = self.critic(states, actions)
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q = rewards + (1 - dones) * self.config['gamma'] * self.critic(next_states, next_actions)
        
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Add gradient clipping for critic
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Update actor - we want to maximize the critic output (Q-value)
        # Since optimizers perform minimization by default, we use negative sign to turn maximization into minimization
        # This is correct for DDPG: we want the actor to choose actions that maximize Q-values
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Add gradient clipping for actor
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self):
        for episode in range(self.config['num_episodes']):
            print(f"Episode {episode} started")
            # Collect trajectory and store in replay buffer
            trajectory, episode_reward, reward_components = self.collect_trajectory()
            
            # Only update after we have enough samples
            losses = {"critic_loss": 0, "actor_loss": 0}
            if len(self.memory) >= self.config['batch_size']:
                # Perform multiple updates per episode
                for _ in range(self.config['updates_per_episode']):
                    update_losses = self.update_networks()
                    # Accumulate losses for reporting
                    losses['critic_loss'] += update_losses['critic_loss']
                    losses['actor_loss'] += update_losses['actor_loss']
                
                # Average the losses
                losses['critic_loss'] /= self.config['updates_per_episode']
                losses['actor_loss'] /= self.config['updates_per_episode']
            
            # Track metrics
            self.metrics.add_metric('episode_reward', episode_reward)
            self.metrics.add_metric('actor_loss', losses['actor_loss'])
            self.metrics.add_metric('critic_loss', losses['critic_loss'])
            
            # Track reward components
            for component_name, value in reward_components.items():
                self.metrics.add_metric(component_name, value)
            
            # Affichage périodique et sauvegarde
            if episode % 100 == 0:
                avg_reward = self.metrics.get_moving_average('episode_reward')
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                
                # Sauvegarder les graphiques
                self.metrics.plot_metrics(f'results/metrics.png')
                
                # Sauvegarder le modèle
                self.save_models(f"models/checkpoint_{episode}")
                
                # Render one episode at 30 FPS
                self.env.render_mode = "human"
                state, _ = self.env.reset()
                done = False
                num_steps = 0
                while not done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        action = self.actor(state_tensor).squeeze().numpy()
                    scaled_action = np.array([action * self.env.force_mag])
                    state, terminated = self.env.step(scaled_action)
                    self.env.render()
                    self.env.clock.tick(30)  # Limit to 30 FPS
                    num_steps += 1
                    if num_steps >= self.max_steps or terminated:
                        done = True
                self.env.render_mode = None  # Turn off rendering
    
    def save_models(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, path)

if __name__ == "__main__":
    config = {
        'num_episodes': 10000,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'gamma': 0.99,
        'batch_size': 64,
        'hidden_dim': 256,
        'buffer_capacity': 100000,
        'updates_per_episode': 10
    }
    
    trainer = TriplePendulumTrainer(config)
    trainer.train() 