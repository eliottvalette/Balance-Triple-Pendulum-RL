import gym
import torch
import numpy as np
from env import TriplePendulumEnv
from model import TriplePendulumActor, TriplePendulumCritic
from reward import RewardManager
from metrics import MetricsTracker
import torch.nn.functional as F
import os

class TriplePendulumTrainer:
    def __init__(self, config):
        self.config = config
        self.env = TriplePendulumEnv(render_mode=None)
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
        
        # Créer le dossier pour sauvegarder les résultats
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def collect_trajectory(self):
        state, _ = self.env.reset()
        done = False
        trajectory = []
        episode_reward = 0
        reward_components = None
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = self.actor(state_tensor).squeeze().numpy()
            
            # Add exploration noise
            action = np.clip(action + np.random.normal(0, 0.1), -1, 1)
            
            # Scale action to environment range
            scaled_action = action * self.env.force_mag
            
            # Take step in environment
            next_state, _, done, _, _ = self.env.step(scaled_action)
            
            # Calculate custom reward and components
            custom_reward = self.reward_manager.calculate_reward(next_state, done)
            reward_components = self.reward_manager.get_reward_components(next_state)
            
            trajectory.append((state, action, custom_reward, next_state, done))
            episode_reward += custom_reward
            state = next_state
            self.total_steps += 1
            
        return trajectory, episode_reward, reward_components
    
    def update_networks(self, trajectory):
        states = torch.FloatTensor([t[0] for t in trajectory])
        actions = torch.FloatTensor([t[1] for t in trajectory])
        rewards = torch.FloatTensor([t[2] for t in trajectory])
        next_states = torch.FloatTensor([t[3] for t in trajectory])
        dones = torch.FloatTensor([t[4] for t in trajectory])
        
        # Update critic
        current_q = self.critic(states, actions)
        with torch.no_grad():
            next_actions = self.actor(next_states)
            target_q = rewards + (1 - dones) * self.config['gamma'] * self.critic(next_states, next_actions)
        
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self):
        for episode in range(self.config['num_episodes']):
            # Collect trajectory and update networks
            trajectory, episode_reward, reward_components = self.collect_trajectory()
            losses = self.update_networks(trajectory)
            
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
                self.metrics.plot_metrics(f'results/metrics_episode_{episode}.png')
                
                # Sauvegarder le modèle
                self.save_models(f"models/checkpoint_{episode}")
    
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
        'hidden_dim': 256
    }
    
    trainer = TriplePendulumTrainer(config)
    trainer.train() 