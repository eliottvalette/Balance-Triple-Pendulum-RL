# train.py
import gym
import torch
import pygame
import numpy as np
import random as rd
from tp_env import TriplePendulumEnv
from model import TriplePendulumActor, TriplePendulumCritic
from reward import RewardManager
from metrics import MetricsTracker
import torch.nn.functional as F
import os
import random
from collections import deque
from config import config

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

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, dt=0.01):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        self.state = np.copy(self.mu)
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) * self.dt + \
             self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class TriplePendulumTrainer:
    def __init__(self, config):
        self.config = config
        self.reward_manager = RewardManager()
        self.env = TriplePendulumEnv(reward_manager=self.reward_manager, render_mode="human", num_nodes=config['num_nodes'], max_steps=1000)  # Enable rendering from the start
        
        # Initialize models
        # Ajustement de la dimension d'état en fonction de l'environnement réel
        # Ici, la dimension est basée sur la taille de l'état retourné par reset()
        self.env.reset(phase = 1)
        self.old_state = self.env.get_state(action = 0, phase = 1)
        initial_state = self.env.get_state(action = 0, phase = 1)
        state_dim = len(initial_state) * 2
        action_dim = 2
        self.actor_model = TriplePendulumActor(state_dim, action_dim, config['hidden_dim'])
        self.critic_model = TriplePendulumCritic(state_dim, config['hidden_dim'])
        self.critic_target = TriplePendulumCritic(state_dim, config['hidden_dim'])
        self.num_exploration_episodes = 0
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=config['critic_lr'])
        
        # Metrics tracking avec la configuration des plots
        plot_config = config.get('plot_config', {})
        self.metrics = MetricsTracker(plot_config)
        self.plot_frequency = plot_config.get('plot_frequency', 500)
        self.full_plot_frequency = plot_config.get('full_plot_frequency', 1000)
        self.previous_phase_cumulated_rewards = {1: 0, -1: 0}
        self.phase_cumulated_rewards = {1: [], -1: []}
        self.phase_counts = {1: 0, -1: 0}
        
        self.total_steps = 0
        self.max_steps = 1_000  # Maximum steps per episode
        
        # Exploration parameters
        self.epsilon = 1.0  # Initial random action probability
        self.epsilon_decay = 0.9985  # Epsilon decay rate
        self.min_epsilon = 0.001  # Minimum epsilon
        self.ou_noise = OrnsteinUhlenbeckNoise(action_dim=1)
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config['buffer_capacity'])
        
        # Reward normalization
        self.reward_scale = 1.0
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.001  # For running statistics update
        self.polyak_tau = 0.995
        
        # Create directories for saving results
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Initialize rendering si la méthode existe
        if hasattr(self.env, '_render_init'):
            self.env._render_init()
        
        # Load models
        if config['load_models']:
            self.load_models()
    
    def normalize_reward(self, reward):
        # Update running statistics
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * abs(reward - self.reward_running_mean)
        
        # Avoid division by zero
        std = max(self.reward_running_std, 1e-6)
        
        # Normalize and scale
        normalized_reward = (reward - self.reward_running_mean) / std
        return normalized_reward * self.reward_scale

    def collect_trajectory(self, episode):# Réinitialiser le RewardManager et le bruit
        self.reward_manager.reset()
        self.ou_noise.reset()

        # Episode phase
        """
        phase_keys = [-1, 1]
        phase_rewards = np.array([np.mean(self.previous_phase_cumulated_rewards[k]) for k in phase_keys])
        # Use negative rewards to favor the phase with the lowest reward
        softmax_phase_probabilities = np.exp(-phase_rewards) / np.sum(np.exp(-phase_rewards))
        softmax_phase_probabilities = np.clip(softmax_phase_probabilities, 0.1, 0.9)
        print('softmax_phase_probabilities', softmax_phase_probabilities)
        phase = np.random.choice(phase_keys, p=softmax_phase_probabilities)  # Higher prob for lowest reward phase
        """
        phase = 1 # TODO: Remove this when model is ok on basic mode

        # Reset before collecting trajectory
        self.env.reset(phase = -phase)
        done = False
        trajectory = []
        episode_reward = 0
        reward_components_accumulated = {}
        num_steps = 0
        
        # Variables pour l'exploration dirigée
        last_action = 0.0
        action_history = []
        
        while not done and num_steps < self.max_steps:
            if num_steps == self.max_steps // 2:
                phase = -phase

            current_state = self.env.get_state(action = last_action, phase = phase)
            old_and_current_state = np.concatenate((self.old_state, current_state))
            old_and_current_state_tensor = torch.FloatTensor(old_and_current_state)
            
            # Exploration: combinaison de bruit OU et exploration dirigée
            with torch.no_grad():
                action_probs = self.actor_model(old_and_current_state_tensor).squeeze().numpy()

            action_side = np.argmax(action_probs)
            action = - (action_side * 2 - 1)
            
            if rd.random() < self.epsilon: # Add Noise  
                action = action * (1 - self.epsilon) + (rd.random() * 2 - 1) * self.epsilon

            # Limiter l'action
            action_history.append(action)
            
            # Take step in environment
            next_state, terminated = self.env.step(action = action, phase = phase)
            
            if np.isnan(np.sum(next_state)):
                print('state:', next_state)
                raise ValueError("Warning: NaN detected in state")
            
            # Render if rendering is enabled
            if self.env.render_mode == "human":
                rendering_successful = self.env.render(action = action, episode = episode, epsilon = self.epsilon, current_step = num_steps, phase = phase)
                if not rendering_successful:
                    done = True
                    raise ValueError("Warning: Rendering failed")
            
            # Calculate custom reward and components
            custom_reward, reward_components, force_terminated = self.reward_manager.calculate_reward(next_state, terminated, num_steps, action, phase)
            
            # Store transition in replay buffer with normalized reward
            current_and_next_state = np.concatenate((current_state, next_state))
            current_and_next_state_tensor = torch.FloatTensor(current_and_next_state)  # Convert to tensor

            # Stocker dans le buffer de replay
            self.memory.push(old_and_current_state_tensor, action, custom_reward, current_and_next_state_tensor, terminated)
            
            trajectory.append((old_and_current_state_tensor, action, custom_reward, current_and_next_state_tensor, terminated))
            episode_reward += custom_reward
            self.total_steps += 1
            num_steps += 1
            self.old_state = current_state

            for component_name, value in reward_components.items():
                if component_name not in reward_components_accumulated:
                    reward_components_accumulated[component_name] = []
                reward_components_accumulated[component_name].append(value)

            if terminated or force_terminated:
                done = True
                break
        
        # Decay exploration parameters
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"Episode {episode} ended with {num_steps} steps")

        # Update the cumulated rewards
        self.phase_cumulated_rewards[phase].append(episode_reward / 2000)
        self.phase_counts[phase] += 1
        return trajectory, episode_reward, reward_components_accumulated
    
    def update_networks(self):
        if len(self.memory) < self.config['batch_size']:
            return {"critic_loss": 0, "actor_loss": 0}  # Not enough samples yet

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(-1)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Critic forward
        state_values = self.critic_model(states_tensor)
        next_state_values = self.critic_target(next_states_tensor).detach()

        # TD target et advantage
        td_targets = rewards_tensor + self.config['gamma'] * next_state_values * (1 - dones_tensor)
        advantages = td_targets - state_values.detach()

        # Critic loss
        critic_loss = F.mse_loss(state_values, td_targets)

        # Actor loss
        probs = self.actor_model(states_tensor)                      # (B, 8, 3)
        max_probs = torch.max(probs, axis=1)
        log_max_prob = torch.log(max_probs.values)

        actor_loss = -(advantages * log_max_prob).mean()

        # Optim Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Optim Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Polyak update
        with torch.no_grad():
            for param, target_param in zip(self.critic_model.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(self.polyak_tau)
                target_param.data.add_((1 - self.polyak_tau) * param.data)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def train(self):
        for episode in range(self.config['num_episodes']):
            print(f"Episode {episode} started")
            
            # Activer ou désactiver le rendu en fonction du numéro d'épisode
            if episode % 50 == 0 and episode > self.num_exploration_episodes:
                print(f"Resetting phase cumulated rewards -1: {np.mean(self.phase_cumulated_rewards[-1])}, 1: {np.mean(self.phase_cumulated_rewards[1])}")
                print(f"Phase counts", self.phase_counts)
                self.previous_phase_cumulated_rewards = self.phase_cumulated_rewards
                self.phase_cumulated_rewards = {1: [], -1: []}
                self.env.render_mode = "human"
            else:
                self.env.render_mode = None
                
            # Collect trajectory and store in replay buffer
            trajectory, episode_reward, reward_components_accumulated = self.collect_trajectory(episode)
            
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
            
            # Track reward components - s'assurer que ce sont bien des valeurs scalaires
            for component_name, values in reward_components_accumulated.items():
                # Convertir en tableau numpy si ce n'est pas déjà le cas
                values = np.array(values)
                
                # Calculer la moyenne pour obtenir une seule valeur scalaire
                if values.size > 0:
                    if values.ndim > 1:  # Si le tableau a plus d'une dimension
                        values = values.flatten()
                    avg_value = float(np.mean(values))
                else:
                    avg_value = 0.0
                
                # Ajouter la valeur scalaire à la métrique
                self.metrics.add_metric(component_name, avg_value)
            
            # Génération des graphiques selon la fréquence configurée
            if episode % self.plot_frequency == self.plot_frequency - 1:
                avg_reward = self.metrics.get_moving_average('episode_reward')
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                
                # Générer seulement le graphique principal
                os.makedirs('results', exist_ok=True)
                self.metrics.plot_metrics('results/metrics.png')
                
                # Génération complète des graphiques à une fréquence moins élevée
                if episode % self.full_plot_frequency == self.full_plot_frequency - 1:
                    # Analyse du modèle
                    if len(self.memory) >= 100:
                        samples = self.memory.sample(100)
                        sample_states = torch.FloatTensor(samples[0])
                        self.metrics.plot_model_analysis(
                            self.actor_model, 
                            self.critic_model, 
                            sample_states,
                            f'results/model_analysis.png'
                        )
                    
                    # Génération de tous les graphiques
                    self.metrics.generate_all_plots()
                
                # Save model
                self.save_models(f"models/checkpoint")
    
    def save_models(self, path):
        torch.save(self.actor_model.state_dict(), path + '_actor.pth')
        torch.save(self.critic_model.state_dict(), path + '_critic.pth')
        torch.save(self.actor_optimizer.state_dict(), path + '_actor_optimizer.pth')
        torch.save(self.critic_optimizer.state_dict(), path + '_critic_optimizer.pth')
        
        # Sauvegarde supplémentaire avec numéro d'épisode pour suivre l'évolution
        episode_num = len(self.metrics.metrics['episode_reward'])
        if episode_num > 0 and episode_num % 1000 == 0:  # Sauvegarde tous les 1000 épisodes
            checkpoint_path = f"models/checkpoint"
            torch.save(self.actor_model.state_dict(), checkpoint_path + '_actor.pth')
            torch.save(self.critic_model.state_dict(), checkpoint_path + '_critic.pth')

    def load_models(self):
        if os.path.exists('models/checkpoint_actor.pth'):
            print("Loading models")
            self.actor_model.load_state_dict(torch.load('models/checkpoint_actor.pth', weights_only=True))
            self.critic_model.load_state_dict(torch.load('models/checkpoint_critic.pth', weights_only=True))
            self.actor_optimizer.load_state_dict(torch.load('models/checkpoint_actor_optimizer.pth', weights_only=True))
            self.critic_optimizer.load_state_dict(torch.load('models/checkpoint_critic_optimizer.pth', weights_only=True))

if __name__ == "__main__":    
    trainer = TriplePendulumTrainer(config)
    trainer.train() 