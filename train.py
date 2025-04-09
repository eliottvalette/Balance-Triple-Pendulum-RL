import gym
import torch
import pygame
import numpy as np
from tp_env import TriplePendulumEnv  # Assurez-vous que le chemin d'import correspond à la nouvelle version
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

class TriplePendulumTrainer:
    def __init__(self, config):
        self.config = config
        self.reward_manager = RewardManager()
        self.env = TriplePendulumEnv(reward_manager=self.reward_manager, render_mode="human", num_nodes=config['num_nodes'])
        self.force_mag = config.get('force_mag', 10.0)
        
        # Dimension d'un état unique (construit à partir de reset() ou get_state())
        state_size = len(self.env.get_state())
        # L'état d'entrée au modèle est la concaténation d'une séquence d'états
        state_dim = state_size * config['seq_length']
        action_dim = 1
        self.actor = TriplePendulumActor(state_dim, action_dim)
        self.critic = TriplePendulumCritic(state_dim, action_dim)
        self.seq_state = []  # Pour stocker une séquence d'états
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'])
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        self.total_steps = 0
        self.max_steps = 500  # Maximum steps par épisode
        
        # Exploration parameters
        self.epsilon = 1.0         # Probabilité d'action aléatoire initiale
        self.epsilon_decay = 0.995 # Décroissance d'epsilon
        self.min_epsilon = 0.001     # Valeur minimale d'epsilon
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=config['buffer_capacity'])
        
        # Reward normalization
        self.reward_scale = 1.0
        self.reward_running_mean = 0
        self.reward_running_std = 1
        self.reward_alpha = 0.001  # Pour la mise à jour des statistiques
        
        # Création des dossiers pour sauvegarder les résultats
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Initialisation du rendu si nécessaire
        self.env._render_init()
        
        # Chargement des modèles si demandé
        if config['load_models']:
            self.load_models()
    
    def normalize_reward(self, reward):
        # Mise à jour des statistiques sur les rewards
        self.reward_running_mean = (1 - self.reward_alpha) * self.reward_running_mean + self.reward_alpha * reward
        self.reward_running_std = (1 - self.reward_alpha) * self.reward_running_std + self.reward_alpha * abs(reward - self.reward_running_mean)
        std = max(self.reward_running_std, 1e-6)
        normalized_reward = (reward - self.reward_running_mean) / std
        return normalized_reward * self.reward_scale

    def collect_trajectory(self, episode):
        # Dans la nouvelle version, reset() renvoie directement l'état initial
        state = self.env.reset()
        rich_state = self.env.get_state()  # On peut ici considérer get_state() comme le "rich state"
        done = False
        trajectory = []
        episode_reward = 0
        reward_components = None
        num_steps = 0
        
        # Réinitialiser le RewardManager
        self.reward_manager.reset()
        
        # Initialiser la séquence d'états avec l'état initial répété pour atteindre la longueur requise
        self.seq_state = []
        state_tensor = torch.FloatTensor(rich_state).unsqueeze(0)
        for _ in range(self.config['seq_length']):
            self.seq_state.append(state_tensor)
        
        while not done and num_steps < self.max_steps:
            # Construire la séquence d'états courante (concatenation horizontale)
            state_tensor = torch.FloatTensor(rich_state).unsqueeze(0)
            # Mise à jour de la séquence : retirer le plus ancien et ajouter l'état courant
            self.seq_state.pop(0)
            self.seq_state.append(state_tensor)
            
            # Exploration epsilon-greedy
            if np.random.random() < self.epsilon:
                action = np.random.uniform(-1, 1)  # Action aléatoire
            else:
                with torch.no_grad():
                    seq_state_tensor = torch.cat(self.seq_state, dim=1).squeeze(0)
                    action = self.actor(seq_state_tensor).squeeze().numpy()
            
            # Mise à l'échelle de l'action : dans la nouvelle version, on travaille avec la force directement
            scaled_action = action * self.force_mag
            
            # --- Mise à jour manuelle de la simulation ---
            # Calculer la dérivée d'état via rhs, puis intégrer par Euler
            dx = self.env.rhs(rich_state, self.env.current_time, self.env.parameter_vals, lambda x: scaled_action)
            next_state = rich_state + self.env.dt * dx
            self.env.current_time += self.env.dt
            self.env.current_state = next_state
            
            # Condition terminale (exemple : position du chariot ou angles trop importants)
            # Le nouvel état est structuré comme : [position, angle(s), vitesse(s)]
            done = (abs(next_state[0]) > 2.4) or (np.any(np.abs(next_state[1:self.env.n+1]) > np.pi))
            next_rich_state = next_state
            
            # Si le rendu est activé, il est possible d'appeler animate_pendulum ou un rendu simplifié ici
            if self.env.render_mode == "human":
                # Vous pouvez appeler ici une méthode de rendu légère si besoin.
                pass
            
            # Calculer la reward via le RewardManager
            custom_reward, _, _, _, _, _, force_terminated = self.reward_manager.calculate_reward(next_rich_state, done, num_steps)
            reward_components = self.reward_manager.get_reward_components(next_rich_state, num_steps)
            normalized_reward = self.normalize_reward(custom_reward)
            
            # Préparer l'état séquentiel suivant
            next_state_tensor = torch.FloatTensor(next_rich_state).unsqueeze(0)
            # Si la séquence n'est pas encore pleine (devrait être remplie au reset), on l'actualise
            next_seq = self.seq_state[1:] + [next_state_tensor]
            if len(next_seq) > self.config['seq_length']:
                next_seq = next_seq[-self.config['seq_length']:]
            next_seq_state = torch.cat(next_seq, dim=1).squeeze(0)
            
            # Concaténer l'état de la séquence actuel (déjà construit) et le stocker dans le buffer
            current_seq_state = torch.cat(self.seq_state, dim=1).squeeze(0)
            self.memory.push(current_seq_state.numpy(), action, normalized_reward, next_seq_state.numpy(), done)
            
            trajectory.append((current_seq_state, action, custom_reward, next_seq_state, done))
            episode_reward += custom_reward
            rich_state = next_rich_state
            self.total_steps += 1
            num_steps += 1

            if done or force_terminated:
                break
            
        # Décroissance de l'exploration
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return trajectory, episode_reward, reward_components
    
    def update_networks(self):
        if len(self.memory) < self.config['batch_size']:
            return {"critic_loss": 0, "actor_loss": 0}  # Pas assez d'exemples
        
        # Échantillonnage du batch depuis le replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        # Conversion en tenseurs
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Ici, les états sont déjà des vecteurs concaténés de taille state_dim * seq_length
        states_reshaped = states  # forme : [batch_size, state_dim * seq_length]
        next_states_reshaped = next_states
        
        # Mise à jour du critic
        current_q = self.critic(states_reshaped, actions)
        with torch.no_grad():
            next_actions = self.actor(next_states_reshaped)
            target_q = rewards + (1 - dones) * self.config['gamma'] * self.critic(next_states_reshaped, next_actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Mise à jour de l'actor (maximiser la Q-valeur estimée)
        actor_loss = -self.critic(states_reshaped, self.actor(states_reshaped)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}
    
    def train(self):
        for episode in range(self.config['num_episodes']):
            print(f"Episode {episode} started")
            
            # Pour le rendu, on peut éventuellement ajuster render_mode ou la cadence
            if episode % 100 == 0:
                self.env.render_mode = "human"
            else:
                self.env.render_mode = None
                
            trajectory, episode_reward, reward_components = self.collect_trajectory(episode)
            
            # Mettre à jour les réseaux seulement si le buffer est suffisamment rempli
            losses = {"critic_loss": 0, "actor_loss": 0}
            if len(self.memory) >= self.config['batch_size']:
                for _ in range(self.config['updates_per_episode']):
                    update_losses = self.update_networks()
                    losses['critic_loss'] += update_losses['critic_loss']
                    losses['actor_loss'] += update_losses['actor_loss']
                losses['critic_loss'] /= self.config['updates_per_episode']
                losses['actor_loss'] /= self.config['updates_per_episode']
            
            # Suivi des métriques
            self.metrics.add_metric('episode_reward', episode_reward)
            self.metrics.add_metric('actor_loss', losses['actor_loss'])
            self.metrics.add_metric('critic_loss', losses['critic_loss'])
            for component_name, value in reward_components.items():
                self.metrics.add_metric(component_name, value)
            
            if episode % 100 == 99:
                avg_reward = self.metrics.get_moving_average('episode_reward')
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
                self.metrics.plot_metrics(f'results/metrics.png')
                self.save_models(f"models/checkpoint")
    
    def save_models(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic.state_dict(), path + '_critic.pth')
        torch.save(self.actor_optimizer.state_dict(), path + '_actor_optimizer.pth')
        torch.save(self.critic_optimizer.state_dict(), path + '_critic_optimizer.pth')
    
    def load_models(self):
        if os.path.exists('models/checkpoint_actor.pth'):
            print("Loading models")
            self.actor.load_state_dict(torch.load('models/check
