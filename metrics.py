import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import seaborn as sns
import torch

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_window = 100  # For moving average
        
    def add_metric(self, name, value):
        self.metrics[name].append(value)
    
    def get_moving_average(self, name):
        values = self.metrics[name]
        if len(values) < self.episode_window:
            return np.mean(values)
        return np.mean(values[-self.episode_window:])
    
    def plot_metrics(self, save_path=None):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Metrics')
        
        # Plot rewards
        ax = axes[0, 0]
        rewards = self.metrics['episode_reward']
        ax.plot(rewards, alpha=0.3, label='Episode Reward')
        ax.plot(np.convolve(rewards, np.ones(self.episode_window)/self.episode_window, mode='valid'),
                label=f'Moving Average ({self.episode_window} episodes)')
        ax.set_title('Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # Plot losses
        ax = axes[0, 1]
        ax.plot(self.metrics['actor_loss'], label='Actor Loss')
        ax.plot(self.metrics['critic_loss'], label='Critic Loss')
        ax.set_title('Network Losses')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot reward components
        ax = axes[0, 2]
        reward_components = ['upright_reward', 'x_penalty', 
                           'non_alignement_penalty', 'stability_penalty', 'mse_penalty']
        for component in reward_components:
            if component in self.metrics:
                ax.plot(self.metrics[component], label=component)
        ax.set_title('Reward Components')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot moving averages of reward components
        ax = axes[1, 0]
        for component in reward_components:
            if component in self.metrics:
                moving_avg = np.convolve(self.metrics[component], 
                                       np.ones(self.episode_window)/self.episode_window, 
                                       mode='valid')
                ax.plot(moving_avg, label=f'{component} (MA)')
        ax.set_title('Moving Averages of Reward Components')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot stability metrics
        ax = axes[1, 1]
        if 'stability_penalty' in self.metrics:
            ax.plot(self.metrics['stability_penalty'], label='Stability Penalty')
            moving_avg = np.convolve(self.metrics['stability_penalty'], 
                                   np.ones(self.episode_window)/self.episode_window, 
                                   mode='valid')
            ax.plot(moving_avg, label='Stability Penalty (MA)')
        ax.set_title('Stability Metrics')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot alignment metrics
        ax = axes[1, 2]
        if 'non_alignement_penalty' in self.metrics:
            ax.plot(self.metrics['non_alignement_penalty'], label='Alignment Penalty')
            moving_avg = np.convolve(self.metrics['non_alignement_penalty'], 
                                   np.ones(self.episode_window)/self.episode_window, 
                                   mode='valid')
            ax.plot(moving_avg, label='Alignment Penalty (MA)')
        ax.set_title('Alignment Metrics')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_detailed_rewards(self, save_path=None):
        """Génère un graphique détaillé des récompenses et leurs composantes"""
        plt.figure(figsize=(16, 10))
        
        # Tracer la récompense totale
        plt.subplot(2, 1, 1)
        rewards = self.metrics['episode_reward']
        plt.plot(rewards, alpha=0.3, label='Récompense par épisode')
        
        # Calculer et tracer la moyenne mobile
        if len(rewards) >= self.episode_window:
            moving_avg = np.convolve(rewards, np.ones(self.episode_window)/self.episode_window, mode='valid')
            plt.plot(np.arange(len(moving_avg)) + self.episode_window - 1, moving_avg, 
                    label=f'Moyenne mobile ({self.episode_window} épisodes)')
        
        plt.title('Évolution de la récompense au cours de l\'entraînement')
        plt.xlabel('Épisode')
        plt.ylabel('Récompense')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Tracer les composantes de récompense
        plt.subplot(2, 1, 2)
        reward_components = ['upright_reward', 'x_penalty', 
                             'non_alignement_penalty', 'stability_penalty', 'mse_penalty']
        
        for component in reward_components:
            if component in self.metrics and len(self.metrics[component]) > 0:
                plt.plot(self.metrics[component], label=component)
        
        plt.title('Composantes de la récompense')
        plt.xlabel('Épisode')
        plt.ylabel('Valeur')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_losses(self, save_path=None):
        """Génère un graphique détaillé des pertes du réseau"""
        plt.figure(figsize=(14, 8))
        
        plt.plot(self.metrics['actor_loss'], label='Perte de l\'acteur')
        plt.plot(self.metrics['critic_loss'], label='Perte du critique')
        
        # Calculer et tracer les moyennes mobiles
        if len(self.metrics['actor_loss']) >= self.episode_window:
            actor_ma = np.convolve(self.metrics['actor_loss'], 
                                  np.ones(self.episode_window)/self.episode_window, 
                                  mode='valid')
            plt.plot(np.arange(len(actor_ma)) + self.episode_window - 1, actor_ma, 
                    label=f'Acteur MA ({self.episode_window} épisodes)')
            
        if len(self.metrics['critic_loss']) >= self.episode_window:
            critic_ma = np.convolve(self.metrics['critic_loss'], 
                                   np.ones(self.episode_window)/self.episode_window, 
                                   mode='valid')
            plt.plot(np.arange(len(critic_ma)) + self.episode_window - 1, critic_ma, 
                    label=f'Critique MA ({self.episode_window} épisodes)')
        
        plt.title('Évolution des pertes des réseaux')
        plt.xlabel('Épisode')
        plt.ylabel('Perte')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_reward_distribution(self, save_path=None):
        """Affiche la distribution des récompenses"""
        if len(self.metrics['episode_reward']) < 10:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Distribution complète
        plt.subplot(2, 1, 1)
        sns.histplot(self.metrics['episode_reward'], kde=True)
        plt.title('Distribution des récompenses sur tous les épisodes')
        plt.xlabel('Récompense')
        plt.ylabel('Fréquence')
        
        # Comparer début et fin de l'entraînement
        plt.subplot(2, 1, 2)
        
        num_episodes = len(self.metrics['episode_reward'])
        first_quarter = num_episodes // 4
        last_quarter = max(first_quarter, num_episodes - first_quarter)
        
        # Comparaison des distributions de début et de fin d'entraînement
        if first_quarter > 0:
            sns.kdeplot(self.metrics['episode_reward'][:first_quarter], 
                      label='Premier quart des épisodes')
            sns.kdeplot(self.metrics['episode_reward'][last_quarter:], 
                      label='Dernier quart des épisodes')
            plt.title('Comparaison des distributions de récompenses (début vs fin)')
            plt.xlabel('Récompense')
            plt.ylabel('Densité')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_all_plots(self, base_path='results'):
        """Génère tous les graphiques disponibles dans le dossier spécifié"""
        os.makedirs(base_path, exist_ok=True)
        
        # Graphique principal
        self.plot_metrics(f'{base_path}/metrics.png')
        
        # Graphiques détaillés
        self.plot_detailed_rewards(f'{base_path}/detailed_rewards.png')
        self.plot_losses(f'{base_path}/network_losses.png')
        self.plot_reward_distribution(f'{base_path}/reward_distribution.png')
        
        print(f"Tous les graphiques ont été générés dans le dossier {base_path}")
    
    def plot_model_analysis(self, actor, critic, sample_states, save_path=None):
        """Analyse les réponses des modèles d'acteur et de critique sur des états échantillonnés
        
        Args:
            actor: Le modèle d'acteur
            critic: Le modèle de critique
            sample_states: Un tenseur d'états échantillonnés pour l'analyse
            save_path: Chemin où sauvegarder le graphique
        """
        if not isinstance(sample_states, torch.Tensor):
            return
            
        with torch.no_grad():
            # Générer des actions pour chaque état
            actions = actor(sample_states)
            
            # Obtenir les valeurs Q pour ces paires état-action
            q_values = critic(sample_states, actions)
            
            # Convertir en numpy pour le tracé
            actions_np = actions.cpu().numpy()
            q_values_np = q_values.cpu().numpy()
            
        plt.figure(figsize=(15, 10))
        
        # Distribution des actions
        plt.subplot(2, 1, 1)
        sns.histplot(actions_np, kde=True)
        plt.title('Distribution des actions générées par le modèle d\'acteur')
        plt.xlabel('Action')
        plt.ylabel('Fréquence')
        plt.xlim(-1, 1)

        # Distribution des valeurs Q
        plt.subplot(2, 1, 2)
        sns.histplot(q_values_np, kde=True)
        plt.title('Distribution des valeurs Q générées par le modèle de critique')
        plt.xlabel('Valeur Q')
        plt.ylabel('Fréquence')
        plt.xlim(-1, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show() 