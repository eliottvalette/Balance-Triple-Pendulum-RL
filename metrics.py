import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import seaborn as sns
import torch
from config import config

class MetricsTracker:
    def __init__(self, plot_config=None):
        self.metrics = defaultdict(list)
        self.episode_window = 100  # For moving average
        
        # Utiliser la configuration des plots si fournie, sinon utiliser config.py
        if plot_config is None and 'plot_config' in config:
            plot_config = config['plot_config']
        else:
            plot_config = plot_config or {}
            
        # Initialiser les paramètres de plotting avec des valeurs par défaut si non définies
        self.max_points_per_plot = plot_config.get('max_points_per_plot', 1000)
        self.plot_dpi = plot_config.get('plot_dpi', 100)
        self.enable_plots = plot_config.get('enable_plots', True)
        
    def add_metric(self, name, value):
        # S'assurer que value est un nombre scalaire
        if isinstance(value, np.ndarray):
            if value.size == 1:
                value = float(value)
            else:
                value = float(value.mean())
        elif isinstance(value, (list, tuple)):
            # Si c'est une liste ou un tuple, prendre la moyenne
            if len(value) > 0:
                value = float(np.mean(value))
            else:
                value = 0.0
                
        self.metrics[name].append(value)
    
    def get_moving_average(self, name):
        values = self.metrics[name]
        if len(values) < self.episode_window:
            return np.mean(values)
        return np.mean(values[-self.episode_window:])
    
    def _downsample_if_needed(self, data):
        """Sous-échantillonne les données si nécessaire pour améliorer les performances"""
        if len(data) <= self.max_points_per_plot:
            return np.array(data), np.arange(len(data))
        
        # Calculer le facteur de sous-échantillonnage
        step = max(1, len(data) // self.max_points_per_plot)
        # Utiliser le début, la fin et les points intermédiaires sous-échantillonnés
        indices = np.concatenate([
            np.arange(0, 100, 1),  # Inclure les premières valeurs intégralement
            np.arange(100, len(data) - 100, step),  # Sous-échantillonner le milieu
            np.arange(max(100, len(data) - 100), len(data), 1)  # Inclure les dernières valeurs
        ])
        indices = np.unique(indices)
        indices = indices[indices < len(data)]
        return np.array(data)[indices], indices
    
    def plot_metrics(self, save_path=None):
        """Génère le graphique principal des métriques d'entraînement"""
        if not self.enable_plots:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Metrics')
        
        # Plot rewards
        ax = axes[0, 0]
        rewards = self.metrics['episode_reward']
        rewards_ds, indices = self._downsample_if_needed(rewards)
        ax.plot(indices, rewards_ds, alpha=0.3, label='Episode Reward')
        
        if len(rewards) >= self.episode_window:
            moving_avg = np.convolve(rewards, np.ones(self.episode_window)/self.episode_window, mode='valid')
            ma_ds, ma_indices = self._downsample_if_needed(moving_avg)
            ax.plot(ma_indices + self.episode_window - 1, ma_ds, 
                   label=f'Moving Average ({self.episode_window} episodes)')
        ax.set_title('Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # Plot losses
        ax = axes[0, 1]
        if 'actor_loss' in self.metrics and len(self.metrics['actor_loss']) > 0:
            actor_loss_ds, actor_indices = self._downsample_if_needed(self.metrics['actor_loss'])
            ax.plot(actor_indices, actor_loss_ds, label='Actor Loss')
        
        if 'critic_loss' in self.metrics and len(self.metrics['critic_loss']) > 0:
            critic_loss_ds, critic_indices = self._downsample_if_needed(self.metrics['critic_loss'])
            ax.plot(critic_indices, critic_loss_ds, label='Critic Loss')
        ax.set_title('Network Losses')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot reward components
        ax = axes[0, 2]
        reward_components = ['reward', 'upright_reward', 'x_penalty','non_alignement_penalty', 'stability_penalty', 'mse_penalty', 'heraticness_penalty']
        for component in reward_components:
            if component in self.metrics and len(self.metrics[component]) > 0:
                comp_ds, comp_indices = self._downsample_if_needed(self.metrics[component])
                ax.plot(comp_indices, comp_ds, label=component)
        ax.set_title('Reward Components')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot moving averages of reward components
        ax = axes[1, 0]
        for component in reward_components:
            if component in self.metrics and len(self.metrics[component]) >= self.episode_window:
                moving_avg = np.convolve(self.metrics[component], 
                                       np.ones(self.episode_window)/self.episode_window, 
                                       mode='valid')
                ma_ds, ma_indices = self._downsample_if_needed(moving_avg)
                ax.plot(ma_indices + self.episode_window - 1, ma_ds, label=f'{component} (MA)')
        ax.set_title('Moving Averages of Reward Components')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot stability metrics
        ax = axes[1, 1]
        if 'stability_penalty' in self.metrics and len(self.metrics['stability_penalty']) > 0:
            stab_ds, stab_indices = self._downsample_if_needed(self.metrics['stability_penalty'])
            ax.plot(stab_indices, stab_ds, label='Stability Penalty')
            
            if len(self.metrics['stability_penalty']) >= self.episode_window:
                moving_avg = np.convolve(self.metrics['stability_penalty'], 
                                      np.ones(self.episode_window)/self.episode_window, 
                                      mode='valid')
                ma_ds, ma_indices = self._downsample_if_needed(moving_avg)
                ax.plot(ma_indices + self.episode_window - 1, ma_ds, label='Stability Penalty (MA)')
        ax.set_title('Stability Metrics')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Plot alignment metrics
        ax = axes[1, 2]
        if 'non_alignement_penalty' in self.metrics and len(self.metrics['non_alignement_penalty']) > 0:
            align_ds, align_indices = self._downsample_if_needed(self.metrics['non_alignement_penalty'])
            ax.plot(align_indices, align_ds, label='Alignment Penalty')
            
            if len(self.metrics['non_alignement_penalty']) >= self.episode_window:
                moving_avg = np.convolve(self.metrics['non_alignement_penalty'], 
                                      np.ones(self.episode_window)/self.episode_window, 
                                      mode='valid')
                ma_ds, ma_indices = self._downsample_if_needed(moving_avg)
                ax.plot(ma_indices + self.episode_window - 1, ma_ds, label='Alignment Penalty (MA)')
        ax.set_title('Alignment Metrics')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi)
        plt.close()
    
    def plot_detailed_rewards(self, save_path=None):
        """Génère un graphique détaillé des récompenses et leurs composantes"""
        if not self.enable_plots:
            return
            
        plt.figure(figsize=(16, 10))
        
        # Tracer la récompense totale
        plt.subplot(2, 1, 1)
        rewards = self.metrics['episode_reward']
        rewards_ds, indices = self._downsample_if_needed(rewards)
        plt.plot(indices, rewards_ds, alpha=0.3, label='Récompense par épisode')
        
        # Calculer et tracer la moyenne mobile
        if len(rewards) >= self.episode_window:
            moving_avg = np.convolve(rewards, np.ones(self.episode_window)/self.episode_window, mode='valid')
            ma_ds, ma_indices = self._downsample_if_needed(moving_avg)
            plt.plot(ma_indices + self.episode_window - 1, ma_ds,
                    label=f'Moyenne mobile ({self.episode_window} épisodes)')
        
        plt.title('Évolution de la récompense au cours de l\'entraînement')
        plt.xlabel('Épisode')
        plt.ylabel('Récompense')
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        
        # Tracer les composantes de récompense
        plt.subplot(2, 1, 2)
        reward_components = ['upright_reward', 'x_penalty', 
                             'non_alignement_penalty', 'stability_penalty', 'mse_penalty']
        
        for component in reward_components:
            if component in self.metrics and len(self.metrics[component]) > 0:
                comp_ds, comp_indices = self._downsample_if_needed(self.metrics[component])
                plt.plot(comp_indices, comp_ds, label=component)
        
        plt.title('Composantes de la récompense')
        plt.xlabel('Épisode')
        plt.ylabel('Valeur')
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi)
            plt.close()
        else:
            plt.show()

    def plot_losses(self, save_path=None):
        """Génère un graphique détaillé des pertes du réseau"""
        if not self.enable_plots:
            return
            
        plt.figure(figsize=(14, 8))
        
        if 'actor_loss' in self.metrics and len(self.metrics['actor_loss']) > 0:
            actor_loss_ds, actor_indices = self._downsample_if_needed(self.metrics['actor_loss'])
            plt.plot(actor_indices, actor_loss_ds, label='Perte de l\'acteur')
        
        if 'critic_loss' in self.metrics and len(self.metrics['critic_loss']) > 0:
            critic_loss_ds, critic_indices = self._downsample_if_needed(self.metrics['critic_loss'])
            plt.plot(critic_indices, critic_loss_ds, label='Perte du critique')
        
        # Calculer et tracer les moyennes mobiles
        if 'actor_loss' in self.metrics and len(self.metrics['actor_loss']) >= self.episode_window:
            actor_ma = np.convolve(self.metrics['actor_loss'], 
                                  np.ones(self.episode_window)/self.episode_window, 
                                  mode='valid')
            ma_ds, ma_indices = self._downsample_if_needed(actor_ma)
            plt.plot(ma_indices + self.episode_window - 1, ma_ds,
                    label=f'Acteur MA ({self.episode_window} épisodes)')
            
        if 'critic_loss' in self.metrics and len(self.metrics['critic_loss']) >= self.episode_window:
            critic_ma = np.convolve(self.metrics['critic_loss'], 
                                   np.ones(self.episode_window)/self.episode_window, 
                                   mode='valid')
            ma_ds, ma_indices = self._downsample_if_needed(critic_ma)
            plt.plot(ma_indices + self.episode_window - 1, ma_ds,
                    label=f'Critique MA ({self.episode_window} épisodes)')
        
        plt.title('Évolution des pertes des réseaux')
        plt.xlabel('Épisode')
        plt.ylabel('Perte')
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi)
            plt.close()
        else:
            plt.show()
    
    def plot_reward_distribution(self, save_path=None):
        """Affiche la distribution des récompenses"""
        if not self.enable_plots or len(self.metrics['episode_reward']) < 10:
            return
            
        plt.figure(figsize=(12, 8))
        
        # Distribution complète - limiter à max_points pour accélérer
        plt.subplot(2, 1, 1)
        rewards = self.metrics['episode_reward']
        if len(rewards) > self.max_points_per_plot:
            sample_indices = np.linspace(0, len(rewards)-1, self.max_points_per_plot, dtype=int)
            sampled_rewards = [rewards[i] for i in sample_indices]
            sns.histplot(sampled_rewards, kde=True)
        else:
            sns.histplot(rewards, kde=True)
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
            # Sous-échantillonnage pour les grands ensembles de données
            if first_quarter > self.max_points_per_plot // 2:
                sample_indices = np.linspace(0, first_quarter-1, self.max_points_per_plot // 2, dtype=int)
                first_data = [self.metrics['episode_reward'][i] for i in sample_indices]
            else:
                first_data = self.metrics['episode_reward'][:first_quarter]
                
            if (num_episodes - last_quarter) > self.max_points_per_plot // 2:
                sample_indices = np.linspace(last_quarter, num_episodes-1, self.max_points_per_plot // 2, dtype=int)
                last_data = [self.metrics['episode_reward'][i] for i in sample_indices]
            else:
                last_data = self.metrics['episode_reward'][last_quarter:]
                
            sns.kdeplot(first_data, label='Premier quart des épisodes')
            sns.kdeplot(last_data, label='Dernier quart des épisodes')
            plt.title('Comparaison des distributions de récompenses (début vs fin)')
            plt.xlabel('Récompense')
            plt.ylabel('Densité')
            plt.legend(loc="upper right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi)
            plt.close()
        else:
            plt.show()
    
    def generate_all_plots(self, base_path='results'):
        """Génère tous les graphiques disponibles dans le dossier spécifié"""
        if not self.enable_plots:
            return
            
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
        if not self.enable_plots or not isinstance(sample_states, torch.Tensor):
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
        plt.xlim(max(-1, min(actions_np) - 0.2), min(max(actions_np) + 0.2, 1))

        # Distribution des valeurs Q
        plt.subplot(2, 1, 2)
        sns.histplot(q_values_np, kde=True)
        plt.title('Distribution des valeurs Q générées par le modèle de critique')
        plt.xlabel('Valeur Q')
        plt.ylabel('Fréquence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi)
            plt.close()
        else:
            plt.show() 