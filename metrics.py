import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_window = 100  # Pour la moyenne glissante
        
    def add_metric(self, name, value):
        self.metrics[name].append(value)
    
    def get_moving_average(self, name):
        values = self.metrics[name]
        if len(values) < self.episode_window:
            return np.mean(values)
        return np.mean(values[-self.episode_window:])
    
    def plot_metrics(self, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics')
        
        # Plot des récompenses
        ax = axes[0, 0]
        rewards = self.metrics['episode_reward']
        ax.plot(rewards, alpha=0.3, label='Episode Reward')
        ax.plot(np.convolve(rewards, np.ones(self.episode_window)/self.episode_window, mode='valid'),
                label=f'Moving Average ({self.episode_window} episodes)')
        ax.set_title('Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        
        # Plot des pertes Actor
        ax = axes[0, 1]
        ax.plot(self.metrics['actor_loss'], label='Actor Loss')
        ax.set_title('Actor Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot des pertes Critic
        ax = axes[1, 0]
        ax.plot(self.metrics['critic_loss'], label='Critic Loss')
        ax.set_title('Critic Loss')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.legend()
        
        # Plot des composantes de récompense
        ax = axes[1, 1]
        for component in ['upright_reward', 'cart_penalty', 'velocity_penalty']:
            if component in self.metrics:
                ax.plot(self.metrics[component], label=component)
        ax.set_title('Reward Components')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close() 