import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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