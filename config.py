# config.py
config = {
    'num_episodes': 10_000,
    'max_steps': 1000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'hidden_dim': 256,
    'buffer_capacity': 10_000,
    'load_models': True,
    'num_nodes': 2,
    'gravity': 0.81,
    'friction_coefficient': 0.1,
    'max_action': 0.5,
    'exploration_noise': 0.10,
    'policy_noise': 0.08,
    'noise_clip': 0.15,
    'policy_delay': 2,
    'polyak_tau': 0.005,
    'learning_starts': 1_000,
    'train_every_steps': 4,
    'updates_per_train': 1,
    'initial_angle_noise': 0.1122,
    'initial_velocity_noise': 0.04,
    'down_to_up_episode_probability': 0.25,
    'down_angle_noise': 0.1122,
    'post_switch_reward_weight': 2.0,
    'transition_improvement_weight': 0.5,
    'post_switch_low_score_threshold': 0.2,
    'post_switch_low_score_penalty': 0.5,
    'render_training': True,
    'render_every_episodes': 20,
    'render_first_episode': True,
    'debug': False,
    
    # Options de visualisation et de plots
    'plot_config': {
        'enable_plots': True,           # Activer/désactiver tous les graphiques
        'plot_frequency': 100,          # Fréquence de génération des graphiques principaux
        'full_plot_frequency': 1_000,    # Fréquence de génération des graphiques complets
        'max_points_per_plot': 1_000,    # Nombre maximal de points par graphique
        'plot_dpi': 100                 # Résolution des graphiques (DPI)
    }
}
