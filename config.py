# config.py
config = {
    'num_episodes': 10_000,
    'max_steps': 1000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'hidden_dim': 512,
    'buffer_capacity': 10_000,
    'load_models': False,
    'num_nodes': 2,
    'gravity': 9.81,
    'friction_coefficient': 0.1,
    'max_action': 0.5,
    'exploration_noise': 0.10,
    'policy_noise': 0.08,
    'noise_clip': 0.15,
    'policy_delay': 2,
    'polyak_tau': 0.005,
    'updates_per_step': 1,
    'initial_angle_noise': 0.1122,
    'initial_velocity_noise': 0.04,
    'render_training': False,
    'render_every_episodes': 200,
    'render_first_episode': True,
    'debug': False,
    
    # Options de visualisation et de plots
    'plot_config': {
        'enable_plots': True,           # Activer/désactiver tous les graphiques
        'plot_frequency': 200,          # Fréquence de génération des graphiques principaux
        'full_plot_frequency': 1_000,    # Fréquence de génération des graphiques complets
        'max_points_per_plot': 1_000,    # Nombre maximal de points par graphique
        'plot_dpi': 100                 # Résolution des graphiques (DPI)
    }
}
