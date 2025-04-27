config = {
    'num_episodes': 10_000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.99,
    'batch_size': 16,
    'hidden_dim': 512,
    'buffer_capacity': 100_000,
    'updates_per_episode': 10,
    'load_models': False,
    'num_nodes': 2,
    'gravity': 9.81,
    'friction_coefficient': 0.1,
    
    # Options de visualisation et de plots
    'plot_config': {
        'enable_plots': True,           # Activer/désactiver tous les graphiques
        'plot_frequency': 500,          # Fréquence de génération des graphiques principaux
        'full_plot_frequency': 1_000,    # Fréquence de génération des graphiques complets
        'max_points_per_plot': 1_000,    # Nombre maximal de points par graphique
        'plot_dpi': 100                 # Résolution des graphiques (DPI)
    }
}