config = {
    'num_episodes': 10_000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.99,
    'batch_size': 64,
    'hidden_dim': 512,
    'buffer_capacity': 100000,
    'updates_per_episode': 10,
    'load_models': True,
    'num_nodes': 2,
    'seq_length': 1,
    'gravity': 0.81,
    
    # Options de visualisation et de plots
    'plot_config': {
        'enable_plots': True,           # Activer/désactiver tous les graphiques
        'plot_frequency': 500,          # Fréquence de génération des graphiques principaux
        'full_plot_frequency': 1000,    # Fréquence de génération des graphiques complets
        'max_points_per_plot': 1000,    # Nombre maximal de points par graphique
        'plot_dpi': 100                 # Résolution des graphiques (DPI)
    }
}