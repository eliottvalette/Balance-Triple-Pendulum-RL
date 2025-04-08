config = {
    'num_episodes': 10_000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.999,
    'batch_size': 64,
    'hidden_dim': 256,
    'buffer_capacity': 100000,
    'updates_per_episode': 10,
    'load_models': False,
    'num_nodes': 3,
    'seq_length': 10
}