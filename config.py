# config.py
import math


EPISODE_MODES = ("down_to_up", "capture_vertical", "fold_to_up", "up_to_fold")


config = {
    'num_episodes': 10_000,
    'max_steps': 1000,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'gamma': 0.995,
    'batch_size': 64,
    'hidden_dim': 256,
    'buffer_capacity': 10_000,
    'load_models': True,
    'num_nodes': 2,
    'gravity': 9.81,
    'friction_coefficient': 0.1,
    'max_action': 0.5,
    'exploration_noise': 0.10,
    'swing_up_exploration_noise': 0.05,
    'swing_up_exploration_amplitude': 0.45,
    'swing_up_exploration_period_min': 60,
    'swing_up_exploration_period_max': 120,
    'swing_up_capture_score_threshold': 0.75,
    'swing_up_capture_noise': 0.03,
    'policy_noise': 0.08,
    'noise_clip': 0.15,
    'policy_delay': 2,
    'polyak_tau': 0.005,
    'learning_starts': 1_000,
    'train_every_steps': 4,
    'updates_per_train': 1,
    'initial_angle_noise': 0.1122,
    'initial_velocity_noise': 0.04,
    'episode_mode_probabilities': {
        'down_to_up': 0.9,
        'capture_vertical': 0.1,
        'fold_to_up': 0.0,
        'up_to_fold': 0.0,
    },
    'adaptive_curriculum_enabled': False,
    'curriculum_start_episode': 300,
    'curriculum_window': 200,
    'curriculum_min_probabilities': {
        'down_to_up': 0.20,
        'capture_vertical': 0.20,
        'fold_to_up': 0.10,
        'up_to_fold': 0.05,
    },
    'curriculum_max_probabilities': {
        'down_to_up': 0.50,
        'capture_vertical': 0.45,
        'fold_to_up': 0.35,
        'up_to_fold': 0.20,
    },
    'transition_switch_step_min': 50,
    'transition_switch_step_max': 150,
    'capture_angle_noise': 0.18,
    'capture_cart_velocity_noise': 0.05,
    'capture_angular_velocity_noise': 2.0,
    'down_angle_noise': 0.1122,
    'cart_failure_penalty': -50.0,
    'success_bonus': 100.0,
    'capture_entry_bonus': 5.0,
    'swing_up_energy_progress_weight': 6.0,
    'swing_up_height_progress_weight': 2.0,
    'swing_up_cart_safety_weight': 1.0,
    'capture_allowed_angular_speed': 1.5,
    'hold_progress_bonus': 4.0,
    'hold_required_steps': 120,
    'render_training': True,
    'render_every_episodes': 50,
    'render_first_episode': True,
    # Options de visualisation et de plots
    'plot_config': {
        'enable_plots': True,           # Activer/désactiver tous les graphiques
        'plot_frequency': 100,          # Fréquence de génération des graphiques principaux
        'max_points_per_plot': 1_000,    # Nombre maximal de points par graphique
        'plot_dpi': 100                 # Résolution des graphiques (DPI)
    }
}


def validate_config(cfg):
    required_keys = {
        'num_episodes', 'max_steps', 'actor_lr', 'critic_lr', 'gamma',
        'batch_size', 'hidden_dim', 'buffer_capacity', 'load_models',
        'num_nodes', 'gravity', 'friction_coefficient', 'max_action',
        'exploration_noise', 'swing_up_exploration_noise',
        'swing_up_exploration_amplitude', 'swing_up_exploration_period_min',
        'swing_up_exploration_period_max', 'swing_up_capture_score_threshold',
        'swing_up_capture_noise', 'policy_noise', 'noise_clip',
        'policy_delay', 'polyak_tau', 'learning_starts', 'train_every_steps',
        'updates_per_train', 'initial_angle_noise', 'initial_velocity_noise',
        'episode_mode_probabilities', 'adaptive_curriculum_enabled',
        'curriculum_start_episode', 'curriculum_window',
        'curriculum_min_probabilities', 'curriculum_max_probabilities',
        'transition_switch_step_min', 'transition_switch_step_max',
        'capture_angle_noise',
        'capture_cart_velocity_noise', 'capture_angular_velocity_noise',
        'down_angle_noise', 'cart_failure_penalty', 'success_bonus',
        'capture_entry_bonus', 'swing_up_energy_progress_weight',
        'swing_up_height_progress_weight', 'swing_up_cart_safety_weight',
        'capture_allowed_angular_speed', 'hold_progress_bonus', 'hold_required_steps',
        'render_training', 'render_every_episodes', 'render_first_episode',
        'plot_config',
    }
    missing = sorted(required_keys - set(cfg))
    if missing:
        raise ValueError(f"missing required config keys: {missing}")
    unknown = sorted(set(cfg) - required_keys)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")

    positive_integer_keys = (
        'num_episodes', 'max_steps', 'batch_size', 'hidden_dim',
        'buffer_capacity', 'num_nodes', 'policy_delay', 'train_every_steps',
        'updates_per_train', 'curriculum_window', 'hold_required_steps',
    )
    for key in positive_integer_keys:
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"config[{key!r}] must be a positive integer, got {value!r}")

    positive_keys = (
        'actor_lr', 'critic_lr', 'gravity', 'max_action',
        'swing_up_exploration_period_min', 'swing_up_exploration_period_max',
        'capture_allowed_angular_speed',
    )
    for key in positive_keys:
        value = cfg[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"config[{key!r}] must be a finite positive number, got {value!r}")

    if not 0.0 <= cfg['gamma'] < 1.0:
        raise ValueError(f"config['gamma'] must be in [0, 1), got {cfg['gamma']!r}")
    if not 0.0 < cfg['polyak_tau'] <= 1.0:
        raise ValueError(f"config['polyak_tau'] must be in (0, 1], got {cfg['polyak_tau']!r}")
    if cfg['transition_switch_step_min'] > cfg['transition_switch_step_max']:
        raise ValueError("transition_switch_step_min must be <= transition_switch_step_max")
    if cfg['swing_up_exploration_period_min'] > cfg['swing_up_exploration_period_max']:
        raise ValueError("swing_up_exploration_period_min must be <= swing_up_exploration_period_max")

    nonnegative_integer_keys = (
        'learning_starts', 'curriculum_start_episode', 'transition_switch_step_min',
        'transition_switch_step_max', 'render_every_episodes',
    )
    for key in nonnegative_integer_keys:
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"config[{key!r}] must be a nonnegative integer, got {value!r}")

    nonnegative_keys = (
        'friction_coefficient', 'exploration_noise', 'swing_up_exploration_noise',
        'swing_up_exploration_amplitude', 'swing_up_capture_noise', 'policy_noise',
        'noise_clip', 'initial_angle_noise', 'initial_velocity_noise',
        'capture_angle_noise', 'capture_cart_velocity_noise',
        'capture_angular_velocity_noise', 'down_angle_noise', 'success_bonus',
        'capture_entry_bonus', 'swing_up_energy_progress_weight',
        'swing_up_height_progress_weight', 'swing_up_cart_safety_weight',
        'hold_progress_bonus',
    )
    for key in nonnegative_keys:
        value = cfg[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"config[{key!r}] must be finite and nonnegative, got {value!r}")
    if not 0.0 < cfg['swing_up_capture_score_threshold'] <= 1.0:
        raise ValueError("swing_up_capture_score_threshold must be in (0, 1]")
    if not math.isfinite(cfg['cart_failure_penalty']) or cfg['cart_failure_penalty'] >= 0.0:
        raise ValueError("cart_failure_penalty must be finite and negative")

    _validate_probability_map('episode_mode_probabilities', cfg['episode_mode_probabilities'], require_sum=True)
    minimums = _validate_probability_map(
        'curriculum_min_probabilities', cfg['curriculum_min_probabilities'], require_sum=False
    )
    maximums = _validate_probability_map(
        'curriculum_max_probabilities', cfg['curriculum_max_probabilities'], require_sum=False
    )
    if sum(minimums.values()) > 1.0 + 1e-9:
        raise ValueError("curriculum_min_probabilities cannot sum to more than 1")
    if sum(maximums.values()) < 1.0 - 1e-9:
        raise ValueError("curriculum_max_probabilities must sum to at least 1")
    for mode in EPISODE_MODES:
        if minimums[mode] > maximums[mode]:
            raise ValueError(f"curriculum minimum exceeds maximum for {mode}")

    plot_config = cfg['plot_config']
    required_plot_keys = {
        'enable_plots', 'plot_frequency', 'max_points_per_plot', 'plot_dpi',
    }
    if not isinstance(plot_config, dict) or set(plot_config) != required_plot_keys:
        raise ValueError(
            f"plot_config keys must be exactly {sorted(required_plot_keys)}, "
            f"got {plot_config!r}"
        )
    if not isinstance(plot_config['enable_plots'], bool):
        raise ValueError("plot_config['enable_plots'] must be bool")
    for key in ('plot_frequency', 'max_points_per_plot', 'plot_dpi'):
        if not isinstance(plot_config[key], int) or plot_config[key] <= 0:
            raise ValueError(f"plot_config[{key!r}] must be a positive integer")
    for key in ('load_models', 'adaptive_curriculum_enabled', 'render_training', 'render_first_episode'):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"config[{key!r}] must be bool")
    return cfg


def _validate_probability_map(name, probabilities, require_sum):
    if not isinstance(probabilities, dict) or set(probabilities) != set(EPISODE_MODES):
        raise ValueError(f"{name} must define exactly {EPISODE_MODES}")
    validated = {}
    for mode in EPISODE_MODES:
        value = probabilities[mode]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name}[{mode!r}] must be finite and nonnegative")
        validated[mode] = float(value)
    if require_sum and not math.isclose(sum(validated.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} must sum to 1.0, got {sum(validated.values())}")
    return validated


validate_config(config)
