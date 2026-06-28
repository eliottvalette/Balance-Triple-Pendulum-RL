# metrics.py
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

DARK_FIGURE = "#0d1117"
DARK_AXES = "#161b22"
DARK_GRID = "#30363d"
DARK_TEXT = "#e6edf3"

REWARD_COMPONENTS = [
    "target_score",
    "effective_target_score",
    "target_error",
    "energy_score",
    "height_score",
    "cart_safety_score",
    "potential_progress",
    "capture_quality",
    "capture_quality_bonus",
    "capture_maintenance_bonus",
    "capture_score_decay_penalty",
    "capture_in_target_bonus",
    "capture_entry_bonus",
    "capture_drop_penalty",
    "hold_bonus",
    "hold_progress",
    "hold_progress_delta",
    "cart_proximity_penalty",
    "cart_limit_step_penalty",
    "cart_failure_penalty",
    "transition_reward",
]

EPISODE_DIAGNOSTIC_METRICS = [
    "peak_target_score",
    "hold_before_switch",
    "hold_after_switch",
    "balanced_hold",
    "overall_hold",
    "hold_vs_max_steps",
    "final_hold",
]

ACTION_METRICS = [
    "action_mean",
    "action_std",
    "action_abs_mean",
]

LOSS_METRICS = [
    "policy_loss",
    "value_loss",
]


class MetricsTracker:
    def __init__(self, plot_config):
        self.metrics = defaultdict(list)
        self.episode_window = 100
        required = {"max_points_per_plot", "plot_dpi", "enable_plots", "plot_frequency"}
        if not isinstance(plot_config, dict) or set(plot_config) != required:
            raise ValueError(f"plot_config keys must be exactly {sorted(required)}")
        self.max_points_per_plot = plot_config["max_points_per_plot"]
        self.plot_dpi = plot_config["plot_dpi"]
        self.enable_plots = plot_config["enable_plots"]

    def add_metric(self, name, value):
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise TypeError(f"metric {name!r} must be scalar, got array shape {value.shape}")
            value = float(value.item())
        elif isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(f"metric {name!r} must be scalar, got tensor shape {tuple(value.shape)}")
            value = float(value.item())
        elif isinstance(value, (list, tuple)) or not np.isscalar(value):
            raise TypeError(f"metric {name!r} must be scalar, got {type(value).__name__}")
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError(f"metric {name!r} is non-finite: {value}")
        self.metrics[name].append(value)

    def get_moving_average(self, name):
        if name not in self.metrics or not self.metrics[name]:
            raise KeyError(f"metric {name!r} is missing or empty")
        values = self.metrics[name]
        if len(values) < self.episode_window:
            return np.mean(values)
        return np.mean(values[-self.episode_window :])

    def _downsample_if_needed(self, data):
        if len(data) <= self.max_points_per_plot:
            return np.array(data), np.arange(len(data))
        step = max(1, len(data) // self.max_points_per_plot)
        indices = np.concatenate(
            [
                np.arange(0, 100, 1),
                np.arange(100, len(data) - 100, step),
                np.arange(max(100, len(data) - 100), len(data), 1),
            ]
        )
        indices = np.unique(indices)
        indices = indices[indices < len(data)]
        return np.array(data)[indices], indices

    def _panel_color_map(self, names):
        present = [name for name in names if name in self.metrics and self.metrics[name]]
        if not present:
            return {}
        positions = np.linspace(0.52, 0.98, len(present))
        cmap = plt.cm.inferno
        return {name: cmap(position) for name, position in zip(present, positions)}

    def _apply_dark_axes(self, ax):
        ax.set_facecolor(DARK_AXES)
        ax.tick_params(colors=DARK_TEXT, labelcolor=DARK_TEXT)
        ax.xaxis.label.set_color(DARK_TEXT)
        ax.yaxis.label.set_color(DARK_TEXT)
        ax.title.set_color(DARK_TEXT)
        for spine in ax.spines.values():
            spine.set_color(DARK_GRID)
        ax.grid(color=DARK_GRID, alpha=0.45, linewidth=0.5)

    def _legend_below(self, ax, ncol):
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=ncol,
            fontsize=8,
            framealpha=0.95,
            facecolor=DARK_AXES,
            edgecolor=DARK_GRID,
            labelcolor=DARK_TEXT,
            columnspacing=1.0,
            handlelength=2.0,
        )

    def _legend_right(self, ax):
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            framealpha=0.95,
            facecolor=DARK_AXES,
            edgecolor=DARK_GRID,
            labelcolor=DARK_TEXT,
            handlelength=2.2,
        )

    def _moving_average(self, values):
        if len(values) < self.episode_window:
            return None, None
        moving_avg = np.convolve(values, np.ones(self.episode_window) / self.episode_window, mode="valid")
        indices = np.arange(self.episode_window - 1, self.episode_window - 1 + len(moving_avg))
        return moving_avg, indices

    def _plot_moving_average(self, ax, values, color, label, *, linestyle="-", linewidth=2.6, alpha=1.0):
        moving_avg, moving_indices = self._moving_average(values)
        if moving_avg is None:
            return
        ma_ds, ma_idx = self._downsample_if_needed(moving_avg)
        ax.plot(moving_indices[ma_idx], ma_ds, color=color, linewidth=linewidth, linestyle=linestyle, label=label, alpha=alpha)

    def _percentile_ylim(self, values, low=3.0, high=97.0, padding=0.12):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        lower, upper = np.percentile(finite, [low, high])
        if upper <= lower:
            upper = lower + 1.0
        margin = (upper - lower) * padding
        return lower - margin, upper + margin

    def _loss_spike_threshold(self, values):
        median = float(np.median(values))
        percentile = float(np.percentile(values, 99.5))
        return max(median * 8.0, percentile, 1e-6)

    def _plot_loss_panel(self, ax, color_map):
        spike_notes = []
        for loss_name in LOSS_METRICS:
            if loss_name not in self.metrics or not self.metrics[loss_name]:
                continue
            values = np.array(self.metrics[loss_name], dtype=float)
            color = color_map[loss_name]
            spike_threshold = self._loss_spike_threshold(values)
            spike_count = int(np.sum(values > spike_threshold))
            if spike_count > 0:
                spike_notes.append(f"{loss_name}>{spike_threshold:.1f}: {spike_count}")
            clipped = np.clip(values, None, spike_threshold)
            raw_ds, raw_idx = self._downsample_if_needed(clipped)
            ax.plot(raw_idx, raw_ds, color=color, alpha=0.12, linewidth=0.7)
            self._plot_moving_average(ax, values, color, f"{loss_name} MA{self.episode_window}")
        title = "PPO Losses"
        if spike_notes:
            title += f"\nspikes: {' | '.join(spike_notes[:2])}"
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.set_yscale("symlog", linthresh=1e-3)

    def plot_metrics(self, save_path=None):
        if not self.enable_plots:
            return

        fig = plt.figure(figsize=(28, 15), facecolor=DARK_FIGURE)
        grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.38)
        reward_ax = fig.add_subplot(grid[0, 0])
        loss_ax = fig.add_subplot(grid[0, 1])
        diagnostic_ax = fig.add_subplot(grid[0, 2])
        components_ax = fig.add_subplot(grid[1, 0:2])
        action_ax = fig.add_subplot(grid[1, 2])
        fig.suptitle("Training Metrics", color=DARK_TEXT, fontsize=17, y=0.98)

        self._apply_dark_axes(reward_ax)
        reward_colors = self._panel_color_map(["episode_reward"])
        if "episode_reward" in self.metrics and self.metrics["episode_reward"]:
            rewards = np.array(self.metrics["episode_reward"], dtype=float)
            reward_color = reward_colors["episode_reward"]
            rewards_ds, reward_idx = self._downsample_if_needed(rewards)
            reward_ax.plot(reward_idx, rewards_ds, color=reward_color, alpha=0.14, linewidth=0.8)
            self._plot_moving_average(reward_ax, rewards, reward_color, f"reward MA{self.episode_window}")
            reward_ylim = self._percentile_ylim(rewards, low=1.0, high=99.5, padding=0.15)
            if reward_ylim is not None:
                reward_ax.set_ylim(reward_ylim)
        reward_ax.set_title("Rewards", fontsize=11, pad=10)
        reward_ax.set_xlabel("Episode")
        reward_ax.set_ylabel("Reward")
        self._legend_right(reward_ax)

        self._apply_dark_axes(loss_ax)
        loss_colors = self._panel_color_map(LOSS_METRICS)
        self._plot_loss_panel(loss_ax, loss_colors)
        self._legend_right(loss_ax)

        self._apply_dark_axes(diagnostic_ax)
        diagnostic_names = list(EPISODE_DIAGNOSTIC_METRICS)
        diagnostic_colors = self._panel_color_map(diagnostic_names)
        for metric_name in EPISODE_DIAGNOSTIC_METRICS:
            if metric_name not in self.metrics or len(self.metrics[metric_name]) < self.episode_window:
                continue
            values = np.array(self.metrics[metric_name], dtype=float)
            self._plot_moving_average(diagnostic_ax, values, diagnostic_colors[metric_name], metric_name)
        diagnostic_ax.set_title(
            "Episode Diagnostics (MA)\npeak | before | after | balanced",
            fontsize=11,
            pad=10,
        )
        diagnostic_ax.set_xlabel("Episode")
        diagnostic_ax.set_ylabel("Value")
        diagnostic_ax.set_ylim(-0.02, 1.05)
        self._legend_right(diagnostic_ax)

        self._apply_dark_axes(components_ax)
        component_names = [name for name in REWARD_COMPONENTS if name in self.metrics and len(self.metrics[name]) >= self.episode_window]
        component_colors = self._panel_color_map(component_names)
        plotted_components = 0
        for component_name in component_names:
            values = np.array(self.metrics[component_name], dtype=float)
            self._plot_moving_average(components_ax, values, component_colors[component_name], component_name, linewidth=2.4)
            plotted_components += 1
        components_ax.set_title(f"Reward Components (MA{self.episode_window})", fontsize=11, pad=10)
        components_ax.set_xlabel("Episode")
        components_ax.set_ylabel("Value")
        if plotted_components > 0:
            self._legend_below(components_ax, ncol=4)

        self._apply_dark_axes(action_ax)
        action_names = [name for name in ACTION_METRICS if name in self.metrics and len(self.metrics[name]) >= self.episode_window]
        action_colors = self._panel_color_map(action_names)
        for action_metric in action_names:
            values = np.array(self.metrics[action_metric], dtype=float)
            self._plot_moving_average(action_ax, values, action_colors[action_metric], action_metric, linewidth=2.6)
        action_ax.set_title("Action Metrics (MA)", fontsize=11, pad=10)
        action_ax.set_xlabel("Episode")
        action_ax.set_ylabel("Value")
        self._legend_right(action_ax)

        fig.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.14)
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi, facecolor=fig.get_facecolor())
        plt.close()

    def plot_reward_distribution(self, save_path=None):
        if not self.enable_plots or len(self.metrics["episode_reward"]) < 10:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), facecolor=DARK_FIGURE)
        for axis in axes:
            self._apply_dark_axes(axis)

        rewards = self.metrics["episode_reward"]
        if len(rewards) > self.max_points_per_plot:
            sample_indices = np.linspace(0, len(rewards) - 1, self.max_points_per_plot, dtype=int)
            sampled_rewards = [rewards[index] for index in sample_indices]
            sns.histplot(sampled_rewards, kde=True, ax=axes[0], color=plt.cm.inferno(0.72))
        else:
            sns.histplot(rewards, kde=True, ax=axes[0], color=plt.cm.inferno(0.72))
        axes[0].set_title("Reward distribution", color=DARK_TEXT)
        axes[0].set_xlabel("Reward")
        axes[0].set_ylabel("Count")

        num_episodes = len(self.metrics["episode_reward"])
        first_quarter = num_episodes // 4
        last_quarter = max(first_quarter, num_episodes - first_quarter)
        if first_quarter > 0:
            if first_quarter > self.max_points_per_plot // 2:
                sample_indices = np.linspace(0, first_quarter - 1, self.max_points_per_plot // 2, dtype=int)
                first_data = [self.metrics["episode_reward"][index] for index in sample_indices]
            else:
                first_data = self.metrics["episode_reward"][:first_quarter]
            if (num_episodes - last_quarter) > self.max_points_per_plot // 2:
                sample_indices = np.linspace(last_quarter, num_episodes - 1, self.max_points_per_plot // 2, dtype=int)
                last_data = [self.metrics["episode_reward"][index] for index in sample_indices]
            else:
                last_data = self.metrics["episode_reward"][last_quarter:]
            sns.kdeplot(first_data, label="first quarter", ax=axes[1], color=plt.cm.inferno(0.58))
            sns.kdeplot(last_data, label="last quarter", ax=axes[1], color=plt.cm.inferno(0.92))
            axes[1].set_title("Reward distribution: start vs end", color=DARK_TEXT)
            axes[1].set_xlabel("Reward")
            axes[1].set_ylabel("Density")
            self._legend_right(axes[1])

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi, facecolor=fig.get_facecolor())
        plt.close()

    def generate_all_plots(self, base_path="results"):
        if not self.enable_plots:
            return
        os.makedirs(base_path, exist_ok=True)
        self.plot_metrics(f"{base_path}/metrics.png")
        self.plot_reward_distribution(f"{base_path}/reward_distribution.png")
        print(f"Graphiques générés dans {base_path}")
