import json
import math
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from config import config
from metrics import MetricsTracker
from model import TriplePendulumActor, TriplePendulumCritic
from reward import RewardManager
from tp_env import TriplePendulumEnv


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, np.array([action], dtype=np.float32), reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class TriplePendulumTrainer:
    EPISODE_MODES = ("down_to_up", "capture_vertical", "fold_to_up", "up_to_fold")

    def __init__(self, cfg):
        self.config = cfg
        self.max_action = cfg.get("max_action", 0.5)
        self.reward_manager = RewardManager()
        self.env = TriplePendulumEnv(
            reward_manager=self.reward_manager,
            render_mode=None,
            num_nodes=cfg["num_nodes"],
            max_steps=cfg["max_steps"],
            env_config=cfg,
        )

        initial_state = self.env.reset(episode_mode="capture_vertical")
        self.state_dim = len(initial_state)
        self.action_dim = 1

        self.actor_model = TriplePendulumActor(
            self.state_dim,
            self.action_dim,
            cfg["hidden_dim"],
            self.max_action,
        )
        self.actor_target = TriplePendulumActor(
            self.state_dim,
            self.action_dim,
            cfg["hidden_dim"],
            self.max_action,
        )
        self.critic_model = TriplePendulumCritic(self.state_dim, self.action_dim, cfg["hidden_dim"])
        self.critic_target = TriplePendulumCritic(self.state_dim, self.action_dim, cfg["hidden_dim"])
        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.critic_target.load_state_dict(self.critic_model.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=cfg["critic_lr"])

        self.metrics = MetricsTracker(cfg.get("plot_config", {}))
        self.plot_frequency = cfg.get("plot_config", {}).get("plot_frequency", 500)
        self.full_plot_frequency = cfg.get("plot_config", {}).get("full_plot_frequency", 1000)
        self.memory = ReplayBuffer(capacity=cfg["buffer_capacity"])
        self.total_it = 0
        self.total_env_steps = 0

        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        if cfg["load_models"]:
            self.load_models()

    def select_action(self, state, noise_std=0.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor_model(state_tensor).cpu().numpy()[0, 0]
        if noise_std > 0:
            action += np.random.normal(0.0, noise_std)
        return float(np.clip(action, -self.max_action, self.max_action))

    def collect_trajectory(self, episode):
        selected_mode, mode_probabilities = self._select_episode_mode(episode)
        state = self.env.reset(episode_mode=selected_mode)
        should_render = self._should_render_episode(episode)
        episode_reward = 0.0
        reward_components_accumulated = {}
        action_values = []
        hold_before = []
        hold_after = []
        phase_hold = {1: [], -1: []}
        switch_step = self.env.switch_step
        termination_reason = None
        initial_phase = self.env.initial_phase
        initial_pose_mode = getattr(self.env, "initial_pose_mode", "target")
        if selected_mode == "capture_vertical":
            transition_direction = "capture_vertical"
        elif initial_pose_mode == "down":
            transition_direction = "down_to_up"
        else:
            transition_direction = "up_to_fold" if initial_phase == 1 else "fold_to_up"
        previous_post_switch_target_error = None
        transition_rewards = []
        in_target_history = []
        target_score_history = []
        end_y_history = []
        swing_period = random.uniform(
            self.config.get("swing_up_exploration_period_min", 60),
            self.config.get("swing_up_exploration_period_max", 120),
        )
        swing_phase = random.uniform(0.0, 2.0 * math.pi)
        capture_started = False

        for step_idx in range(self.config["max_steps"]):
            if initial_pose_mode == "down":
                noise_std = (
                    self.config.get("swing_up_capture_noise", 0.03)
                    if capture_started
                    else self.config.get("swing_up_exploration_noise", 0.30)
                )
            else:
                noise_std = self.config.get("exploration_noise", 0.10)
            action = self.select_action(state, noise_std=noise_std)
            if initial_pose_mode == "down" and not capture_started:
                swing_action = self.config.get("swing_up_exploration_amplitude", 0.45) * math.sin(
                    2.0 * math.pi * step_idx / max(1.0, swing_period) + swing_phase
                )
                action = float(np.clip(action + swing_action, -self.max_action, self.max_action))
            next_state, reward, done, info = self.env.step(action)
            if should_render:
                rendering_successful = self.env.render(
                    action=action,
                    episode=episode,
                    epsilon=noise_std,
                    current_step=step_idx,
                    phase=info["phase"],
                )
                if not rendering_successful:
                    done = True
                    info["termination_reason"] = "render_closed"

            components = info.get("reward_components", {})
            if (
                initial_pose_mode == "down"
                and not capture_started
                and float(info.get("target_score", 0.0)) >= self.config.get("swing_up_capture_score_threshold", 0.75)
            ):
                capture_started = True
            pre_switch_reward_weight = 1.0
            if selected_mode in ("fold_to_up", "up_to_fold") and not self.env.has_switched:
                pre_switch_reward_weight = self.config.get("transition_pre_switch_reward_weight", 0.20)
                reward *= pre_switch_reward_weight
            transition_reward = 0.0
            if info.get("switched"):
                previous_post_switch_target_error = None
            if self.env.has_switched:
                target_error = float(components.get("target_error", 0.0))
                if previous_post_switch_target_error is not None:
                    improvement = previous_post_switch_target_error - target_error
                    transition_reward = self.config.get("transition_improvement_weight", 0.5) * improvement
                    reward += transition_reward
                previous_post_switch_target_error = target_error
            components["transition_reward"] = float(transition_reward)
            components["pre_switch_reward_weight"] = float(pre_switch_reward_weight)
            components["reward"] = float(reward)
            transition_rewards.append(transition_reward)

            self.memory.push(state, action, reward, next_state, done)
            self.total_env_steps += 1
            action_values.append(action)
            episode_reward += reward

            for component_name, value in components.items():
                reward_components_accumulated.setdefault(component_name, []).append(value)

            in_target = float(info.get("in_target", 0.0))
            in_target_history.append(in_target)
            target_score_history.append(float(info.get("target_score", 0.0)))
            end_y_history.append(float(components.get("end_y", 0.0)))
            if step_idx < switch_step:
                hold_before.append(in_target)
            else:
                hold_after.append(in_target)
            phase_hold[int(info["phase"])].append(in_target)

            if self._should_update():
                for _ in range(self.config.get("updates_per_train", 1)):
                    self.update_networks()

            state = next_state
            if done:
                termination_reason = info.get("termination_reason")
                break

        episode_length = step_idx + 1
        hold_before_switch = float(np.mean(hold_before)) if hold_before else 0.0
        hold_after_switch = float(np.mean(hold_after)) if hold_after else 0.0
        final_window = max(1, int(0.2 * len(in_target_history)))
        final_hold = float(np.mean(in_target_history[-final_window:])) if in_target_history else 0.0
        peak_target_score = float(np.max(target_score_history)) if target_score_history else 0.0
        max_end_y = float(np.max(end_y_history)) if end_y_history else 0.0
        if transition_direction in ("down_to_up", "capture_vertical"):
            hold_after_switch = final_hold
            balanced_hold = final_hold
            episode_success = float(final_hold > 0.8)
        else:
            balanced_hold = min(hold_before_switch, hold_after_switch)
            episode_success = float(hold_before_switch > 0.8 and hold_after_switch > 0.8)
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "reward_components": reward_components_accumulated,
            "hold_before_switch": hold_before_switch,
            "hold_after_switch": hold_after_switch,
            "balanced_hold": balanced_hold,
            "episode_success": episode_success,
            "final_hold": final_hold,
            "peak_target_score": peak_target_score,
            "max_end_y": max_end_y,
            "success_rate_phase_1": float(np.mean(phase_hold[1])) if phase_hold[1] else 0.0,
            "success_rate_phase_2": float(np.mean(phase_hold[-1])) if phase_hold[-1] else 0.0,
            "initial_phase": initial_phase,
            "initial_pose_mode": initial_pose_mode,
            "selected_mode": selected_mode,
            "transition_direction": transition_direction,
            f"{transition_direction}_hold_before": hold_before_switch,
            f"{transition_direction}_hold_after": hold_after_switch,
            f"{transition_direction}_balanced_hold": balanced_hold,
            f"{transition_direction}_success": episode_success,
            f"{transition_direction}_final_hold": final_hold,
            f"{transition_direction}_peak_target_score": peak_target_score,
            f"{transition_direction}_max_end_y": max_end_y,
            "transition_reward_mean": float(np.mean(transition_rewards)) if transition_rewards else 0.0,
            "action_mean": float(np.mean(action_values)) if action_values else 0.0,
            "action_std": float(np.std(action_values)) if action_values else 0.0,
            "action_abs_mean": float(np.mean(np.abs(action_values))) if action_values else 0.0,
            "termination_reason": termination_reason or "none",
            "mode_probability_down_to_up": mode_probabilities["down_to_up"],
            "mode_probability_capture_vertical": mode_probabilities["capture_vertical"],
            "mode_probability_fold_to_up": mode_probabilities["fold_to_up"],
            "mode_probability_up_to_fold": mode_probabilities["up_to_fold"],
        }

    def _select_episode_mode(self, episode):
        probabilities = self._episode_mode_probabilities(episode)
        modes = list(self.EPISODE_MODES)
        weights = [probabilities[mode] for mode in modes]
        return random.choices(modes, weights=weights, k=1)[0], probabilities

    def _episode_mode_probabilities(self, episode):
        base_probabilities = self._normalized_mode_probabilities(
            self.config["episode_mode_probabilities"]
        )
        if not self.config.get("adaptive_curriculum_enabled", False):
            return base_probabilities
        if episode < self.config.get("curriculum_start_episode", 300):
            return base_probabilities

        window = self.config.get("curriculum_window", 200)
        mode_scores = {
            "down_to_up": self._recent_metric_mean("down_to_up_final_hold", window, default=0.0),
            "capture_vertical": self._recent_metric_mean("capture_vertical_final_hold", window, default=0.0),
            "fold_to_up": self._recent_metric_mean("fold_to_up_balanced_hold", window, default=0.0),
            "up_to_fold": self._recent_metric_mean("up_to_fold_balanced_hold", window, default=0.0),
        }
        raw_probabilities = {
            mode: base_probabilities[mode] * max(0.05, 1.0 - mode_scores[mode])
            for mode in self.EPISODE_MODES
        }
        normalized = self._normalized_mode_probabilities(raw_probabilities)
        return self._bounded_mode_probabilities(normalized)

    def _recent_metric_mean(self, metric_name, window, default=0.0):
        values = self.metrics.metrics.get(metric_name, [])
        if not values:
            return default
        return float(np.mean(values[-window:]))

    def _normalized_mode_probabilities(self, probabilities):
        sanitized = {
            mode: max(0.0, float(probabilities.get(mode, 0.0)))
            for mode in self.EPISODE_MODES
        }
        total = sum(sanitized.values())
        if total <= 0.0:
            return {mode: 1.0 / len(self.EPISODE_MODES) for mode in self.EPISODE_MODES}
        return {mode: value / total for mode, value in sanitized.items()}

    def _bounded_mode_probabilities(self, probabilities):
        min_probabilities = self.config.get("curriculum_min_probabilities", {})
        max_probabilities = self.config.get("curriculum_max_probabilities", {})
        bounded = {
            mode: min(
                float(max_probabilities.get(mode, 1.0)),
                max(float(min_probabilities.get(mode, 0.0)), probabilities[mode]),
            )
            for mode in self.EPISODE_MODES
        }
        for _ in range(8):
            total = sum(bounded.values())
            diff = 1.0 - total
            if abs(diff) < 1e-9:
                break
            if diff > 0:
                adjustable = [
                    mode for mode in self.EPISODE_MODES
                    if bounded[mode] < float(max_probabilities.get(mode, 1.0))
                ]
            else:
                adjustable = [
                    mode for mode in self.EPISODE_MODES
                    if bounded[mode] > float(min_probabilities.get(mode, 0.0))
                ]
            if not adjustable:
                break
            share = diff / len(adjustable)
            for mode in adjustable:
                bounded[mode] = min(
                    float(max_probabilities.get(mode, 1.0)),
                    max(float(min_probabilities.get(mode, 0.0)), bounded[mode] + share),
                )
        return self._normalized_mode_probabilities(bounded)

    def _should_render_episode(self, episode):
        if not self.config.get("render_training", False):
            return False
        if episode == 0 and self.config.get("render_first_episode", True):
            return True
        render_every = self.config.get("render_every_episodes", 200)
        return render_every > 0 and episode % render_every == 0

    def _should_update(self):
        if len(self.memory) < self.config["batch_size"]:
            return False
        if self.total_env_steps < self.config.get("learning_starts", self.config["batch_size"]):
            return False
        train_every = self.config.get("train_every_steps", 1)
        return train_every <= 1 or self.total_env_steps % train_every == 0

    def update_networks(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.config["batch_size"])
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)

        self.total_it += 1
        with torch.no_grad():
            noise = torch.randn_like(actions_tensor) * self.config.get("policy_noise", 0.08)
            noise = noise.clamp(
                -self.config.get("noise_clip", 0.15),
                self.config.get("noise_clip", 0.15),
            )
            next_actions = (self.actor_target(next_states_tensor) + noise).clamp(
                -self.max_action,
                self.max_action,
            )
            target_q1, target_q2 = self.critic_target(next_states_tensor, next_actions)
            target_q = torch.min(target_q1, target_q2)
            td_targets = rewards_tensor + self.config["gamma"] * (1 - dones_tensor) * target_q

        current_q1, current_q2 = self.critic_model(states_tensor, actions_tensor)
        critic_loss = F.mse_loss(current_q1, td_targets) + F.mse_loss(current_q2, td_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % self.config.get("policy_delay", 2) == 0:
            actor_actions = self.actor_model(states_tensor)
            actor_loss = -self.critic_model.q1_value(states_tensor, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()

            self._polyak_update(self.actor_model, self.actor_target)
            self._polyak_update(self.critic_model, self.critic_target)

        self.metrics.add_metric("critic_loss", critic_loss.item())
        self.metrics.add_metric("actor_loss", actor_loss_value)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_value}

    def _polyak_update(self, online_model, target_model):
        tau = self.config.get("polyak_tau", 0.005)
        with torch.no_grad():
            for param, target_param in zip(online_model.parameters(), target_model.parameters()):
                target_param.data.mul_(1 - tau)
                target_param.data.add_(tau * param.data)

    def train(self):
        try:
            for episode in range(self.config["num_episodes"]):
                summary = self.collect_trajectory(episode)
                self._record_episode_metrics(summary)

                if episode % 10 == 0:
                    print(
                        f"Episode {episode:5d} | "
                        f"reward={summary['episode_reward']:8.2f} | "
                        f"len={summary['episode_length']:4d} | "
                        f"dir={summary['transition_direction']:<10s} | "
                        f"p=({summary['mode_probability_down_to_up']:.2f},"
                        f"{summary['mode_probability_capture_vertical']:.2f},"
                        f"{summary['mode_probability_fold_to_up']:.2f},"
                        f"{summary['mode_probability_up_to_fold']:.2f}) | "
                        f"before={summary['hold_before_switch']:4.2f} | "
                        f"after={summary['hold_after_switch']:4.2f} | "
                        f"balanced={summary['balanced_hold']:4.2f} | "
                        f"peak={summary['peak_target_score']:4.2f}"
                    )

                if episode % self.plot_frequency == self.plot_frequency - 1:
                    self.metrics.generate_all_plots()
                    self.save_models("models/checkpoint", episode=episode)
        except KeyboardInterrupt:
            episode = len(self.metrics.metrics["episode_reward"])
            self.save_models("models/checkpoint", episode=episode, interrupted=True)
            self.save_models("models/interrupted", episode=episode, interrupted=True)
            print(f"\nKeyboardInterrupt: saved checkpoint at episode {episode}")
            raise

    def _record_episode_metrics(self, summary):
        scalar_keys = [
            "episode_reward",
            "episode_length",
            "hold_before_switch",
            "hold_after_switch",
            "balanced_hold",
            "episode_success",
            "final_hold",
            "peak_target_score",
            "max_end_y",
            "success_rate_phase_1",
            "success_rate_phase_2",
            "transition_reward_mean",
            "action_mean",
            "action_std",
            "action_abs_mean",
            "mode_probability_down_to_up",
            "mode_probability_capture_vertical",
            "mode_probability_fold_to_up",
            "mode_probability_up_to_fold",
        ]
        for key in scalar_keys:
            self.metrics.add_metric(key, summary[key])

        for component_name, values in summary["reward_components"].items():
            if values:
                self.metrics.add_metric(component_name, float(np.mean(values)))

        for direction in ("up_to_fold", "fold_to_up", "down_to_up", "capture_vertical"):
            for suffix in (
                "hold_before",
                "hold_after",
                "balanced_hold",
                "success",
                "final_hold",
                "peak_target_score",
                "max_end_y",
            ):
                key = f"{direction}_{suffix}"
                if key in summary:
                    self.metrics.add_metric(key, summary[key])

    def save_models(self, path, episode=None, interrupted=False):
        torch.save(self.actor_model.state_dict(), path + "_actor.pth")
        torch.save(self.critic_model.state_dict(), path + "_critic.pth")
        torch.save(self.actor_optimizer.state_dict(), path + "_actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), path + "_critic_optimizer.pth")
        metadata = {
            "algorithm": "td3",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "num_nodes": self.config["num_nodes"],
            "episode": episode,
            "total_updates": self.total_it,
            "total_env_steps": self.total_env_steps,
            "interrupted": interrupted,
            "hidden_dim": self.config["hidden_dim"],
            "model_version": 3,
        }
        with open(path + "_metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def load_models(self):
        metadata_path = "models/checkpoint_metadata.json"
        if not os.path.exists(metadata_path):
            print("Skipping checkpoint load: missing TD3 metadata")
            return
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        expected = {
            "algorithm": "td3",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.config["hidden_dim"],
            "model_version": 3,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                print(f"Skipping checkpoint load: {key} is {metadata.get(key)}, expected {value}")
                return

        self.actor_model.load_state_dict(torch.load("models/checkpoint_actor.pth", weights_only=True))
        self.critic_model.load_state_dict(torch.load("models/checkpoint_critic.pth", weights_only=True))
        self.actor_target.load_state_dict(self.actor_model.state_dict())
        self.critic_target.load_state_dict(self.critic_model.state_dict())
        self.actor_optimizer.load_state_dict(torch.load("models/checkpoint_actor_optimizer.pth", weights_only=True))
        self.critic_optimizer.load_state_dict(torch.load("models/checkpoint_critic_optimizer.pth", weights_only=True))


if __name__ == "__main__":
    trainer = TriplePendulumTrainer(config)
    trainer.train()
