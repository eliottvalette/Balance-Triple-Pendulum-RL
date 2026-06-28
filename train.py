from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor

from config import EPISODE_MODES as CONFIG_EPISODE_MODES
from config import config, validate_config
from metrics import MetricsTracker
from model import PendulumActorPolicy, PendulumValueCritic
from reward import RewardManager
from tp_env import PendulumEnv

MODEL_SAVE_PREFIX = "models/interrupted"
MODEL_LOAD_PREFIX = "models/checkpoint"
ACTOR_EVAL_CAPTURE_SEEDS = (0, 1, 2, 3, 4)

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


def compute_gae(
    rewards: FloatArray,
    values: FloatArray,
    next_values: FloatArray,
    dones: BoolArray,
    terminated: BoolArray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Compute GAE without leaking advantages across episode boundaries.

    A true termination disables value bootstrapping. A time-limit truncation still
    bootstraps from the final observation, while stopping recursive GAE propagation.
    """

    arrays = (rewards, values, next_values, dones, terminated)
    lengths = {len(array) for array in arrays}
    if lengths == {0}:
        raise ValueError("cannot compute GAE for an empty rollout")
    if len(lengths) != 1:
        raise ValueError("GAE inputs must have the same length")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    for name, array in zip(("rewards", "values", "next_values"), arrays[:3]):
        if not np.all(np.isfinite(array)):
            raise FloatingPointError(f"{name} contains non-finite values")

    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap_mask = 0.0 if bool(terminated[index]) else 1.0
        continuation_mask = 0.0 if bool(dones[index]) else 1.0
        delta = (
            float(rewards[index])
            + gamma * bootstrap_mask * float(next_values[index])
            - float(values[index])
        )
        gae = delta + gamma * gae_lambda * continuation_mask * gae
        advantages[index] = gae
    returns = advantages + values.astype(np.float32, copy=False)
    return advantages, returns


class RolloutBuffer:
    """Short-lived storage for samples from the current policy only."""

    def __init__(self) -> None:
        self.clear()

    def add(
        self,
        *,
        state: NDArray[np.floating[Any]],
        action: float,
        reward: float,
        done: bool,
        terminated: bool,
        value: float,
        next_value: float,
        log_prob: float,
    ) -> None:
        numeric_values = (action, reward, value, next_value, log_prob)
        if not np.all(np.isfinite(state)) or not all(np.isfinite(item) for item in numeric_values):
            raise FloatingPointError("rollout transition contains non-finite values")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be bool")
        if not isinstance(terminated, (bool, np.bool_)):
            raise TypeError("terminated must be bool")
        if terminated and not done:
            raise ValueError("a terminated transition must also be done")

        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(float(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.terminated.append(bool(terminated))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.log_probs.append(float(log_prob))

    def compute_advantages(self, *, gamma: float, gae_lambda: float) -> None:
        self.advantages, self.returns = compute_gae(
            np.asarray(self.rewards, dtype=np.float32),
            np.asarray(self.values, dtype=np.float32),
            np.asarray(self.next_values, dtype=np.float32),
            np.asarray(self.dones, dtype=np.bool_),
            np.asarray(self.terminated, dtype=np.bool_),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

    def as_tensors(self) -> dict[str, Tensor]:
        if len(self) == 0:
            raise ValueError("cannot create a batch from an empty rollout")
        if self.advantages is None or self.returns is None:
            raise RuntimeError("advantages must be computed before batching")
        return {
            "states": torch.as_tensor(np.stack(self.states), dtype=torch.float32),
            "actions": torch.as_tensor(self.actions, dtype=torch.float32).unsqueeze(-1),
            "old_log_probs": torch.as_tensor(self.log_probs, dtype=torch.float32),
            "advantages": torch.as_tensor(self.advantages, dtype=torch.float32),
            "returns": torch.as_tensor(self.returns, dtype=torch.float32),
        }

    def clear(self) -> None:
        self.states: list[FloatArray] = []
        self.actions: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.terminated: list[bool] = []
        self.values: list[float] = []
        self.next_values: list[float] = []
        self.log_probs: list[float] = []
        self.advantages: FloatArray | None = None
        self.returns: FloatArray | None = None

    def __len__(self) -> int:
        return len(self.states)


class PendulumTrainer:
    EPISODE_MODES = CONFIG_EPISODE_MODES
    ACTOR_EVAL_CAPTURE_SEEDS = ACTOR_EVAL_CAPTURE_SEEDS

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        validate_config(cfg)
        self.config = dict(cfg)
        self.max_action = float(cfg["max_action"])
        self.ppo_reward_scale = float(cfg["ppo_reward_scale"])
        self.reward_manager = RewardManager(self.config)
        self.env = PendulumEnv(
            reward_manager=self.reward_manager,
            render_mode=None,
            num_nodes=int(cfg["num_nodes"]),
            max_steps=int(cfg["max_steps"]),
            env_config=self.config,
        )
        self.eval_env = PendulumEnv(
            reward_manager=RewardManager(self.config),
            render_mode=None,
            num_nodes=int(cfg["num_nodes"]),
            max_steps=int(cfg["max_steps"]),
            env_config=self.config,
        )

        initial_state = self.env.reset(episode_mode="capture_vertical")
        self.state_dim = len(initial_state)
        self.action_dim = 1
        self.actor_model = PendulumActorPolicy(
            self.state_dim,
            self.action_dim,
            int(cfg["hidden_dim"]),
            self.max_action,
            float(cfg["initial_log_std"]),
        )
        self.critic_model = PendulumValueCritic(self.state_dim, int(cfg["hidden_dim"]))
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=float(cfg["actor_lr"])
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=float(cfg["critic_lr"])
        )

        self.rollout_buffer = RolloutBuffer()
        self.metrics = MetricsTracker(cfg["plot_config"])
        self.plot_frequency = int(cfg["plot_config"]["plot_frequency"])
        self.total_updates = 0
        self.total_env_steps = 0

        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        if cfg["load_models"]:
            self.load_models()

    def select_action(self, state: NDArray[np.floating[Any]], *, deterministic: bool) -> float:
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("policy state contains non-finite values")
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action_tensor = self.actor_model.deterministic(state_tensor)
            else:
                action_tensor, _log_prob, _entropy = self.actor_model.sample(state_tensor)
        action = float(action_tensor.item())
        if not np.isfinite(action):
            raise FloatingPointError("policy produced a non-finite action")
        return action

    def _policy_step(self, state: NDArray[np.floating[Any]]) -> tuple[float, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _entropy = self.actor_model.sample(state_tensor)
            value = self.critic_model(state_tensor)
        return float(action.item()), float(log_prob.item()), float(value.item())

    def _actor_raw_action(self, state: NDArray[np.floating[Any]]) -> float:
        return self.select_action(state, deterministic=True)

    def collect_rollout(self, episode: int) -> dict[str, Any]:
        if len(self.rollout_buffer) != 0:
            raise RuntimeError("the previous rollout must be consumed before collecting another")

        selected_mode, mode_probabilities = self._select_episode_mode(episode)
        state = self.env.reset(episode_mode=selected_mode)
        should_render = self._should_render_episode(episode)
        episode_reward = 0.0
        reward_components_accumulated: dict[str, list[float]] = {}
        action_values: list[float] = []
        hold_before: list[float] = []
        hold_after: list[float] = []
        phase_hold: dict[int, list[float]] = {1: [], -1: []}
        switch_step = self.env.switch_step
        termination_reason: str | None = None
        initial_phase = self.env.initial_phase
        initial_pose_mode = self.env.initial_pose_mode
        transition_direction = self._transition_direction(
            selected_mode, initial_pose_mode, initial_phase
        )
        in_target_history: list[float] = []
        target_score_history: list[float] = []
        end_y_history: list[float] = []
        capture_drop_recovery_step: int | None = None

        for step_index in range(int(self.config["max_steps"])):
            action, log_prob, value = self._policy_step(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            stop_requested = False
            if should_render:
                rendering_successful = self.env.render(
                    action=action,
                    episode=episode,
                    epsilon=0.0,
                    current_step=step_index,
                    phase=info["phase"],
                )
                stop_requested = not rendering_successful

            done = bool(terminated or truncated or stop_requested)
            effective_termination = bool(terminated or stop_requested)
            with torch.no_grad():
                next_value = 0.0
                if not effective_termination:
                    next_state_tensor = torch.as_tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0)
                    next_value = float(self.critic_model(next_state_tensor).item())
            self.rollout_buffer.add(
                state=state,
                action=action,
                reward=float(reward) * self.ppo_reward_scale,
                done=done,
                terminated=effective_termination,
                value=value,
                next_value=next_value,
                log_prob=log_prob,
            )
            self.total_env_steps += 1

            components = info["reward_components"]
            components["transition_reward"] = 0.0
            components["pre_switch_reward_weight"] = 1.0
            action_values.append(action)
            episode_reward += float(reward)
            for component_name, component_value in components.items():
                reward_components_accumulated.setdefault(component_name, []).append(
                    float(component_value)
                )

            in_target = float(info["in_target"])
            in_target_history.append(in_target)
            target_score_history.append(float(info["target_score"]))
            end_y_history.append(float(components["end_y"]))
            if step_index < switch_step:
                hold_before.append(in_target)
            else:
                hold_after.append(in_target)
            phase_hold[int(info["phase"])].append(in_target)
            if info["capture_drop_recovered"]:
                capture_drop_recovery_step = step_index + 1

            state = next_state
            if done:
                termination_reason = "render_closed" if stop_requested else info["termination_reason"]
                break

        self.rollout_buffer.compute_advantages(
            gamma=float(self.config["gamma"]),
            gae_lambda=float(self.config["gae_lambda"]),
        )
        return self._build_episode_summary(
            episode_reward=episode_reward,
            episode_length=step_index + 1,
            reward_components=reward_components_accumulated,
            hold_before=hold_before,
            hold_after=hold_after,
            phase_hold=phase_hold,
            in_target_history=in_target_history,
            target_score_history=target_score_history,
            end_y_history=end_y_history,
            action_values=action_values,
            initial_phase=initial_phase,
            initial_pose_mode=initial_pose_mode,
            selected_mode=selected_mode,
            transition_direction=transition_direction,
            termination_reason=termination_reason or "none",
            mode_probabilities=mode_probabilities,
            capture_drop_recovery_step=capture_drop_recovery_step,
        )

    def collect_trajectory(self, episode: int) -> dict[str, Any]:
        """Compatibility name for callers; collection is still an on-policy rollout."""
        return self.collect_rollout(episode)

    @staticmethod
    def _transition_direction(selected_mode: str, initial_pose_mode: str, initial_phase: int) -> str:
        if selected_mode == "capture_vertical":
            return "capture_vertical"
        if initial_pose_mode == "down":
            return "down_to_up"
        return "up_to_fold" if initial_phase == 1 else "fold_to_up"

    def _build_episode_summary(
        self,
        *,
        episode_reward: float,
        episode_length: int,
        reward_components: dict[str, list[float]],
        hold_before: list[float],
        hold_after: list[float],
        phase_hold: dict[int, list[float]],
        in_target_history: list[float],
        target_score_history: list[float],
        end_y_history: list[float],
        action_values: list[float],
        initial_phase: int,
        initial_pose_mode: str,
        selected_mode: str,
        transition_direction: str,
        termination_reason: str,
        mode_probabilities: Mapping[str, float],
        capture_drop_recovery_step: int | None,
    ) -> dict[str, Any]:
        hold_before_switch = float(np.mean(hold_before)) if hold_before else 0.0
        hold_after_switch = float(np.mean(hold_after)) if hold_after else 0.0
        final_window = max(1, int(0.2 * len(in_target_history)))
        final_hold = float(np.mean(in_target_history[-final_window:]))
        overall_hold = float(np.mean(in_target_history)) if in_target_history else 0.0
        max_steps = int(self.config["max_steps"])
        steps_in_target = float(np.sum(in_target_history)) if in_target_history else 0.0
        hold_vs_max_steps = steps_in_target / max_steps
        peak_target_score = float(np.max(target_score_history))
        max_end_y = float(np.max(end_y_history))
        if transition_direction == "capture_vertical":
            hold_after_switch = hold_vs_max_steps
            balanced_hold = hold_vs_max_steps
            episode_success = float(hold_vs_max_steps > 0.8)
        elif transition_direction == "down_to_up":
            hold_after_switch = final_hold
            balanced_hold = final_hold
            episode_success = float(final_hold > 0.8)
        else:
            balanced_hold = min(hold_before_switch, hold_after_switch)
            episode_success = float(hold_before_switch > 0.8 and hold_after_switch > 0.8)

        summary: dict[str, Any] = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "reward_components": reward_components,
            "hold_before_switch": hold_before_switch,
            "hold_after_switch": hold_after_switch,
            "balanced_hold": balanced_hold,
            "episode_success": episode_success,
            "final_hold": final_hold,
            "overall_hold": overall_hold,
            "hold_vs_max_steps": hold_vs_max_steps,
            "peak_target_score": peak_target_score,
            "max_end_y": max_end_y,
            "success_rate_phase_1": float(np.mean(phase_hold[1])) if phase_hold[1] else 0.0,
            "success_rate_phase_2": float(np.mean(phase_hold[-1])) if phase_hold[-1] else 0.0,
            "initial_phase": initial_phase,
            "initial_pose_mode": initial_pose_mode,
            "selected_mode": selected_mode,
            "transition_direction": transition_direction,
            "transition_reward_mean": 0.0,
            "action_mean": float(np.mean(action_values)),
            "action_std": float(np.std(action_values)),
            "action_abs_mean": float(np.mean(np.abs(action_values))),
            "termination_reason": termination_reason,
            "capture_drop_recovered": capture_drop_recovery_step is not None,
            "capture_drop_recovery_step": capture_drop_recovery_step,
        }
        for mode in self.EPISODE_MODES:
            summary[f"mode_probability_{mode}"] = float(mode_probabilities[mode])
        for suffix, value in (
            ("hold_before", hold_before_switch),
            ("hold_after", hold_after_switch),
            ("balanced_hold", balanced_hold),
            ("success", episode_success),
            ("final_hold", final_hold),
            ("peak_target_score", peak_target_score),
            ("max_end_y", max_end_y),
        ):
            summary[f"{transition_direction}_{suffix}"] = value
        return summary

    def update_ppo(self) -> dict[str, float]:
        batch = self.rollout_buffer.as_tensors()
        advantages = batch["advantages"]
        if self.config["normalize_advantages"] and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = advantages.shape[0]
        minibatch_size = min(int(self.config["minibatch_size"]), batch_size)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kl_values: list[float] = []
        clip_fractions: list[float] = []
        early_stop = False

        for _epoch in range(int(self.config["ppo_epochs"])):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = indices[start : start + minibatch_size]
                states = batch["states"][minibatch_indices]
                actions = batch["actions"][minibatch_indices]
                old_log_probs = batch["old_log_probs"][minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices]
                returns = batch["returns"][minibatch_indices]

                new_log_probs, entropy = self.actor_model.evaluate_actions(states, actions)
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                unclipped_objective = ratio * minibatch_advantages
                clipped_ratio = ratio.clamp(
                    1.0 - float(self.config["clip_epsilon"]),
                    1.0 + float(self.config["clip_epsilon"]),
                )
                clipped_objective = clipped_ratio * minibatch_advantages
                policy_loss = -torch.min(unclipped_objective, clipped_objective).mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - float(self.config["entropy_coefficient"]) * entropy_mean

                predicted_values = self.critic_model(states)
                value_loss = F.mse_loss(predicted_values, returns)
                critic_loss = float(self.config["value_loss_coefficient"]) * value_loss
                if not torch.isfinite(actor_loss) or not torch.isfinite(critic_loss):
                    raise FloatingPointError("PPO loss became non-finite")

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_model.parameters(), float(self.config["max_grad_norm"])
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_model.parameters(), float(self.config["max_grad_norm"])
                )
                self.critic_optimizer.step()
                self.total_updates += 1

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().clamp_min(0.0)
                    clip_fraction = ((ratio - 1.0).abs() > float(self.config["clip_epsilon"])).float().mean()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                kl_values.append(float(approx_kl.item()))
                clip_fractions.append(float(clip_fraction.item()))

                target_kl = self.config["target_kl"]
                if target_kl is not None and approx_kl.item() > float(target_kl):
                    early_stop = True
                    break
            if early_stop:
                break

        result = {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kl_values)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "ppo_epochs_completed": float(_epoch + 1),
            "kl_early_stop": float(early_stop),
        }
        for name, value in result.items():
            self.metrics.add_metric(name, value)
        self.rollout_buffer.clear()
        return result

    def _actor_eval_on_fixed_capture_states(self) -> dict[str, float]:
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        try:
            actions = [
                self._actor_raw_action(
                    self.eval_env.reset(episode_mode="capture_vertical", seed=seed)
                )
                for seed in self.ACTOR_EVAL_CAPTURE_SEEDS
            ]
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
        actions_array = np.asarray(actions, dtype=float)
        return {
            "actor_eval_mean_action": float(np.mean(actions_array)),
            "actor_eval_min_action": float(np.min(actions_array)),
            "actor_eval_max_action": float(np.max(actions_array)),
            "actor_eval_saturation_fraction": float(
                np.mean(np.abs(actions_array) >= 0.95 * self.max_action)
            ),
        }

    def _select_episode_mode(self, episode: int) -> tuple[str, dict[str, float]]:
        probabilities = self._episode_mode_probabilities(episode)
        modes = list(self.EPISODE_MODES)
        weights = [probabilities[mode] for mode in modes]
        return random.choices(modes, weights=weights, k=1)[0], probabilities

    def _episode_mode_probabilities(self, episode: int) -> dict[str, float]:
        base_probabilities = dict(self.config["episode_mode_probabilities"])
        if not self.config["adaptive_curriculum_enabled"]:
            return base_probabilities
        if episode < int(self.config["curriculum_start_episode"]):
            return base_probabilities
        window = int(self.config["curriculum_window"])
        mode_scores = {
            "down_to_up": self._recent_metric_mean("down_to_up_final_hold", window),
            "capture_vertical": self._recent_metric_mean("capture_vertical_final_hold", window),
            "fold_to_up": self._recent_metric_mean("fold_to_up_balanced_hold", window),
            "up_to_fold": self._recent_metric_mean("up_to_fold_balanced_hold", window),
        }
        raw_probabilities = {
            mode: base_probabilities[mode] * max(0.05, 1.0 - mode_scores[mode])
            for mode in self.EPISODE_MODES
        }
        return self._bounded_mode_probabilities(
            self._normalized_mode_probabilities(raw_probabilities)
        )

    def _recent_metric_mean(self, metric_name: str, window: int) -> float:
        values = self.metrics.metrics.get(metric_name, [])
        return float(np.mean(values[-window:])) if values else 0.0

    def _normalized_mode_probabilities(
        self, probabilities: Mapping[str, float]
    ) -> dict[str, float]:
        if set(probabilities) != set(self.EPISODE_MODES):
            raise ValueError(f"probabilities must define exactly {self.EPISODE_MODES}")
        values = {mode: float(probabilities[mode]) for mode in self.EPISODE_MODES}
        if not all(np.isfinite(value) and value >= 0.0 for value in values.values()):
            raise ValueError("probabilities must be finite and nonnegative")
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("probabilities must have a positive sum")
        return {mode: value / total for mode, value in values.items()}

    def _bounded_mode_probabilities(
        self, probabilities: Mapping[str, float]
    ) -> dict[str, float]:
        minimums = self.config["curriculum_min_probabilities"]
        maximums = self.config["curriculum_max_probabilities"]
        bounded = {
            mode: min(float(maximums[mode]), max(float(minimums[mode]), probabilities[mode]))
            for mode in self.EPISODE_MODES
        }
        for _ in range(len(self.EPISODE_MODES) * 2):
            difference = 1.0 - sum(bounded.values())
            if abs(difference) < 1e-9:
                break
            adjustable = [
                mode
                for mode in self.EPISODE_MODES
                if (
                    bounded[mode] < float(maximums[mode])
                    if difference > 0.0
                    else bounded[mode] > float(minimums[mode])
                )
            ]
            if not adjustable:
                raise ValueError("curriculum probability bounds cannot produce a unit sum")
            share = difference / len(adjustable)
            for mode in adjustable:
                bounded[mode] = min(
                    float(maximums[mode]),
                    max(float(minimums[mode]), bounded[mode] + share),
                )
        if not np.isclose(sum(bounded.values()), 1.0, atol=1e-9):
            raise ValueError("curriculum bounds failed to produce a unit sum")
        return bounded

    def _should_render_episode(self, episode: int) -> bool:
        if not self.config["render_training"]:
            return False
        if episode == 0 and self.config["render_first_episode"]:
            return True
        frequency = int(self.config["render_every_episodes"])
        return frequency > 0 and episode % frequency == 0

    def train(self) -> None:
        try:
            for episode in range(int(self.config["num_episodes"])):
                summary = self.collect_rollout(episode)
                ppo_metrics = self.update_ppo()
                summary.update(self._actor_eval_on_fixed_capture_states())
                self._record_episode_metrics(summary)
                if episode % 10 == 0:
                    recover_log = ""
                    if summary["capture_drop_recovered"]:
                        recover_log = f" | recover@{summary['capture_drop_recovery_step']}"
                    print(
                        f"Episode {episode:5d} | reward={summary['episode_reward']:8.2f} | "
                        f"len={summary['episode_length']:4d} | dir={summary['transition_direction']:<16s} | "
                        f"hold={summary['balanced_hold']:4.2f} | peak={summary['peak_target_score']:4.2f} | "
                        f"value={ppo_metrics['value_loss']:.4f} | "
                        f"term={summary['termination_reason']}{recover_log}"
                    )
                if episode % self.plot_frequency == self.plot_frequency - 1:
                    self.metrics.generate_all_plots()
        except KeyboardInterrupt:
            completed = len(self.metrics.metrics["episode_reward"])
            self.save_models(MODEL_SAVE_PREFIX, episode=completed, interrupted=True)
            print(f"\nKeyboardInterrupt: saved interrupted model at episode {completed}")
            raise

        completed = len(self.metrics.metrics["episode_reward"])
        self.save_models(MODEL_SAVE_PREFIX, episode=completed, interrupted=False)
        print(f"Training finished: saved model at episode {completed}")

    def _record_episode_metrics(self, summary: Mapping[str, Any]) -> None:
        scalar_keys = (
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
            "actor_eval_mean_action",
            "actor_eval_min_action",
            "actor_eval_max_action",
            "actor_eval_saturation_fraction",
        )
        for key in scalar_keys:
            self.metrics.add_metric(key, summary[key])
        for component_name, values in summary["reward_components"].items():
            if values:
                self.metrics.add_metric(component_name, float(np.mean(values)))
        for direction in self.EPISODE_MODES:
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

    def save_models(self, path: str, episode: int | None = None, interrupted: bool = False) -> None:
        torch.save(self.actor_model.state_dict(), path + "_actor.pth")
        torch.save(self.critic_model.state_dict(), path + "_critic.pth")
        torch.save(self.actor_optimizer.state_dict(), path + "_actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(), path + "_critic_optimizer.pth")
        metadata = {
            "algorithm": "ppo",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "num_nodes": self.config["num_nodes"],
            "episode": episode,
            "total_updates": self.total_updates,
            "total_env_steps": self.total_env_steps,
            "interrupted": interrupted,
            "hidden_dim": self.config["hidden_dim"],
        }
        with open(path + "_metadata.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def load_models(self) -> None:
        metadata_path = f"{MODEL_LOAD_PREFIX}_metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"load_models=True but saved model metadata is missing: {metadata_path}"
            )
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        expected = {
            "algorithm": "ppo",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.config["hidden_dim"],
        }
        missing_keys = sorted(set(expected) - set(metadata))
        if missing_keys:
            raise ValueError(f"saved model metadata is missing required keys: {missing_keys}")
        for key, expected_value in expected.items():
            if metadata[key] != expected_value:
                raise ValueError(
                    f"incompatible saved model: {key} is {metadata[key]!r}, "
                    f"expected {expected_value!r}"
                )

        paths = {
            "actor": f"{MODEL_LOAD_PREFIX}_actor.pth",
            "critic": f"{MODEL_LOAD_PREFIX}_critic.pth",
            "actor optimizer": f"{MODEL_LOAD_PREFIX}_actor_optimizer.pth",
            "critic optimizer": f"{MODEL_LOAD_PREFIX}_critic_optimizer.pth",
        }
        missing_files = [path for path in paths.values() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"saved model files are missing: {missing_files}")
        self.actor_model.load_state_dict(torch.load(paths["actor"], weights_only=True))
        self.critic_model.load_state_dict(torch.load(paths["critic"], weights_only=True))
        self.actor_optimizer.load_state_dict(
            torch.load(paths["actor optimizer"], weights_only=True)
        )
        self.critic_optimizer.load_state_dict(
            torch.load(paths["critic optimizer"], weights_only=True)
        )
        self.total_updates = int(metadata.get("total_updates", 0))
        self.total_env_steps = int(metadata.get("total_env_steps", 0))


if __name__ == "__main__":
    PendulumTrainer(config).train()
