import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import config  # noqa: E402
from model import TriplePendulumActor  # noqa: E402
from reward import RewardManager  # noqa: E402
from tp_env import TriplePendulumEnv  # noqa: E402


MODES = ("down_to_up", "capture_vertical", "fold_to_up", "up_to_fold")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_actor(env, checkpoint_path):
    initial_state = env.reset(episode_mode="up_to_fold")
    actor = TriplePendulumActor(
        state_dim=len(initial_state),
        action_dim=1,
        hidden_dim=config["hidden_dim"],
        max_action=config["max_action"],
    )
    try:
        actor.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot load {checkpoint_path}: checkpoint is incompatible with the current "
            f"2-node observation state_dim={len(initial_state)}. Train a new checkpoint."
        ) from exc
    actor.eval()
    return actor


def select_action(actor, state, max_action, noise_std):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = float(actor(state_tensor).cpu().numpy()[0, 0])
    if noise_std > 0.0:
        action += float(np.random.normal(0.0, noise_std))
    return float(np.clip(action, -max_action, max_action))


def physical_metrics(env):
    physical_state = env.get_physical_state()
    x, _q1, _q2, _x_dot, u1, u2, _force = physical_state[:7]
    return {
        "cart_x": float(x),
        "angular_speed": float(abs(u1) + abs(u2)),
    }


def max_streak(values):
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_rows(prefix, rows):
    if not rows:
        return {
            f"{prefix}_peak_target_score": 0.0,
            f"{prefix}_peak_effective_target_score": 0.0,
            f"{prefix}_final_hold": 0.0,
            f"{prefix}_overall_hold": 0.0,
            f"{prefix}_near_target_fraction": 0.0,
            f"{prefix}_near_fast_fraction": 0.0,
            f"{prefix}_mean_angular_speed_near": 0.0,
            f"{prefix}_max_in_target_streak": 0,
        }

    target_scores = np.array([row["target_score"] for row in rows], dtype=float)
    effective_scores = np.array([row["effective_target_score"] for row in rows], dtype=float)
    in_target = np.array([row["in_target"] for row in rows], dtype=float)
    angular_speeds = np.array([row["angular_speed"] for row in rows], dtype=float)
    capture_threshold = config["swing_up_capture_score_threshold"]
    near_mask = target_scores >= capture_threshold
    overspeed_limit = config["capture_allowed_angular_speed"]
    near_fast_mask = near_mask & (angular_speeds > overspeed_limit)
    final_window = max(1, int(0.2 * len(in_target)))
    return {
        f"{prefix}_peak_target_score": float(np.max(target_scores)),
        f"{prefix}_peak_effective_target_score": float(np.max(effective_scores)),
        f"{prefix}_final_hold": float(np.mean(in_target[-final_window:])),
        f"{prefix}_overall_hold": float(np.mean(in_target)),
        f"{prefix}_near_target_fraction": float(np.mean(near_mask)),
        f"{prefix}_near_fast_fraction": float(np.mean(near_fast_mask)),
        f"{prefix}_mean_angular_speed_near": float(np.mean(angular_speeds[near_mask])) if np.any(near_mask) else 0.0,
        f"{prefix}_max_in_target_streak": int(max_streak(in_target > 0.5)),
    }


def summarize_episode(mode, episode_index, step_rows, total_reward, termination_reason):
    target_scores = np.array([row["target_score"] for row in step_rows], dtype=float)
    effective_scores = np.array([row["effective_target_score"] for row in step_rows], dtype=float)
    in_target = np.array([row["in_target"] for row in step_rows], dtype=float)
    angular_speeds = np.array([row["angular_speed"] for row in step_rows], dtype=float)
    actions = np.array([row["action"] for row in step_rows], dtype=float)
    cart_x = np.array([row["cart_x"] for row in step_rows], dtype=float)
    rewards = np.array([row["reward"] for row in step_rows], dtype=float)

    capture_threshold = config["swing_up_capture_score_threshold"]
    near_mask = target_scores >= capture_threshold
    first_near_indices = np.flatnonzero(near_mask)
    first_near_step = int(first_near_indices[0]) if len(first_near_indices) else None
    final_window = max(1, int(0.2 * len(in_target)))
    overspeed_limit = config["capture_allowed_angular_speed"]
    near_fast_mask = near_mask & (angular_speeds > overspeed_limit)

    before_switch_rows = [row for row in step_rows if not row["has_switched"]]
    after_switch_rows = [row for row in step_rows if row["has_switched"]]
    focus_rows = step_rows if mode in ("down_to_up", "capture_vertical") else after_switch_rows

    summary = {
        "mode": mode,
        "episode_index": episode_index,
        "steps": len(step_rows),
        "total_reward": float(total_reward),
        "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "termination_reason": termination_reason,
        "peak_target_score": float(np.max(target_scores)) if len(target_scores) else 0.0,
        "peak_effective_target_score": float(np.max(effective_scores)) if len(effective_scores) else 0.0,
        "mean_effective_target_score": float(np.mean(effective_scores)) if len(effective_scores) else 0.0,
        "final_hold": float(np.mean(in_target[-final_window:])) if len(in_target) else 0.0,
        "overall_hold": float(np.mean(in_target)) if len(in_target) else 0.0,
        "max_in_target_streak": int(max_streak(in_target > 0.5)),
        "first_near_step": first_near_step,
        "near_target_fraction": float(np.mean(near_mask)) if len(near_mask) else 0.0,
        "near_fast_fraction": float(np.mean(near_fast_mask)) if len(near_fast_mask) else 0.0,
        "mean_angular_speed_near": float(np.mean(angular_speeds[near_mask])) if np.any(near_mask) else 0.0,
        "max_angular_speed_near": float(np.max(angular_speeds[near_mask])) if np.any(near_mask) else 0.0,
        "max_abs_cart_x": float(np.max(np.abs(cart_x))) if len(cart_x) else 0.0,
        "mean_abs_action": float(np.mean(np.abs(actions))) if len(actions) else 0.0,
        "saturated_action_fraction": float(np.mean(np.abs(actions) > 0.98 * config["max_action"])) if len(actions) else 0.0,
    }
    summary.update(summarize_rows("pre_switch", before_switch_rows))
    summary.update(summarize_rows("post_switch", after_switch_rows))
    summary.update(summarize_rows("focus", focus_rows))
    return summary


def swing_up_sinus_episode_probability(episode_index):
    start = float(config["swing_up_sinus_episode_probability_start"])
    end = float(config["swing_up_sinus_episode_probability_end"])
    decay_episodes = int(config["swing_up_sinus_episode_decay_episodes"])
    progress = min(1.0, episode_index / decay_episodes)
    return start + (end - start) * progress


def use_sinus_swing_episode(episode_index, seed):
    episode_random = random.Random(seed)
    return episode_random.random() < swing_up_sinus_episode_probability(episode_index)


def run_episode(actor, mode, episode_index, seed, noise_std, use_training_exploration):
    seed_everything(seed)
    reward_manager = RewardManager(config)
    env = TriplePendulumEnv(
        reward_manager=reward_manager,
        render_mode=None,
        num_nodes=config["num_nodes"],
        max_steps=config["max_steps"],
        env_config=config,
    )
    state = env.reset(episode_mode=mode)
    rows = []
    total_reward = 0.0
    termination_reason = "none"
    swing_period = random.uniform(
        config["swing_up_exploration_period_min"],
        config["swing_up_exploration_period_max"],
    )
    swing_phase = random.uniform(0.0, 2.0 * math.pi)
    capture_started = env.capture_started
    sinus_swing_episode = (
        mode == "down_to_up"
        and use_training_exploration
        and use_sinus_swing_episode(episode_index, seed)
    )

    for step_idx in range(config["max_steps"]):
        if sinus_swing_episode and not capture_started:
            action = config["swing_up_exploration_amplitude"] * math.sin(
                2.0 * math.pi * step_idx / swing_period + swing_phase
            )
            if config["swing_up_exploration_noise"] > 0.0:
                action += float(np.random.normal(0.0, config["swing_up_exploration_noise"]))
            action = float(np.clip(action, -config["max_action"], config["max_action"]))
        else:
            action = select_action(actor, state, config["max_action"], noise_std)
        next_state, reward, terminated, truncated, info = env.step(action)
        components = info["reward_components"]
        capture_started = bool(info["capture_started"])
        phys = physical_metrics(env)
        row = {
            "mode": mode,
            "episode_index": episode_index,
            "seed": seed,
            "step": step_idx,
            "phase": int(info["phase"]),
            "switch_step": int(info["switch_step"]),
            "has_switched": bool(env.has_switched),
            "action": float(action),
            "reward": float(reward),
            "target_score": float(components["target_score"]),
            "effective_target_score": float(components["effective_target_score"]),
            "target_error": float(components["target_error"]),
            "in_target": float(components["in_target"]),
            "end_y": float(components["end_y"]),
            "end_x": float(components["end_x"]),
            "energy_score": float(components["energy_score"]),
            "height_score": float(components["height_score"]),
            "cart_safety_score": float(components["cart_safety_score"]),
            "potential_progress": float(components["potential_progress"]),
            "capture_quality": float(components["capture_quality"]),
            "capture_entry_bonus": float(components["capture_entry_bonus"]),
            "hold_bonus": float(components["hold_bonus"]),
            "hold_progress": float(components["hold_progress"]),
            "hold_streak": float(components["hold_streak"]),
            "cart_x": phys["cart_x"],
            "angular_speed": phys["angular_speed"],
            "done": bool(terminated or truncated),
            "termination_reason": info["termination_reason"] or "none",
        }
        rows.append(row)
        total_reward += reward
        state = next_state
        if terminated or truncated:
            termination_reason = info["termination_reason"] or "done"
            break

    return rows, summarize_episode(mode, episode_index, rows, total_reward, termination_reason)


def aggregate_summary(episode_summaries):
    grouped = {}
    for mode in MODES:
        mode_rows = [row for row in episode_summaries if row["mode"] == mode]
        grouped[mode] = {}
        for key in (
            "steps",
            "total_reward",
            "peak_target_score",
            "peak_effective_target_score",
            "final_hold",
            "overall_hold",
            "max_in_target_streak",
            "near_target_fraction",
            "near_fast_fraction",
            "mean_angular_speed_near",
            "max_abs_cart_x",
            "mean_abs_action",
            "saturated_action_fraction",
            "pre_switch_final_hold",
            "post_switch_final_hold",
            "focus_peak_target_score",
            "focus_peak_effective_target_score",
            "focus_final_hold",
            "focus_overall_hold",
            "focus_near_target_fraction",
            "focus_near_fast_fraction",
            "focus_mean_angular_speed_near",
            "focus_max_in_target_streak",
        ):
            values = [row[key] for row in mode_rows]
            grouped[mode][f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            grouped[mode][f"{key}_max"] = float(np.max(values)) if values else 0.0
        grouped[mode]["terminations"] = {
            reason: sum(1 for row in mode_rows if row["termination_reason"] == reason)
            for reason in sorted({row["termination_reason"] for row in mode_rows})
        }
    return grouped


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-mode", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--use-training-exploration", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="results/audits/current_policy")
    parser.add_argument("--checkpoint", default="models/interrupted_actor.pth")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    bootstrap_env = TriplePendulumEnv(
        reward_manager=RewardManager(config),
        render_mode=None,
        num_nodes=config["num_nodes"],
        max_steps=config["max_steps"],
        env_config=config,
    )
    actor = load_actor(bootstrap_env, args.checkpoint)

    all_step_rows = []
    episode_summaries = []
    for mode in MODES:
        for episode_index in range(args.episodes_per_mode):
            seed = args.seed + len(episode_summaries)
            rows, summary = run_episode(
                actor,
                mode,
                episode_index,
                seed,
                args.noise_std,
                args.use_training_exploration,
            )
            all_step_rows.extend(rows)
            episode_summaries.append(summary)
            print(
                f"{mode:10s} ep={episode_index} "
                f"reward={summary['total_reward']:8.2f} "
                f"steps={summary['steps']:4d} "
                f"peak={summary['peak_target_score']:.2f} "
                f"eff_peak={summary['peak_effective_target_score']:.2f} "
                f"focus_peak={summary['focus_peak_target_score']:.2f} "
                f"focus_hold={summary['focus_final_hold']:.2f} "
                f"focus_fast={summary['focus_near_fast_fraction']:.2f} "
                f"term={summary['termination_reason']}"
            )

    aggregate = aggregate_summary(episode_summaries)
    write_csv(out_dir / "steps.csv", all_step_rows)
    write_csv(out_dir / "episodes.csv", episode_summaries)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "episodes_per_mode": args.episodes_per_mode,
                "noise_std": args.noise_std,
                "use_training_exploration": args.use_training_exploration,
                "seed": args.seed,
                "aggregate": aggregate,
                "episodes": episode_summaries,
            },
            file,
            indent=2,
        )
    print(f"Wrote audit files to {out_dir}")


if __name__ == "__main__":
    main()
