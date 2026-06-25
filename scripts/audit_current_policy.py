import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from model import TriplePendulumActor
from reward import RewardManager
from tp_env import TriplePendulumEnv


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
        max_action=config.get("max_action", 0.5),
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
    capture_threshold = config.get("swing_up_capture_score_threshold", 0.75)
    near_mask = target_scores >= capture_threshold
    overspeed_limit = config.get("capture_allowed_angular_speed", 1.5)
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

    capture_threshold = config.get("swing_up_capture_score_threshold", 0.75)
    near_mask = target_scores >= capture_threshold
    first_near_indices = np.flatnonzero(near_mask)
    first_near_step = int(first_near_indices[0]) if len(first_near_indices) else None
    final_window = max(1, int(0.2 * len(in_target)))
    overspeed_limit = config.get("capture_allowed_angular_speed", 1.5)
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
        "saturated_action_fraction": float(np.mean(np.abs(actions) > 0.98 * config.get("max_action", 0.5))) if len(actions) else 0.0,
    }
    summary.update(summarize_rows("pre_switch", before_switch_rows))
    summary.update(summarize_rows("post_switch", after_switch_rows))
    summary.update(summarize_rows("focus", focus_rows))
    return summary


def run_episode(actor, mode, episode_index, seed, noise_std, use_training_exploration):
    seed_everything(seed)
    reward_manager = RewardManager()
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
        config.get("swing_up_exploration_period_min", 60),
        config.get("swing_up_exploration_period_max", 120),
    )
    swing_phase = random.uniform(0.0, 2.0 * math.pi)
    capture_started = False

    for step_idx in range(config["max_steps"]):
        action = select_action(actor, state, config.get("max_action", 0.5), noise_std)
        if mode == "down_to_up" and use_training_exploration and not capture_started:
            swing_action = config.get("swing_up_exploration_amplitude", 0.45) * math.sin(
                2.0 * math.pi * step_idx / max(1.0, swing_period) + swing_phase
            )
            action = float(np.clip(action + swing_action, -config.get("max_action", 0.5), config.get("max_action", 0.5)))
        next_state, reward, done, info = env.step(action)
        components = info.get("reward_components", {})
        if (
            mode == "down_to_up"
            and not capture_started
            and float(info.get("target_score", 0.0)) >= config.get("swing_up_capture_score_threshold", 0.75)
        ):
            capture_started = True
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
            "target_score": float(components.get("target_score", 0.0)),
            "effective_target_score": float(components.get("effective_target_score", components.get("target_score", 0.0))),
            "target_error": float(components.get("target_error", 0.0)),
            "in_target": float(components.get("in_target", 0.0)),
            "end_y": float(components.get("end_y", 0.0)),
            "end_x": float(components.get("end_x", 0.0)),
            "velocity_penalty": float(components.get("velocity_penalty", 0.0)),
            "action_penalty": float(components.get("action_penalty", 0.0)),
            "cart_penalty": float(components.get("cart_penalty", 0.0)),
            "capture_overspeed_penalty": float(components.get("capture_overspeed_penalty", 0.0)),
            "capture_height_penalty": float(components.get("capture_height_penalty", 0.0)),
            "capture_lost_penalty": float(components.get("capture_lost_penalty", 0.0)),
            "capture_rest_penalty": float(components.get("capture_rest_penalty", 0.0)),
            "capture_velocity_bonus": float(components.get("capture_velocity_bonus", 0.0)),
            "swing_up_velocity_bonus": float(components.get("swing_up_velocity_bonus", 0.0)),
            "swing_up_height_progress_bonus": float(components.get("swing_up_height_progress_bonus", 0.0)),
            "swing_up_score_progress_bonus": float(components.get("swing_up_score_progress_bonus", 0.0)),
            "target_shaping_reward": float(components.get("target_shaping_reward", 0.0)),
            "target_entry_bonus": float(components.get("target_entry_bonus", 0.0)),
            "hold_progress_bonus": float(components.get("hold_progress_bonus", 0.0)),
            "hold_progress": float(components.get("hold_progress", 0.0)),
            "hold_streak": float(components.get("hold_streak", 0.0)),
            "cart_x": phys["cart_x"],
            "angular_speed": phys["angular_speed"],
            "done": bool(done),
            "termination_reason": info.get("termination_reason") or "none",
        }
        rows.append(row)
        total_reward += reward
        state = next_state
        if done:
            termination_reason = info.get("termination_reason") or "done"
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
    parser.add_argument("--checkpoint", default="models/checkpoint_actor.pth")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    bootstrap_env = TriplePendulumEnv(
        reward_manager=RewardManager(),
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
