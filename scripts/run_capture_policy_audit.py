import argparse
import csv
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from reward import RewardManager
from tp_env import PendulumEnv


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)


def base_config(max_steps):
    cfg = deepcopy(config)
    cfg.update(
        {
            "max_steps": max_steps,
            "load_models": False,
            "render_training": False,
            "render_first_episode": False,
            "episode_mode_probabilities": {
                "down_to_up": 0.0,
                "capture_vertical": 1.0,
                "fold_to_up": 0.0,
                "up_to_fold": 0.0,
            },
            "plot_config": {
                **cfg["plot_config"],
                "enable_plots": False,
            },
        }
    )
    return cfg


def angle_error(angle, target):
    return math.atan2(math.sin(angle - target), math.cos(angle - target))


def clipped(action, max_action):
    return float(np.clip(action, -max_action, max_action))


def policy_action(policy_name, physical_state, step_index, rng, max_action):
    x, q1, q2, x_dot, q1_dot, q2_dot = physical_state[:6]
    upright_error = angle_error(q1, math.pi / 2.0) + angle_error(q2, math.pi / 2.0)
    upright_speed = q1_dot + q2_dot

    if policy_name == "zero_action":
        return 0.0
    if policy_name == "small_random":
        del step_index
        return clipped(rng.normal(0.0, 0.08 * max_action), max_action)
    if policy_name == "pd_upright":
        return clipped(-0.18 * upright_error - 0.035 * upright_speed, max_action)
    if policy_name == "pd_cart_center":
        return clipped(-0.45 * x - 0.16 * x_dot, max_action)
    if policy_name == "pd_upright_plus_cart_center":
        return clipped(
            -0.16 * upright_error
            - 0.03 * upright_speed
            - 0.35 * x
            - 0.12 * x_dot,
            max_action,
        )
    raise ValueError(f"unknown policy: {policy_name}")


def compute_episode_metrics(step_rows, *, max_action):
    if not step_rows:
        raise ValueError("step_rows cannot be empty")

    in_target = np.array([bool(row["in_target"]) for row in step_rows], dtype=bool)
    target_scores = np.array([float(row["target_score"]) for row in step_rows], dtype=float)
    actions = np.array([float(row["action"]) for row in step_rows], dtype=float)
    cart_x = np.array([float(row["cart_x"]) for row in step_rows], dtype=float)
    angular_speed = np.array([float(row["angular_speed"]) for row in step_rows], dtype=float)
    capture_drop_flags = np.array([bool(row["capture_drop"]) for row in step_rows], dtype=bool)
    cart_hit_flags = np.array([bool(row["hit_cart_limit"]) for row in step_rows], dtype=bool)

    final_window = max(1, int(0.2 * len(step_rows)))
    contact_indices = np.flatnonzero(target_scores >= 0.75)
    first_contact_index = int(contact_indices[0]) if len(contact_indices) else -1
    drop_after_contact = bool(
        first_contact_index >= 0
        and np.any(capture_drop_flags[first_contact_index:] | cart_hit_flags[first_contact_index:])
    )
    near_mask = target_scores >= 0.75
    angular_speed_near_target = (
        float(np.mean(angular_speed[near_mask])) if np.any(near_mask) else 0.0
    )

    return {
        "episode_reward": float(np.sum([row["reward"] for row in step_rows])),
        "episode_length": len(step_rows),
        "final_hold": float(np.mean(in_target[-final_window:])),
        "success": float(np.mean(in_target[-final_window:]) > 0.8),
        "peak_target_score": float(np.max(target_scores)),
        "brief_target_contact": float(np.any(near_mask)),
        "sustained_hold": float(np.max(_streak_lengths(in_target)) >= 15),
        "drop_after_contact": float(drop_after_contact),
        "capture_drop": float(np.any(capture_drop_flags)),
        "cart_hit": float(np.any(cart_hit_flags)),
        "action_saturation": float(np.mean(np.abs(actions) >= 0.95 * max_action)),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "max_abs_cart_x": float(np.max(np.abs(cart_x))),
        "angular_speed_near_target": angular_speed_near_target,
        "termination_reason": step_rows[-1]["termination_reason"] or "none",
    }


def _streak_lengths(flags):
    streaks = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    return streaks or [0]


def run_episode(policy_name, env, cfg, seed, episode):
    rng = np.random.default_rng(seed * 10_000 + episode)
    env.reset(episode_mode="capture_vertical", seed=seed * 10_000 + episode)
    max_action = float(cfg["max_action"])
    rows = []
    for step_index in range(cfg["max_steps"]):
        physical_state = env.get_physical_state()
        action = policy_action(policy_name, physical_state, step_index, rng, max_action)
        _obs, reward, terminated, truncated, info = env.step(action)
        components = info["reward_components"]
        rows.append(
            {
                "policy": policy_name,
                "seed": seed,
                "episode": episode,
                "step": step_index,
                "action": action,
                "reward": float(reward),
                "cart_x": float(env.current_state[0]),
                "target_score": float(info["target_score"]),
                "in_target": bool(info["in_target"]),
                "capture_drop": bool(info["capture_drop"]),
                "hit_cart_limit": bool(info["hit_cart_limit"]),
                "angular_speed": float(components["angular_speed"]),
                "termination_reason": info["termination_reason"],
            }
        )
        if terminated or truncated:
            break
    metrics = compute_episode_metrics(rows, max_action=max_action)
    metrics.update({"policy": policy_name, "seed": seed, "episode": episode})
    return rows, metrics


def summarize_episode_metrics(episode_rows):
    grouped = {}
    for row in episode_rows:
        grouped.setdefault(row["policy"], []).append(row)

    summaries = []
    for policy, rows in sorted(grouped.items()):
        summaries.append(
            {
                "policy": policy,
                "episodes": len(rows),
                "final_hold": float(np.mean([row["final_hold"] for row in rows])),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "drop_rate": float(np.mean([row["capture_drop"] for row in rows])),
                "cart_hit_rate": float(np.mean([row["cart_hit"] for row in rows])),
                "action_saturation": float(np.mean([row["action_saturation"] for row in rows])),
                "max_abs_cart_x": float(np.mean([row["max_abs_cart_x"] for row in rows])),
                "angular_speed_near_target": float(
                    np.mean([row["angular_speed_near_target"] for row in rows])
                ),
                "brief_target_contact_rate": float(
                    np.mean([row["brief_target_contact"] for row in rows])
                ),
                "sustained_hold_rate": float(np.mean([row["sustained_hold"] for row in rows])),
                "drop_after_contact_rate": float(
                    np.mean([row["drop_after_contact"] for row in rows])
                ),
                "reward_mean": float(np.mean([row["episode_reward"] for row in rows])),
                "episode_length_mean": float(np.mean([row["episode_length"] for row in rows])),
            }
        )
    return summaries


def summarize_by_policy_seed(episode_rows):
    grouped = {}
    for row in episode_rows:
        grouped.setdefault((row["policy"], row["seed"]), []).append(row)

    summaries = []
    for (policy, seed), rows in sorted(grouped.items()):
        summaries.append(
            {
                "policy": policy,
                "seed": seed,
                "episodes": len(rows),
                "final_hold": float(np.mean([row["final_hold"] for row in rows])),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "drop_rate": float(np.mean([row["capture_drop"] for row in rows])),
                "cart_hit_rate": float(np.mean([row["cart_hit"] for row in rows])),
                "action_saturation": float(np.mean([row["action_saturation"] for row in rows])),
                "max_abs_cart_x": float(np.mean([row["max_abs_cart_x"] for row in rows])),
                "angular_speed_near_target": float(
                    np.mean([row["angular_speed_near_target"] for row in rows])
                ),
                "brief_target_contact_rate": float(
                    np.mean([row["brief_target_contact"] for row in rows])
                ),
                "sustained_hold_rate": float(np.mean([row["sustained_hold"] for row in rows])),
                "drop_after_contact_rate": float(
                    np.mean([row["drop_after_contact"] for row in rows])
                ),
                "reward_mean": float(np.mean([row["episode_reward"] for row in rows])),
                "episode_length_mean": float(np.mean([row["episode_length"] for row in rows])),
            }
        )
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "zero_action",
            "small_random",
            "pd_upright",
            "pd_cart_center",
            "pd_upright_plus_cart_center",
        ],
    )
    parser.add_argument("--out-dir", default="results/capture_policy_audit")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    cfg = base_config(args.max_steps)
    all_step_rows = []
    episode_rows = []
    for policy in args.policies:
        for seed in args.seeds:
            seeded(seed)
            env = PendulumEnv(
                reward_manager=RewardManager(cfg),
                render_mode=None,
                num_nodes=cfg["num_nodes"],
                max_steps=cfg["max_steps"],
                env_config=cfg,
            )
            for episode in range(args.episodes):
                step_rows, metrics = run_episode(policy, env, cfg, seed, episode)
                all_step_rows.extend(step_rows)
                episode_rows.append(metrics)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "steps.csv", all_step_rows)
    write_csv(out_dir / "episodes.csv", episode_rows)
    write_csv(out_dir / "seed_summary.csv", summarize_by_policy_seed(episode_rows))
    write_csv(out_dir / "summary.csv", summarize_episode_metrics(episode_rows))
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "seeds": args.seeds,
                "policies": args.policies,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
