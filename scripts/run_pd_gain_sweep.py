import argparse
import csv
import itertools
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward import RewardManager
from scripts.run_capture_policy_audit import (
    angle_error,
    base_config,
    clipped,
    compute_episode_metrics,
)
from tp_env import PendulumEnv


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)


def pd_action(physical_state, *, kp_angle, kd_angle, kp_cart, kd_cart, max_action):
    x, q1, q2, x_dot, q1_dot, q2_dot = physical_state[:6]
    upright_error = angle_error(q1, math.pi / 2.0) + angle_error(q2, math.pi / 2.0)
    upright_speed = q1_dot + q2_dot
    action = (
        -kp_angle * upright_error
        - kd_angle * upright_speed
        - kp_cart * x
        - kd_cart * x_dot
    )
    return clipped(action, max_action)


def run_episode(env, cfg, seed, episode, gains):
    env.reset(episode_mode="capture_vertical", seed=seed * 10_000 + episode)
    max_action = float(cfg["max_action"])
    rows = []
    for step_index in range(cfg["max_steps"]):
        del step_index
        action = pd_action(env.get_physical_state(), max_action=max_action, **gains)
        _obs, reward, terminated, truncated, info = env.step(action)
        components = info["reward_components"]
        rows.append(
            {
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
    return compute_episode_metrics(rows, max_action=max_action)


def aggregate(rows, key_fields):
    grouped = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in sorted(grouped.items()):
        summary = dict(zip(key_fields, key))
        summary.update(
            {
                "episodes": len(group),
                "final_hold": float(np.mean([row["final_hold"] for row in group])),
                "success_rate": float(np.mean([row["success"] for row in group])),
                "drop_rate": float(np.mean([row["capture_drop"] for row in group])),
                "cart_hit_rate": float(np.mean([row["cart_hit"] for row in group])),
                "action_saturation": float(np.mean([row["action_saturation"] for row in group])),
                "max_abs_cart_x": float(np.mean([row["max_abs_cart_x"] for row in group])),
                "angular_speed_near_target": float(
                    np.mean([row["angular_speed_near_target"] for row in group])
                ),
                "brief_target_contact_rate": float(
                    np.mean([row["brief_target_contact"] for row in group])
                ),
                "sustained_hold_rate": float(np.mean([row["sustained_hold"] for row in group])),
                "drop_after_contact_rate": float(
                    np.mean([row["drop_after_contact"] for row in group])
                ),
                "mean_abs_action": float(np.mean([row["action_abs_mean"] for row in group])),
                "reward_mean": float(np.mean([row["episode_reward"] for row in group])),
                "episode_length_mean": float(np.mean([row["episode_length"] for row in group])),
            }
        )
        summaries.append(summary)
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--out-dir", default="results/pd_gain_sweep")
    args = parser.parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    kp_angles = [0.5, 1.0, 2.0, 4.0, 8.0]
    kd_angles = [0.05, 0.1, 0.2, 0.4]
    kp_carts = [0.05, 0.1, 0.2, 0.4]
    kd_carts = [0.02, 0.05, 0.1]
    gain_rows = [
        {
            "kp_angle": kp_angle,
            "kd_angle": kd_angle,
            "kp_cart": kp_cart,
            "kd_cart": kd_cart,
        }
        for kp_angle, kd_angle, kp_cart, kd_cart in itertools.product(
            kp_angles, kd_angles, kp_carts, kd_carts
        )
    ]

    cfg = base_config(args.max_steps)
    episode_rows = []
    for gain_index, gains in enumerate(gain_rows):
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
                metrics = run_episode(env, cfg, seed, episode, gains)
                episode_rows.append(
                    {
                        "gain_index": gain_index,
                        **gains,
                        "seed": seed,
                        "episode": episode,
                        **metrics,
                    }
                )

    out_dir = Path(args.out_dir)
    summary = aggregate(episode_rows, ["gain_index", "kp_angle", "kd_angle", "kp_cart", "kd_cart"])
    seed_summary = aggregate(
        episode_rows,
        ["gain_index", "kp_angle", "kd_angle", "kp_cart", "kd_cart", "seed"],
    )
    summary.sort(key=lambda row: (row["success_rate"], row["final_hold"]), reverse=True)
    write_csv(out_dir / "episodes.csv", episode_rows)
    write_csv(out_dir / "seed_summary.csv", seed_summary)
    write_csv(out_dir / "summary.csv", summary)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "seeds": args.seeds,
                "kp_angle": kp_angles,
                "kd_angle": kd_angles,
                "kp_cart": kp_carts,
                "kd_cart": kd_carts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
