import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config, validate_config
from reward import RewardManager
from tp_env import PendulumEnv
from train import PendulumTrainer


CONSTANT_POLICIES = {
    "zero_action": 0.0,
    "constant_+0.02": 0.02,
    "constant_-0.02": -0.02,
    "constant_+0.05": 0.05,
    "constant_-0.05": -0.05,
}


def audit_config(num_steps):
    cfg = deepcopy(config)
    cfg["max_steps"] = num_steps
    cfg["load_models"] = False
    cfg["render_training"] = False
    cfg["render_first_episode"] = False
    cfg["plot_config"] = {**cfg["plot_config"], "enable_plots": False}
    return validate_config(cfg)


def clipped(action, max_action):
    return float(np.clip(action, -max_action, max_action))


def resolve_action(policy_name, state, trainer, max_action):
    if policy_name in CONSTANT_POLICIES:
        return clipped(CONSTANT_POLICIES[policy_name], max_action)
    if policy_name == "actor_deterministic":
        return trainer.select_action(state, noise_std=0.0)
    if policy_name == "actor_plus_noise_0.10":
        return trainer.select_action(state, noise_std=0.10)
    raise ValueError(f"unknown policy: {policy_name}")


def rollout_policy(policy_name, env, trainer, cfg, seed, num_steps):
    env.reset(episode_mode="capture_vertical", seed=seed)
    max_action = float(cfg["max_action"])
    rows = []
    capture_drop_step = None
    state = env.get_state(action=0.0, phase=env.current_phase)
    for step_index in range(num_steps):
        action = resolve_action(policy_name, state, trainer, max_action)
        state, reward, terminated, truncated, info = env.step(action)
        physical_state = env.get_physical_state()
        components = info["reward_components"]
        if info["capture_drop"] and capture_drop_step is None:
            capture_drop_step = step_index
        rows.append(
            {
                "policy": policy_name,
                "seed": seed,
                "step": step_index,
                "action": action,
                "applied_force": float(env.applied_force),
                "target_score": float(info["target_score"]),
                "end_y": float(components["end_y"]),
                "q1": float(physical_state[1]),
                "q2": float(physical_state[2]),
                "q1_dot": float(physical_state[4]),
                "q2_dot": float(physical_state[5]),
                "reward": float(reward),
                "capture_drop": bool(info["capture_drop"]),
                "termination_reason": info["termination_reason"],
            }
        )
        if terminated or truncated:
            break
    return rows, capture_drop_step


def summarize_policy(rows, capture_drop_step):
    if not rows:
        raise ValueError("rows cannot be empty")
    target_scores = np.array([row["target_score"] for row in rows], dtype=float)
    end_y_values = np.array([row["end_y"] for row in rows], dtype=float)
    actions = np.array([row["action"] for row in rows], dtype=float)
    return {
        "policy": rows[0]["policy"],
        "seed": rows[0]["seed"],
        "steps_logged": len(rows),
        "episode_length": len(rows),
        "capture_drop_step": capture_drop_step,
        "terminated_early": float(rows[-1]["termination_reason"] is not None),
        "termination_reason": rows[-1]["termination_reason"] or "none",
        "target_score_min": float(np.min(target_scores)),
        "target_score_final": float(target_scores[-1]),
        "target_score_peak": float(np.max(target_scores)),
        "end_y_min": float(np.min(end_y_values)),
        "end_y_final": float(end_y_values[-1]),
        "action_abs_mean": float(np.mean(np.abs(actions))),
        "episode_reward": float(np.sum([row["reward"] for row in rows])),
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit capture_vertical with fixed actions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("results/capture_constant_action_audit"))
    arguments = parser.parse_args()

    cfg = audit_config(arguments.num_steps)
    env = PendulumEnv(reward_manager=RewardManager(cfg), env_config=cfg)
    trainer = PendulumTrainer(cfg)

    policy_names = list(CONSTANT_POLICIES) + ["actor_deterministic", "actor_plus_noise_0.10"]
    step_rows = []
    summary_rows = []
    for policy_name in policy_names:
        rows, capture_drop_step = rollout_policy(
            policy_name,
            env,
            trainer,
            cfg,
            arguments.seed,
            arguments.num_steps,
        )
        step_rows.extend(rows)
        summary = summarize_policy(rows, capture_drop_step)
        summary_rows.append(summary)

    step_fieldnames = [
        "policy", "seed", "step", "action", "applied_force", "target_score",
        "end_y", "q1", "q2", "q1_dot", "q2_dot", "reward", "capture_drop",
        "termination_reason",
    ]
    summary_fieldnames = [
        "policy", "seed", "steps_logged", "episode_length", "capture_drop_step",
        "terminated_early", "termination_reason", "target_score_min",
        "target_score_final", "target_score_peak", "end_y_min", "end_y_final",
        "action_abs_mean", "episode_reward",
    ]
    output_dir = arguments.output_dir
    write_csv(output_dir / "steps.csv", step_rows, step_fieldnames)
    write_csv(output_dir / "summary.csv", summary_rows, summary_fieldnames)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "seed": arguments.seed,
                "num_steps": arguments.num_steps,
                "policies": policy_names,
                "capture_drop_grace_steps": cfg["capture_drop_grace_steps"],
                "capture_drop_target_score_threshold": cfg["capture_drop_target_score_threshold"],
            },
            config_file,
            indent=2,
        )

    print(f"wrote {len(step_rows)} step rows and {len(summary_rows)} summaries to {output_dir}")
    for summary in summary_rows:
        drop_text = "none" if summary["capture_drop_step"] is None else str(summary["capture_drop_step"])
        print(
            f"{summary['policy']:<24s} "
            f"len={summary['episode_length']:2d} "
            f"drop_step={drop_text:>4s} "
            f"target_peak={summary['target_score_peak']:.3f} "
            f"target_final={summary['target_score_final']:.3f} "
            f"end_y_final={summary['end_y_final']:+.3f} "
            f"term={summary['termination_reason']}"
        )


if __name__ == "__main__":
    main()
