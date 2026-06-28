import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config  # noqa: E402
from train import PendulumTrainer  # noqa: E402


def seeded(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def base_config(episodes, max_steps):
    cfg = deepcopy(config)
    cfg.update(
        {
            "num_episodes": episodes,
            "max_steps": max_steps,
            "load_models": False,
            "render_training": False,
            "render_first_episode": False,
            "plot_config": {**cfg["plot_config"], "enable_plots": False},
        }
    )
    return cfg


def variants(episodes, max_steps):
    mixed = base_config(episodes, max_steps)
    capture_only = deepcopy(mixed)
    capture_only["episode_mode_probabilities"] = {
        "down_to_up": 0.0,
        "capture_vertical": 1.0,
        "fold_to_up": 0.0,
        "up_to_fold": 0.0,
    }
    capture_easy = deepcopy(capture_only)
    capture_easy.update(
        capture_angle_noise=0.02,
        capture_cart_velocity_noise=0.005,
        capture_angular_velocity_noise=0.05,
    )
    capture_soft_drop = deepcopy(capture_only)
    capture_soft_drop["capture_drop_base_penalty"] = -10.0
    capture_soft_drop["capture_drop_remaining_penalty"] = -10.0
    down_only = deepcopy(mixed)
    down_only["episode_mode_probabilities"] = {
        "down_to_up": 1.0,
        "capture_vertical": 0.0,
        "fold_to_up": 0.0,
        "up_to_fold": 0.0,
    }
    return {
        "mixed": mixed,
        "capture_only": capture_only,
        "capture_easy": capture_easy,
        "capture_soft_drop": capture_soft_drop,
        "down_only": down_only,
    }


def mean_component(summary, name):
    values = summary["reward_components"].get(name, [0.0])
    return float(np.mean(values)) if values else 0.0


def run_variant(name, cfg, seed):
    seeded(seed)
    trainer = PendulumTrainer(cfg)
    rows = []
    for episode in range(cfg["num_episodes"]):
        summary = trainer.collect_rollout(episode)
        optimization = trainer.update_ppo()
        summary.update(trainer._actor_eval_on_fixed_capture_states())
        trainer._record_episode_metrics(summary)
        rows.append(
            {
                "variant": name,
                "seed": seed,
                "episode": episode,
                "episode_reward": summary["episode_reward"],
                "episode_length": summary["episode_length"],
                "transition_direction": summary["transition_direction"],
                "final_hold": summary["final_hold"],
                "balanced_hold": summary["balanced_hold"],
                "episode_success": summary["episode_success"],
                "peak_target_score": summary["peak_target_score"],
                "termination_reason": summary["termination_reason"],
                "capture_drop_penalty": mean_component(summary, "capture_drop_penalty"),
                "cart_limit_step_penalty": mean_component(summary, "cart_limit_step_penalty"),
                "action_abs_mean": summary["action_abs_mean"],
                "policy_loss": optimization["policy_loss"],
                "value_loss": optimization["value_loss"],
                "entropy": optimization["entropy"],
                "approx_kl": optimization["approx_kl"],
            }
        )
    return rows


def summarize(rows, tail):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["seed"]), []).append(row)
    summaries = []
    for (variant, seed), group in sorted(grouped.items()):
        group_tail = group[-tail:]
        target_rows = [
            row for row in group_tail if row["transition_direction"] == "capture_vertical"
        ] or group_tail
        summaries.append(
            {
                "variant": variant,
                "seed": seed,
                "episodes": len(group),
                "tail_reward_mean": float(np.mean([row["episode_reward"] for row in group_tail])),
                "tail_final_hold_mean": float(np.mean([row["final_hold"] for row in target_rows])),
                "tail_success_rate": float(np.mean([row["episode_success"] for row in target_rows])),
                "tail_peak_mean": float(np.mean([row["peak_target_score"] for row in target_rows])),
                "tail_capture_drop_rate": float(
                    np.mean([row["capture_drop_penalty"] < 0.0 for row in group_tail])
                ),
                "tail_action_abs_mean": float(np.mean([row["action_abs_mean"] for row in group_tail])),
                "tail_approx_kl_mean": float(np.mean([row["approx_kl"] for row in group_tail])),
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
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--tail", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--out-dir", default="results/ablation")
    args = parser.parse_args()

    all_variants = variants(args.episodes, args.max_steps)
    selected_names = args.variants or list(all_variants)
    rows = []
    for variant_name in selected_names:
        if variant_name not in all_variants:
            raise ValueError(f"unknown variant {variant_name!r}; choices={sorted(all_variants)}")
        for seed in args.seeds:
            rows.extend(run_variant(variant_name, deepcopy(all_variants[variant_name]), seed))

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "episodes.csv", rows)
    summary_rows = summarize(rows, args.tail)
    write_csv(out_dir / "summary.csv", summary_rows)
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "tail": args.tail,
                "seeds": args.seeds,
                "variants": selected_names,
            },
            indent=2,
        )
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
