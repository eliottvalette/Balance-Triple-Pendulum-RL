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

from config import config
from train import PendulumTrainer


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
            "learning_starts": min(cfg["learning_starts"], max(1, max_steps * 2)),
            "plot_config": {
                **cfg["plot_config"],
                "enable_plots": False,
            },
        }
    )
    return cfg


def variants(episodes, max_steps):
    current = base_config(episodes, max_steps)

    capture_only = deepcopy(current)
    capture_only["episode_mode_probabilities"] = {
        "down_to_up": 0.0,
        "capture_vertical": 1.0,
        "fold_to_up": 0.0,
        "up_to_fold": 0.0,
    }

    capture_easy = deepcopy(capture_only)
    capture_easy.update(
        {
            "capture_angle_noise": 0.02,
            "capture_cart_velocity_noise": 0.005,
            "capture_angular_velocity_noise": 0.05,
        }
    )

    capture_no_drop = deepcopy(capture_only)
    capture_no_drop["capture_drop_penalty"] = -1e-6

    capture_actor_gate_safe = deepcopy(capture_only)
    capture_actor_gate_safe["min_near_capture_transitions_for_actor_update"] = 1000

    capture_late_learning = deepcopy(capture_only)
    capture_late_learning.update(
        {
            "learning_starts": max(capture_only["learning_starts"], max_steps * 20),
            "min_non_crash_transitions_for_actor_update": max_steps * 20,
        }
    )

    capture_soft_drop = deepcopy(capture_only)
    capture_soft_drop["capture_drop_penalty"] = -10.0

    capture_prefill_safe = deepcopy(capture_only)

    capture_prefill_safe_actor_gate = deepcopy(capture_actor_gate_safe)

    capture_prefill_safe_late = deepcopy(capture_late_learning)

    down_no_sinus = deepcopy(current)
    down_no_sinus.update(
        {
            "episode_mode_probabilities": {
                "down_to_up": 1.0,
                "capture_vertical": 0.0,
                "fold_to_up": 0.0,
                "up_to_fold": 0.0,
            },
            "swing_up_sinus_episode_probability_start": 0.0,
            "swing_up_sinus_episode_probability_end": 0.0,
        }
    )

    down_sinus = deepcopy(down_no_sinus)
    down_sinus.update(
        {
            "swing_up_sinus_episode_probability_start": 1.0,
            "swing_up_sinus_episode_probability_end": 1.0,
        }
    )

    current_prefill_sinus = deepcopy(current)

    return {
        "current_mixed": current,
        "current_prefill_sinus": current_prefill_sinus,
        "capture_easy": capture_easy,
        "capture_only": capture_only,
        "capture_no_drop": capture_no_drop,
        "capture_late_learning": capture_late_learning,
        "capture_soft_drop": capture_soft_drop,
        "capture_actor_gate_safe": capture_actor_gate_safe,
        "capture_prefill_safe": capture_prefill_safe,
        "capture_prefill_safe_actor_gate": capture_prefill_safe_actor_gate,
        "capture_prefill_safe_late": capture_prefill_safe_late,
        "down_no_sinus": down_no_sinus,
        "down_sinus": down_sinus,
    }


def mean_component(summary, name):
    values = summary["reward_components"].get(name, [0.0])
    if not values:
        return 0.0
    return float(np.mean(values))


def run_variant(name, cfg, seed, prefill_steps):
    seeded(seed)
    trainer = PendulumTrainer(cfg)
    if name == "current_prefill_sinus" and prefill_steps > 0:
        trainer.prefill_replay_buffer(
            strategy="sinus",
            num_steps=prefill_steps,
            episode_mode="down_to_up",
            seed=seed,
        )
    elif name in (
        "capture_prefill_safe",
        "capture_prefill_safe_actor_gate",
        "capture_prefill_safe_late",
    ) and prefill_steps > 0:
        trainer.prefill_replay_buffer(
            strategy="random_safe",
            num_steps=prefill_steps,
            episode_mode="capture_vertical",
            seed=seed,
        )

    rows = []
    for episode in range(cfg["num_episodes"]):
        summary = trainer.collect_trajectory(episode)
        trainer._record_episode_metrics(summary)
        replay_diag = trainer.replay_diagnostics()
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
                "safety_penalty": mean_component(summary, "safety_penalty"),
                "action_abs_mean": summary["action_abs_mean"],
                "replay_size": replay_diag["size"],
                "replay_saturation_fraction": replay_diag["saturation_fraction"],
                "replay_capture_fraction": replay_diag["capture_fraction"],
                "replay_near_capture_fraction": replay_diag["near_capture_fraction"],
                "replay_non_crash_transitions": replay_diag["non_crash_transitions"],
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
        capture_rows = [
            row for row in group_tail
            if row["transition_direction"] == "capture_vertical"
        ]
        down_rows = [
            row for row in group_tail
            if row["transition_direction"] == "down_to_up"
        ]
        target_rows = capture_rows or group_tail
        summaries.append(
            {
                "variant": variant,
                "seed": seed,
                "episodes": len(group),
                "tail": len(group_tail),
                "tail_reward_mean": float(np.mean([row["episode_reward"] for row in group_tail])),
                "tail_final_hold_mean": float(np.mean([row["final_hold"] for row in target_rows])),
                "tail_balanced_hold_mean": float(np.mean([row["balanced_hold"] for row in target_rows])),
                "tail_success_rate": float(np.mean([row["episode_success"] for row in target_rows])),
                "tail_peak_mean": float(np.mean([row["peak_target_score"] for row in target_rows])),
                "tail_capture_drop_rate": float(
                    np.mean([row["capture_drop_penalty"] < 0.0 for row in group_tail])
                ),
                "tail_cart_hit_rate": float(
                    np.mean([row["cart_limit_step_penalty"] < 0.0 for row in group_tail])
                ),
                "tail_action_abs_mean": float(np.mean([row["action_abs_mean"] for row in group_tail])),
                "last_replay_saturation_fraction": float(group[-1]["replay_saturation_fraction"]),
                "last_replay_capture_fraction": float(group[-1]["replay_capture_fraction"]),
                "last_replay_near_capture_fraction": float(group[-1]["replay_near_capture_fraction"]),
                "last_replay_non_crash_transitions": int(group[-1]["replay_non_crash_transitions"]),
                "tail_capture_rows": len(capture_rows),
                "tail_down_rows": len(down_rows),
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
    parser.add_argument("--prefill-steps", type=int, default=1000)
    parser.add_argument("--out-dir", default="results/ablation")
    args = parser.parse_args()

    all_variants = variants(args.episodes, args.max_steps)
    selected_names = args.variants or list(all_variants)
    rows = []
    for variant_name in selected_names:
        if variant_name not in all_variants:
            raise ValueError(f"unknown variant {variant_name!r}; choices={sorted(all_variants)}")
        for seed in args.seeds:
            rows.extend(
                run_variant(
                    variant_name,
                    deepcopy(all_variants[variant_name]),
                    seed,
                    args.prefill_steps,
                )
            )

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
                "prefill_steps": args.prefill_steps,
            },
            indent=2,
        )
    )
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
