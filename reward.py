import numpy as np

from config import config


class RewardManager:
    """Pure reward and success metrics for the two-phase pendulum task."""

    def __init__(self):
        self.num_nodes = config["num_nodes"]
        self.length = 1.0 / 3.0
        self.max_height = self.length * self.num_nodes
        self.threshold_ratio = 0.90
        self.phase_1_height_tolerance = 0.08
        self.phase_2_end_height_tolerance = 0.12
        self.cart_limit = 1.8

    def calculate_reward(self, state, terminated, current_step, action, phase=None, has_switched=False):
        reward, components, force_terminated = self.evaluate(
            state,
            action,
            phase,
            terminated,
            has_switched=has_switched,
        )
        return reward, components, force_terminated

    def evaluate(self, physical_state, action, phase, terminated=False, has_switched=False):
        if phase not in (-1, 1):
            raise ValueError(f"phase must be -1 or 1, got {phase}")

        x, q1, q2, q3, u1, u2, u3, _force = physical_state[:8]
        x1, y1, x2, y2, x3, y3 = physical_state[8:14]
        end_x = x3 if self.num_nodes == 3 else x2 if self.num_nodes == 2 else x1
        end_y = y3 if self.num_nodes == 3 else y2 if self.num_nodes == 2 else y1

        velocity_penalty = 0.025 * (u1**2 + u2**2 + u3**2)
        cart_penalty = 0.35 * x**2
        action_penalty = 0.03 * float(action) ** 2
        terminal_penalty = 2.0 if terminated else 0.0

        if phase == 1:
            height_error = self.max_height - end_y
            end_x_error = end_x - x
            alignment_error = (1.0 - np.cos(q1 - q2)) + (1.0 - np.cos(q2 - q3))
            target_score = np.exp(-8.0 * height_error**2 - 2.0 * end_x_error**2)
            target_error = abs(height_error) + 0.5 * abs(end_x_error) + 0.25 * alignment_error
            shape_penalty = 0.20 * alignment_error
            in_target = (
                abs(height_error) < self.phase_1_height_tolerance
                and abs(end_x_error) < 0.20
                and abs(u1) + abs(u2) + abs(u3) < 3.0
            )
        else:
            y1_error = self.length - y1
            end_y_error = end_y
            end_x_error = end_x - x
            folded_error = 1.0 + np.cos(q1 - q2)
            target_score = np.exp(
                -8.0 * y1_error**2
                -10.0 * end_y_error**2
                -1.5 * end_x_error**2
            )
            target_error = abs(y1_error) + abs(end_y_error) + 0.5 * abs(end_x_error) + 0.25 * folded_error
            shape_penalty = 0.20 * folded_error
            in_target = (
                abs(y1_error) < 0.10
                and abs(end_y_error) < self.phase_2_end_height_tolerance
                and abs(end_x_error) < 0.35
                and abs(u1) + abs(u2) + abs(u3) < 3.0
            )

        low_score_penalty = 0.0
        if has_switched and target_score < config.get("post_switch_low_score_threshold", 0.2):
            low_score_penalty = config.get("post_switch_low_score_penalty", 0.5)

        reward = (
            2.0 * target_score
            + (0.5 if in_target else 0.0)
            - shape_penalty
            - velocity_penalty
            - cart_penalty
            - action_penalty
            - terminal_penalty
            - low_score_penalty
        )
        post_switch_weight = config.get("post_switch_reward_weight", 2.0) if has_switched else 1.0
        reward *= post_switch_weight

        components = {
            "reward": float(reward),
            "target_score": float(target_score),
            "target_error": float(target_error),
            "shape_penalty": float(shape_penalty),
            "velocity_penalty": float(velocity_penalty),
            "cart_penalty": float(cart_penalty),
            "action_penalty": float(action_penalty),
            "terminal_penalty": float(terminal_penalty),
            "low_score_penalty": float(low_score_penalty),
            "post_switch_weight": float(post_switch_weight),
            "has_switched": float(has_switched),
            "in_target": float(in_target),
            "end_y": float(end_y),
            "end_x": float(end_x),
            "phase": float(phase),
        }
        force_terminated = bool(terminated)
        return float(reward), components, force_terminated

    def reset(self):
        """Kept for compatibility; reward evaluation is stateless now."""
        return None
