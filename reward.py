import numpy as np

from config import config


class RewardManager:
    """Pure reward and success metrics for the two-phase pendulum task."""

    def __init__(self):
        self.num_nodes = 2
        self.length = 1.0 / 3.0
        self.max_height = self.length * self.num_nodes
        self.threshold_ratio = 0.90
        self.phase_1_height_tolerance = 0.08
        self.phase_2_end_height_tolerance = 0.12
        self.cart_limit = 1.8
        self.hold_streak = 0
        self.max_hold_streak = 0
        self.best_end_y = -float("inf")
        self.best_target_score = 0.0

    def calculate_reward(
        self,
        state,
        terminated,
        current_step,
        action,
        phase=None,
        has_switched=False,
        swing_up_mode=False,
        update_hold_state=True,
    ):
        reward, components, force_terminated = self.evaluate(
            state,
            action,
            phase,
            terminated,
            has_switched=has_switched,
            swing_up_mode=swing_up_mode,
            hold_streak=self.hold_streak,
            best_end_y=self.best_end_y,
            best_target_score=self.best_target_score,
        )
        if update_hold_state:
            self.hold_streak = int(components["hold_streak"])
            self.max_hold_streak = max(self.max_hold_streak, self.hold_streak)
            self.best_end_y = max(self.best_end_y, float(components["end_y"]))
            self.best_target_score = max(self.best_target_score, float(components["target_score"]))
        return reward, components, force_terminated

    def evaluate(
        self,
        physical_state,
        action,
        phase,
        terminated=False,
        has_switched=False,
        swing_up_mode=False,
        hold_streak=0,
        best_end_y=-float("inf"),
        best_target_score=0.0,
    ):
        if phase not in (-1, 1):
            raise ValueError(f"phase must be -1 or 1, got {phase}")

        if len(physical_state) != 11:
            raise ValueError(f"expected 2-node physical state of length 11, got {len(physical_state)}")

        x, q1, q2, x_dot, u1, u2, _force = physical_state[:7]
        x1, y1, x2, y2 = physical_state[7:11]
        end_x = x2
        end_y = y2
        end_vy = self.length * np.cos(q1) * u1
        end_vy += self.length * np.cos(q2) * u2
        angular_speed = abs(u1) + abs(u2)

        velocity_penalty = 0.025 * (u1**2 + u2**2)
        cart_penalty = 0.35 * x**2
        action_penalty = 0.03 * float(action) ** 2
        terminal_penalty = 2.0 if terminated else 0.0

        if phase == 1:
            height_error = self.max_height - end_y
            end_x_error = end_x - x
            alignment_error = 1.0 - np.cos(q1 - q2)
            target_score = np.exp(-8.0 * height_error**2 - 2.0 * end_x_error**2)
            target_error = abs(height_error) + 0.5 * abs(end_x_error) + 0.25 * alignment_error
            shape_penalty = 0.20 * alignment_error
            in_target = (
                abs(height_error) < self.phase_1_height_tolerance
                and abs(end_x_error) < 0.20
                and angular_speed < 3.0
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
                and angular_speed < 3.0
            )

        low_score_penalty = 0.0
        if has_switched and target_score < config.get("post_switch_low_score_threshold", 0.2):
            low_score_penalty = config.get("post_switch_low_score_penalty", 0.5)

        swing_up_velocity_bonus = 0.0
        swing_up_momentum_bonus = 0.0
        swing_up_height_progress_bonus = 0.0
        swing_up_score_progress_bonus = 0.0
        capture_velocity_bonus = 0.0
        capture_hold_bonus = 0.0
        capture_overspeed_penalty = 0.0
        capture_height_penalty = 0.0
        capture_lost_penalty = 0.0
        capture_rest_penalty = 0.0
        effective_target_score = target_score
        capture_threshold = config.get("swing_up_capture_score_threshold", 0.75)
        if swing_up_mode and phase == 1 and not in_target:
            free_motion_threshold = config.get("swing_up_free_motion_score_threshold", 0.45)
            drive_threshold = config.get("swing_up_drive_score_threshold", 0.55)
            if target_score < free_motion_threshold:
                velocity_penalty *= config.get("swing_up_free_motion_velocity_penalty_scale", 0.0)
                action_penalty *= config.get("swing_up_free_motion_action_penalty_scale", 0.0)
            elif target_score < capture_threshold:
                velocity_penalty *= config.get("swing_up_velocity_penalty_scale", 0.20)
                action_penalty *= config.get("swing_up_action_penalty_scale", 0.25)
            if target_score < drive_threshold:
                swing_up_velocity_bonus = config.get("swing_up_vertical_velocity_weight", 0.8) * max(0.0, end_vy)
            if end_y < self.max_height * 0.55:
                angular_momentum = min(angular_speed, 8.0)
                swing_up_momentum_bonus = config.get("swing_up_momentum_weight", 0.08) * angular_momentum
            if target_score < capture_threshold:
                reference_end_y = best_end_y if np.isfinite(best_end_y) else end_y
                swing_up_height_progress_bonus = (
                    config.get("swing_up_height_progress_weight", 3.0)
                    * max(0.0, end_y - reference_end_y)
                )
                swing_up_score_progress_bonus = (
                    config.get("swing_up_score_progress_weight", 2.0)
                    * max(0.0, target_score - best_target_score)
                )
        if phase == 1:
            capture_floor = self.max_height * config.get("capture_height_floor_ratio", 0.75)
            capture_height_penalty = (
                config.get("capture_height_penalty_weight", 0.0)
                * max(0.0, capture_floor - end_y)
            )
            if best_target_score >= capture_threshold and target_score < capture_threshold:
                capture_lost_penalty = (
                    config.get("capture_lost_penalty_weight", 0.0)
                    * (capture_threshold - target_score)
                    / max(capture_threshold, 1e-6)
                )
            if target_score < 0.10 and angular_speed < 1.0:
                capture_rest_penalty = config.get("capture_rest_penalty_weight", 0.0)

        if phase == 1 and target_score > capture_threshold:
            allowed_speed = config.get("capture_allowed_angular_speed", 1.5)
            overspeed = max(0.0, angular_speed - allowed_speed)
            effective_target_score = target_score / (
                1.0 + config.get("capture_target_speed_discount", 0.6) * overspeed
            )
            capture_overspeed_penalty = (
                config.get("capture_overspeed_penalty_weight", 0.20)
                * target_score
                * min(overspeed, 8.0)
            )
            if in_target:
                capture_velocity_bonus = config.get("capture_velocity_bonus_weight", 1.0) / (1.0 + angular_speed)

        next_hold_streak = hold_streak + 1 if in_target else 0
        hold_required_steps = max(1, int(config.get("hold_required_steps", 120)))
        hold_progress = min(1.0, next_hold_streak / hold_required_steps)
        target_shaping_reward = config.get("target_shaping_weight", 0.15) * effective_target_score
        target_entry_bonus = config.get("target_entry_bonus", 0.75) if in_target else 0.0
        hold_progress_bonus = config.get("hold_progress_bonus", 2.0) * hold_progress if in_target else 0.0
        capture_hold_bonus = (
            config.get("capture_hold_bonus", 0.0)
            + target_entry_bonus
            + hold_progress_bonus
        ) if in_target else 0.0

        reward = (
            target_shaping_reward
            + swing_up_velocity_bonus
            + swing_up_momentum_bonus
            + swing_up_height_progress_bonus
            + swing_up_score_progress_bonus
            + capture_velocity_bonus
            + capture_hold_bonus
            - shape_penalty
            - velocity_penalty
            - cart_penalty
            - action_penalty
            - capture_overspeed_penalty
            - capture_height_penalty
            - capture_lost_penalty
            - capture_rest_penalty
            - terminal_penalty
            - low_score_penalty
        )
        post_switch_weight = config.get("post_switch_reward_weight", 2.0) if has_switched else 1.0
        reward *= post_switch_weight

        components = {
            "reward": float(reward),
            "target_score": float(target_score),
            "effective_target_score": float(effective_target_score),
            "target_error": float(target_error),
            "end_vy": float(end_vy),
            "shape_penalty": float(shape_penalty),
            "velocity_penalty": float(velocity_penalty),
            "cart_penalty": float(cart_penalty),
            "action_penalty": float(action_penalty),
            "swing_up_velocity_bonus": float(swing_up_velocity_bonus),
            "swing_up_momentum_bonus": float(swing_up_momentum_bonus),
            "swing_up_height_progress_bonus": float(swing_up_height_progress_bonus),
            "swing_up_score_progress_bonus": float(swing_up_score_progress_bonus),
            "capture_velocity_bonus": float(capture_velocity_bonus),
            "capture_overspeed_penalty": float(capture_overspeed_penalty),
            "capture_height_penalty": float(capture_height_penalty),
            "capture_lost_penalty": float(capture_lost_penalty),
            "capture_rest_penalty": float(capture_rest_penalty),
            "capture_hold_bonus": float(capture_hold_bonus),
            "target_shaping_reward": float(target_shaping_reward),
            "target_entry_bonus": float(target_entry_bonus),
            "hold_progress_bonus": float(hold_progress_bonus),
            "hold_streak": float(next_hold_streak),
            "hold_progress": float(hold_progress),
            "terminal_penalty": float(terminal_penalty),
            "low_score_penalty": float(low_score_penalty),
            "post_switch_weight": float(post_switch_weight),
            "has_switched": float(has_switched),
            "in_target": float(in_target),
            "end_y": float(end_y),
            "end_x": float(end_x),
            "cart_velocity": float(x_dot),
            "phase": float(phase),
        }
        force_terminated = bool(terminated)
        return float(reward), components, force_terminated

    def reset(self):
        self.hold_streak = 0
        self.max_hold_streak = 0
        self.best_end_y = -float("inf")
        self.best_target_score = 0.0
        return None
