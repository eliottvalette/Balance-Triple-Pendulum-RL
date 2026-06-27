from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class RewardResult:
    reward: float
    components: dict[str, float]
    capture_started: bool
    hold_streak: int
    success: bool


class RewardManager:
    """Deterministic reward for one physical-state transition."""

    def __init__(
        self,
        cfg: Mapping[str, object],
        *,
        arm_length: float = 1.0 / 3.0,
        mass: float = 0.01 / 3.0,
    ) -> None:
        self.config = cfg
        self.num_nodes = 2
        self.length = float(arm_length)
        self.mass = float(mass)
        self.max_height = self.length * self.num_nodes
        self.phase_1_height_tolerance = 0.08
        self.phase_2_end_height_tolerance = 0.12
        self.cart_limit = 1.8

    @property
    def maximum_swing_potential(self) -> float:
        return float(
            self.config["swing_up_energy_progress_weight"]
            + self.config["swing_up_height_progress_weight"]
            + self.config["swing_up_cart_safety_weight"]
        )

    def evaluate_transition(
        self,
        previous_state: np.ndarray,
        next_state: np.ndarray,
        *,
        action: float,
        phase: int,
        capture_started: bool,
        hold_streak: int,
    ) -> RewardResult:
        self._validate_physical_state(previous_state, "previous_state")
        self._validate_physical_state(next_state, "next_state")
        if phase not in (-1, 1):
            raise ValueError(f"phase must be -1 or 1, got {phase}")
        if not np.isfinite(action):
            raise ValueError(f"action must be finite, got {action!r}")
        if hold_streak < 0:
            raise ValueError(f"hold_streak must be nonnegative, got {hold_streak}")

        metrics = self._target_metrics(next_state, phase)
        entered_capture = bool(
            not capture_started
            and metrics["target_score"] >= self.config["swing_up_capture_score_threshold"]
        )
        next_capture_started = bool(capture_started or entered_capture)
        next_hold_streak = hold_streak + 1 if metrics["in_target"] else 0
        hold_progress = float(metrics["in_target"])

        previous_potential = self.swing_potential(previous_state)
        next_potential = self.swing_potential(next_state)
        potential_progress = next_potential - previous_potential
        capture_quality = 0.0
        capture_entry_bonus = 0.0
        hold_bonus = 0.0

        if next_capture_started or phase == -1:
            speed_scale = float(self.config["capture_allowed_angular_speed"])
            speed_score = 1.0 / (1.0 + (metrics["angular_speed"] / speed_scale) ** 2)
            cart_score = self._cart_safety_score(float(next_state[0]))
            action_ratio = float(action) / float(self.config["max_action"])
            action_score = 1.0 / (1.0 + action_ratio**2)
            capture_quality = metrics["target_score"] * speed_score * cart_score * action_score
            capture_entry_bonus = (
                float(self.config["capture_entry_bonus"]) if entered_capture else 0.0
            )
            hold_bonus = (
                float(self.config["hold_progress_bonus"]) * capture_quality
                if metrics["in_target"]
                else 0.0
            )
            reward = capture_quality + capture_entry_bonus + hold_bonus
        else:
            speed_score = 0.0
            cart_score = self._cart_safety_score(float(next_state[0]))
            action_score = 0.0
            reward = potential_progress

        components = {
            "reward": float(reward),
            "reward_mode_capture": float(next_capture_started or phase == -1),
            "target_score": metrics["target_score"],
            "effective_target_score": metrics["target_score"] * speed_score,
            "target_error": metrics["target_error"],
            "in_target": float(metrics["in_target"]),
            "end_y": metrics["end_y"],
            "end_x": metrics["end_x"],
            "end_vy": metrics["end_vy"],
            "angular_speed": metrics["angular_speed"],
            "mechanical_energy": self._mechanical_energy(next_state),
            "energy_score": self._energy_score(next_state),
            "height_score": self._height_score(next_state),
            "cart_safety_score": cart_score,
            "previous_potential": previous_potential,
            "potential_score": next_potential,
            "potential_progress": potential_progress,
            "capture_quality": capture_quality,
            "capture_entry_bonus": capture_entry_bonus,
            "capture_speed_score": speed_score,
            "capture_action_score": action_score,
            "hold_bonus": hold_bonus,
            "hold_streak": float(next_hold_streak),
            "hold_progress": float(hold_progress),
            "phase": float(phase),
        }
        return RewardResult(
            reward=float(reward),
            components=components,
            capture_started=next_capture_started,
            hold_streak=next_hold_streak,
            success=False,
        )

    def swing_potential(self, physical_state: np.ndarray) -> float:
        self._validate_physical_state(physical_state, "physical_state")
        return float(
            self.config["swing_up_energy_progress_weight"] * self._energy_score(physical_state)
            + self.config["swing_up_height_progress_weight"] * self._height_score(physical_state)
            + self.config["swing_up_cart_safety_weight"]
            * self._cart_safety_score(float(physical_state[0]))
        )

    def verify_cart_termination_is_suboptimal(
        self,
        *,
        max_steps: int,
        gamma: float,
    ) -> list[float]:
        """Prove the rail penalty is below the reward-to-go lower bound at every step."""
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")

        penalty = float(self.config["cart_failure_penalty"])
        margins = []
        for step in range(max_steps):
            remaining_steps = max_steps - step
            # For r_t = Phi(s_{t+1}) - Phi(s_t), Phi in [0, Phi_max],
            # every finite discounted continuation is bounded below by -Phi_max.
            continuation_lower_bound = -self.maximum_swing_potential
            if remaining_steps < 1:
                raise AssertionError("verification must include a continuation step")
            margin = continuation_lower_bound - penalty
            if margin <= 0.0:
                raise ValueError(
                    "cart_failure_penalty is not strictly worse than the continuation "
                    f"lower bound at step {step}: penalty={penalty}, "
                    f"lower_bound={continuation_lower_bound}"
                )
            margins.append(float(margin))
        return margins

    def _target_metrics(self, physical_state: np.ndarray, phase: int) -> dict[str, float | bool]:
        x, q1, q2, _x_dot, u1, u2, _force = physical_state[:7]
        _x1, y1, end_x, end_y = physical_state[7:11]
        end_vy = self.length * np.cos(q1) * u1 + self.length * np.cos(q2) * u2
        angular_speed = abs(u1) + abs(u2)
        end_x_error = end_x - x

        if phase == 1:
            height_error = self.max_height - end_y
            alignment_error = 1.0 - np.cos(q1 - q2)
            target_score = np.exp(-8.0 * height_error**2 - 2.0 * end_x_error**2)
            target_error = (
                abs(height_error) + 0.5 * abs(end_x_error) + 0.25 * alignment_error
            )
            in_target = (
                abs(height_error) < self.phase_1_height_tolerance
                and abs(end_x_error) < 0.20
                and angular_speed < 3.0
            )
        else:
            y1_error = self.length - y1
            folded_error = 1.0 + np.cos(q1 - q2)
            target_score = np.exp(
                -8.0 * y1_error**2 - 10.0 * end_y**2 - 1.5 * end_x_error**2
            )
            target_error = (
                abs(y1_error) + abs(end_y) + 0.5 * abs(end_x_error) + 0.25 * folded_error
            )
            in_target = (
                abs(y1_error) < 0.10
                and abs(end_y) < self.phase_2_end_height_tolerance
                and abs(end_x_error) < 0.35
                and angular_speed < 3.0
            )

        return {
            "target_score": float(target_score),
            "target_error": float(target_error),
            "in_target": bool(in_target),
            "end_y": float(end_y),
            "end_x": float(end_x),
            "end_vy": float(end_vy),
            "angular_speed": float(angular_speed),
        }

    def _mechanical_energy(self, physical_state: np.ndarray) -> float:
        x, q1, q2, x_dot, u1, u2 = physical_state[:6]
        _x1, y1, _x2, y2 = physical_state[7:11]
        del x
        vx1 = x_dot - self.length * np.sin(q1) * u1
        vy1 = self.length * np.cos(q1) * u1
        vx2 = vx1 - self.length * np.sin(q2) * u2
        vy2 = vy1 + self.length * np.cos(q2) * u2
        kinetic = 0.5 * self.mass * (
            x_dot**2 + vx1**2 + vy1**2 + vx2**2 + vy2**2
        )
        potential = self.mass * float(self.config["gravity"]) * (y1 + y2)
        return float(kinetic + potential)

    def _energy_score(self, physical_state: np.ndarray) -> float:
        gravity = float(self.config["gravity"])
        upright_energy = 3.0 * self.mass * gravity * self.length
        energy_span = 6.0 * self.mass * gravity * self.length
        error_ratio = abs(self._mechanical_energy(physical_state) - upright_energy) / energy_span
        return float(1.0 - np.clip(error_ratio, 0.0, 1.0))

    def _height_score(self, physical_state: np.ndarray) -> float:
        end_y = float(physical_state[10])
        normalized = (end_y + self.max_height) / (2.0 * self.max_height)
        return float(np.clip(normalized, 0.0, 1.0))

    def _cart_safety_score(self, cart_x: float) -> float:
        normalized = abs(cart_x) / self.cart_limit
        return float(np.clip(1.0 - normalized**2, 0.0, 1.0))

    @staticmethod
    def _validate_physical_state(physical_state: np.ndarray, name: str) -> None:
        if np.shape(physical_state) != (11,):
            raise ValueError(f"{name} must have shape (11,), got {np.shape(physical_state)}")
        if not np.all(np.isfinite(physical_state)):
            raise FloatingPointError(f"{name} contains non-finite values: {physical_state}")
