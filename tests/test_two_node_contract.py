import math
import unittest
from unittest import mock

import numpy as np

from config import config, validate_config
from metrics import MetricsTracker
from reward import RewardManager, RewardResult
from train import PendulumTrainer
from tp_env import PHYSICAL_STATE_LAYOUT, PendulumEnv


def physical_state(
    *,
    x=0.0,
    q1=-math.pi / 2,
    q2=-math.pi / 2,
    x_dot=0.0,
    u1=0.0,
    u2=0.0,
):
    length = 1.0 / 3.0
    x1 = x + length * math.cos(q1)
    y1 = length * math.sin(q1)
    x2 = x1 + length * math.cos(q2)
    y2 = y1 + length * math.sin(q2)
    return np.array([x, q1, q2, x_dot, u1, u2, 0.0, x1, y1, x2, y2])


def env_config(**overrides):
    cfg = dict(config)
    cfg.update(overrides)
    return cfg


def mechanical_energy(env):
    state = env.get_physical_state()
    x, q1, q2, x_dot, u1, u2 = state[:6]
    _force, _x1, y1, _x2, y2 = state[6:11]
    arm = env.arm_length
    vx1 = x_dot - arm * math.sin(q1) * u1
    vy1 = arm * math.cos(q1) * u1
    vx2 = vx1 - arm * math.sin(q2) * u2
    vy2 = vy1 + arm * math.cos(q2) * u2
    kinetic = 0.5 * env.cart_mass * x_dot**2
    kinetic += 0.5 * env.bob_mass * (vx1**2 + vy1**2)
    kinetic += 0.5 * env.bob_mass * (vx2**2 + vy2**2)
    potential = env.bob_mass * float(env.config["gravity"]) * (y1 + y2)
    return float(kinetic + potential)


class StrictContractTests(unittest.TestCase):
    def test_config_rejects_invalid_episode_probabilities(self):
        invalid = dict(config)
        invalid["episode_mode_probabilities"] = {"down_to_up": 1.0}

        with self.assertRaisesRegex(ValueError, "episode_mode_probabilities"):
            validate_config(invalid)

    def test_config_rejects_invalid_transition_range(self):
        invalid = dict(config)
        invalid["transition_switch_step_min"] = 200
        invalid["transition_switch_step_max"] = 100

        with self.assertRaisesRegex(ValueError, "transition_switch_step"):
            validate_config(invalid)

    def test_step_before_reset_raises(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(RuntimeError, "before reset"):
            env.step(0.0)

    def test_out_of_range_action_raises_instead_of_being_clipped(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")

        with self.assertRaisesRegex(ValueError, "action"):
            env.step(config["max_action"] + 0.01)

    def test_action_near_rail_is_not_silently_replaced(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")
        env.current_state[0] = 1.66
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.2)

        self.assertEqual(0.2, env.previous_action)
        env.rhs.assert_called_once()

    def test_nonfinite_dynamics_raise_immediately(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")
        env.rhs = mock.Mock(return_value=np.full_like(env.current_state, np.nan))

        with self.assertRaisesRegex(FloatingPointError, "non-finite derivative"):
            env.step(0.0)

    def test_empty_environment_config_is_not_replaced_by_global_config(self):
        with self.assertRaisesRegex(ValueError, "missing required config"):
            PendulumEnv(env_config={})

    def test_environment_uses_injected_gravity(self):
        custom = env_config(gravity=1.234)
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )

        self.assertEqual(1.234, env.parameter_vals[0])

    def test_invalid_phase_raises(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(ValueError, "phase"):
            env.reset(phase=0)

    def test_reset_requires_an_explicit_mode_or_phase(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(ValueError, "episode_mode or phase"):
            env.reset()

    def test_missing_saved_model_is_not_ignored(self):
        trainer = object.__new__(PendulumTrainer)
        with mock.patch("train.os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                trainer.load_models()

    def test_metrics_reject_non_scalar_values(self):
        tracker = MetricsTracker(config["plot_config"])

        with self.assertRaises(TypeError):
            tracker.add_metric("invalid", np.array([1.0, 2.0]))

    def test_missing_moving_average_metric_raises(self):
        tracker = MetricsTracker(config["plot_config"])

        with self.assertRaisesRegex(KeyError, "missing"):
            tracker.get_moving_average("missing")


class RewardContractTests(unittest.TestCase):
    def setUp(self):
        self.manager = RewardManager(config)

    def test_stationary_bottom_has_nonpositive_swing_up_reward(self):
        state = physical_state()

        result = self.manager.evaluate_transition(
            state,
            state,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertLessEqual(result.reward, 0.0)
        self.assertAlmostEqual(result.reward, result.components["swing_up_progress"])

    def test_reward_components_expose_contract_layers(self):
        state = physical_state()

        result = self.manager.evaluate_transition(
            state,
            state,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        for component in (
            "swing_up_progress",
            "capture_entry",
            "capture_quality_reward",
            "hold_stabilization",
            "safety_penalty",
            "terminal_failure_penalty",
        ):
            self.assertIn(component, result.components)

    def test_potential_based_progress_rewards_moving_toward_swing_up(self):
        previous = physical_state(q1=-math.pi / 2, q2=-math.pi / 2)
        next_state = physical_state(q1=0.2, q2=0.1, u1=0.2, u2=0.2)

        result = self.manager.evaluate_transition(
            previous,
            next_state,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertGreater(result.components["potential_progress"], 0.0)
        self.assertGreater(result.reward, 0.0)

    def test_potential_based_progress_penalizes_moving_away(self):
        previous = physical_state(q1=0.2, q2=0.1, u1=0.2, u2=0.2)
        next_state = physical_state(q1=-math.pi / 2, q2=-math.pi / 2)

        result = self.manager.evaluate_transition(
            previous,
            next_state,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertLess(result.components["potential_progress"], 0.0)
        self.assertLess(result.reward, 0.0)

    def test_high_energy_near_rail_is_not_over_rewarded(self):
        safe = physical_state(q1=0.1, q2=0.1, u1=0.5, u2=0.5, x=0.0)
        unsafe = physical_state(q1=0.1, q2=0.1, u1=4.0, u2=-4.0, x=1.79)

        self.assertLess(self.manager.swing_potential(unsafe), self.manager.swing_potential(safe))

    def test_high_height_with_excessive_speed_is_not_successful_capture(self):
        fast_upright = physical_state(q1=math.pi / 2, q2=math.pi / 2, u1=4.0, u2=4.0)

        result = self.manager.evaluate_transition(
            fast_upright,
            fast_upright,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertFalse(result.capture_started)
        self.assertFalse(result.components["in_target"])

    def test_reward_is_deterministic_for_the_same_transition(self):
        previous = physical_state()
        current = physical_state(q1=-1.2, q2=-1.3, u1=0.4, u2=0.2)
        kwargs = dict(
            action=0.1,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        first = self.manager.evaluate_transition(previous, current, **kwargs)
        second = self.manager.evaluate_transition(previous, current, **kwargs)

        self.assertEqual(first, second)

    def test_capture_reward_is_nonnegative_and_enters_capture(self):
        upright = physical_state(q1=math.pi / 2, q2=math.pi / 2)

        result = self.manager.evaluate_transition(
            upright,
            upright,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertTrue(result.capture_started)
        self.assertGreaterEqual(result.reward, 0.0)
        self.assertEqual(1, result.hold_streak)

    def test_cart_penalty_is_worse_than_continuation_at_every_episode_step(self):
        margins = self.manager.verify_cart_termination_is_suboptimal(
            max_steps=config["max_steps"],
            gamma=config["gamma"],
        )

        self.assertEqual(config["max_steps"], len(margins))
        self.assertGreater(min(margins), 0.0)

    def test_cart_penalty_validation_rejects_insufficient_penalty(self):
        invalid = dict(config)
        invalid["cart_failure_penalty"] = -5.0
        manager = RewardManager(invalid)

        with self.assertRaisesRegex(ValueError, "cart_failure_penalty"):
            manager.verify_cart_termination_is_suboptimal(
                max_steps=invalid["max_steps"],
                gamma=invalid["gamma"],
            )

    def test_hold_reward_tracks_progress_without_discrete_success_bonus(self):
        upright = physical_state(q1=math.pi / 2, q2=math.pi / 2)

        first = self.manager.evaluate_transition(
            upright,
            upright,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=0,
        )
        second = self.manager.evaluate_transition(
            upright,
            upright,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=1,
        )

        self.assertFalse(first.success)
        self.assertFalse(second.success)
        self.assertGreater(first.components["hold_bonus"], 0.0)
        self.assertGreater(second.components["hold_progress"], first.components["hold_progress"])
        self.assertGreater(second.components["hold_progress_delta"], 0.0)
        self.assertNotIn("success_bonus", first.components)

    def test_sustained_hold_is_better_than_one_step_contact(self):
        upright = physical_state(q1=math.pi / 2, q2=math.pi / 2)

        brief = self.manager.evaluate_transition(
            upright,
            upright,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=0,
        )
        sustained = self.manager.evaluate_transition(
            upright,
            upright,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=50,
        )

        self.assertGreater(sustained.components["hold_progress"], brief.components["hold_progress"])
        self.assertGreater(sustained.reward, 0.0)

    def test_losing_target_removes_accumulated_hold_progress_reward(self):
        upright = physical_state(q1=math.pi / 2, q2=math.pi / 2)
        bottom = physical_state()

        result = self.manager.evaluate_transition(
            upright,
            bottom,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=100,
        )

        self.assertEqual(0, result.hold_streak)
        self.assertLess(result.components["hold_progress_delta"], 0.0)
        self.assertEqual(0.0, result.components["hold_bonus"])
        self.assertLess(result.components["capture_score_decay_penalty"], 0.0)
        self.assertLess(result.components["capture_quality_bonus"], 1.0)

    def test_losing_target_is_better_than_recoverable_rail_crash(self):
        upright = physical_state(q1=math.pi / 2, q2=math.pi / 2)
        bottom = physical_state()
        loss = self.manager.evaluate_transition(
            upright,
            bottom,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=10,
        )

        self.assertGreater(loss.reward, config["cart_failure_penalty"])

    def test_capture_quality_gives_small_dense_reward_without_hold(self):
        near_target = physical_state(q1=math.pi / 2, q2=math.pi / 2, u1=4.0, u2=4.0)

        result = self.manager.evaluate_transition(
            near_target,
            near_target,
            action=0.0,
            phase=1,
            capture_started=True,
            hold_streak=0,
        )

        self.assertFalse(result.components["in_target"])
        self.assertGreater(result.components["capture_quality_bonus"], 0.0)
        self.assertLess(result.components["capture_quality_bonus"], 1.0)
        self.assertEqual(0.0, result.components["hold_bonus"])

    def test_capture_vertical_cart_center_bonus_in_phase_up_above_threshold(self):
        custom = dict(config)
        custom["capture_cart_center_weight"] = 2.0
        manager = RewardManager(custom)
        upright_center = physical_state(q1=math.pi / 2, q2=math.pi / 2, x=0.0)
        upright_offset = physical_state(q1=math.pi / 2, q2=math.pi / 2, x=1.0)
        kwargs = dict(action=0.0, phase=1, capture_started=True, hold_streak=10, initial_pose_mode="capture")
        center = manager.evaluate_transition(upright_center, upright_center, **kwargs)
        offset = manager.evaluate_transition(upright_offset, upright_offset, **kwargs)
        self.assertGreater(center.components["capture_cart_center_bonus"], 0.0)
        self.assertGreater(center.components["capture_cart_center_bonus"], offset.components["capture_cart_center_bonus"])
        below_threshold = manager.evaluate_transition(physical_state(), physical_state(), action=0.0, phase=1, capture_started=True, hold_streak=0, initial_pose_mode="capture")
        self.assertEqual(0.0, below_threshold.components["capture_cart_center_bonus"])
        phase_down = manager.evaluate_transition(upright_center, upright_center, action=0.0, phase=-1, capture_started=True, hold_streak=10, initial_pose_mode="capture")
        self.assertEqual(0.0, phase_down.components["capture_cart_center_bonus"])
        other_mode = manager.evaluate_transition(upright_center, upright_center, action=0.0, phase=1, capture_started=True, hold_streak=10, initial_pose_mode="down")
        self.assertEqual(0.0, other_mode.components["capture_cart_center_bonus"])

    def test_excessive_angular_speed_cannot_enter_capture(self):
        fast_upright = physical_state(q1=math.pi / 2, q2=math.pi / 2, u1=2.0, u2=2.0)

        result = self.manager.evaluate_transition(
            fast_upright,
            fast_upright,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertFalse(result.capture_started)
        self.assertEqual(0, result.hold_streak)

    def test_hold_threshold_config_is_not_required(self):
        cfg = dict(config)
        cfg.pop("hold_required_steps", None)
        cfg.pop("success_bonus", None)

        self.assertIs(validate_config(cfg), cfg)


class TwoNodeEnvironmentTests(unittest.TestCase):
    def _rollout_return(self, cfg, policy, *, mode="down_to_up", seed=0, initial_x=None):
        env = PendulumEnv(reward_manager=RewardManager(cfg), max_steps=cfg["max_steps"], env_config=cfg)
        state = env.reset(episode_mode=mode, seed=seed)
        if initial_x is not None:
            env.current_state[0] = float(initial_x)
            state = env.get_state(action=0.0)
        total = 0.0
        for step in range(cfg["max_steps"]):
            action = policy(step, state)
            state, reward, terminated, truncated, _info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        return total

    def test_episode_modes_reset_to_expected_phase_and_pose(self):
        custom = env_config(
            initial_angle_noise=0.0,
            initial_velocity_noise=0.0,
            capture_angle_noise=0.0,
            capture_cart_position_noise=0.0,
            capture_cart_velocity_noise=0.0,
            capture_angular_velocity_noise=0.0,
            down_angle_noise=0.0,
            transition_switch_step_min=5,
            transition_switch_step_max=5,
        )
        expected = {
            "down_to_up": (1, "down", [-math.pi / 2, -math.pi / 2], False),
            "capture_vertical": (1, "capture", [math.pi / 2, math.pi / 2], True),
            "up_to_fold": (1, "target", [math.pi / 2, math.pi / 2], True),
            "fold_to_up": (-1, "target", [math.pi / 2, -math.pi / 2], True),
        }

        for mode, (phase, pose_mode, angles, capture_started) in expected.items():
            with self.subTest(mode=mode):
                env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
                env.reset(episode_mode=mode, seed=123)

                self.assertEqual(phase, env.current_phase)
                self.assertEqual(pose_mode, env.initial_pose_mode)
                self.assertTrue(np.allclose(angles, env.current_state[1:3]))
                self.assertEqual(capture_started, env.capture_started)

    def test_reset_is_deterministic_under_seed(self):
        custom = env_config(transition_switch_step_min=3, transition_switch_step_max=7)
        env_a = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env_b = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)

        state_a = env_a.reset(episode_mode="up_to_fold", seed=42)
        state_b = env_b.reset(episode_mode="up_to_fold", seed=42)

        self.assertTrue(np.allclose(state_a, state_b))
        self.assertEqual(env_a.switch_step, env_b.switch_step)

    def test_phase_switch_occurs_only_at_configured_step(self):
        custom = env_config(
            transition_switch_step_min=2,
            transition_switch_step_max=2,
            initial_angle_noise=0.0,
            initial_velocity_noise=0.0,
        )
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="up_to_fold", seed=7)
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        _state, _reward, _terminated, _truncated, first_info = env.step(0.0)
        _state, _reward, _terminated, _truncated, second_info = env.step(0.0)
        _state, _reward, _terminated, _truncated, third_info = env.step(0.0)

        self.assertFalse(first_info["switched"])
        self.assertFalse(second_info["switched"])
        self.assertTrue(third_info["switched"])
        self.assertEqual(-1, third_info["phase"])

    def test_down_to_up_never_switches_before_horizon(self):
        custom = env_config(max_steps=3)
        env = PendulumEnv(reward_manager=RewardManager(custom), max_steps=3, env_config=custom)
        env.reset(episode_mode="down_to_up", seed=5)
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        infos = [env.step(0.0)[4] for _ in range(3)]

        self.assertTrue(all(not info["switched"] for info in infos))
        self.assertEqual("max_steps", infos[-1]["termination_reason"])

    def test_physical_state_layout_is_documented_and_stable(self):
        self.assertEqual(
            ("x", "q1", "q2", "x_dot", "q1_dot", "q2_dot", "force", "x1", "y1", "x2", "y2"),
            PHYSICAL_STATE_LAYOUT,
        )

        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="capture_vertical")

        self.assertEqual((len(PHYSICAL_STATE_LAYOUT),), env.get_physical_state().shape)

    def test_symbolic_angular_friction_enters_generalized_equations(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)

        active_forces = str(env.fr)

        self.assertIn("-b_theta*u1", active_forces)
        self.assertIn("-b_theta*u2", active_forces)
        self.assertNotIn("b_theta*u0", active_forces)

    def test_angular_damping_decreases_mechanical_energy(self):
        custom = env_config(
            angular_friction=0.0005,
            cart_friction=0.0,
            angular_velocity_damping=0.0,
            max_steps=100,
        )
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.current_state[:] = np.array([0.0, 0.2, -0.4, 0.0, 1.2, -0.7])
        start_energy = mechanical_energy(env)

        for _ in range(20):
            env.step(0.0)

        self.assertLess(mechanical_energy(env), start_energy)

    def test_zero_damping_energy_drift_is_bounded_over_short_rollout(self):
        custom = env_config(
            angular_friction=0.0,
            cart_friction=0.0,
            angular_velocity_damping=0.0,
            max_steps=100,
        )
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.current_state[:] = np.array([0.0, 0.25, -0.35, 0.02, 0.35, -0.2])
        start_energy = mechanical_energy(env)

        for _ in range(10):
            env.step(0.0)

        self.assertLess(abs(mechanical_energy(env) - start_energy), 1e-4)

    def test_angular_friction_measurably_changes_angular_acceleration(self):
        no_damping = env_config(angular_friction=0.0, cart_friction=0.0, angular_velocity_damping=0.0)
        with_damping = env_config(angular_friction=0.0005, cart_friction=0.0, angular_velocity_damping=0.0)
        state = np.array([0.0, 0.3, -0.6, 0.0, 1.0, -1.0])

        env_a = PendulumEnv(reward_manager=RewardManager(no_damping), env_config=no_damping)
        env_b = PendulumEnv(reward_manager=RewardManager(with_damping), env_config=with_damping)
        derivative_a = env_a.rhs(state, 0.0, env_a.parameter_vals, lambda _state: 0.0)
        derivative_b = env_b.rhs(state, 0.0, env_b.parameter_vals, lambda _state: 0.0)

        self.assertGreater(abs(derivative_a[4] - derivative_b[4]), 1e-6)
        self.assertGreater(abs(derivative_a[5] - derivative_b[5]), 1e-6)

    def test_missing_or_nonpositive_cart_mass_is_rejected(self):
        missing = env_config()
        missing.pop("cart_mass")
        with self.assertRaisesRegex(ValueError, "cart_mass"):
            PendulumEnv(env_config=missing)

        invalid = env_config(cart_mass=0.0)
        with self.assertRaisesRegex(ValueError, "cart_mass"):
            PendulumEnv(env_config=invalid)

    def test_cart_mass_changes_cart_acceleration_under_same_force(self):
        light = env_config(cart_mass=0.003, angular_friction=0.0, cart_friction=0.0)
        heavy = env_config(cart_mass=0.03, angular_friction=0.0, cart_friction=0.0)
        state = np.array([0.0, math.pi / 2, math.pi / 2, 0.0, 0.0, 0.0])

        env_light = PendulumEnv(reward_manager=RewardManager(light), env_config=light)
        env_heavy = PendulumEnv(reward_manager=RewardManager(heavy), env_config=heavy)
        light_derivative = env_light.rhs(state, 0.0, env_light.parameter_vals, lambda _state: 0.2)
        heavy_derivative = env_heavy.rhs(state, 0.0, env_heavy.parameter_vals, lambda _state: 0.2)

        self.assertGreater(abs(light_derivative[3]), abs(heavy_derivative[3]))

    def test_bob_mass_changes_pendulum_dynamics_without_overwriting_cart_mass(self):
        light_bob = env_config(cart_mass=0.02, bob_mass=0.002, angular_friction=0.0, cart_friction=0.0)
        heavy_bob = env_config(cart_mass=0.02, bob_mass=0.02, angular_friction=0.0, cart_friction=0.0)
        state = np.array([0.0, 0.4, -0.2, 0.0, 0.0, 0.0])

        env_light = PendulumEnv(reward_manager=RewardManager(light_bob), env_config=light_bob)
        env_heavy = PendulumEnv(reward_manager=RewardManager(heavy_bob), env_config=heavy_bob)
        light_derivative = env_light.rhs(state, 0.0, env_light.parameter_vals, lambda _state: 0.1)
        heavy_derivative = env_heavy.rhs(state, 0.0, env_heavy.parameter_vals, lambda _state: 0.1)

        self.assertEqual(0.02, env_light.cart_mass)
        self.assertEqual(0.02, env_heavy.cart_mass)
        self.assertGreater(np.max(np.abs(light_derivative - heavy_derivative)), 1e-6)

    def test_cart_velocity_is_not_damped_when_only_angular_velocity_damping_is_enabled(self):
        custom = env_config(
            angular_friction=0.0,
            cart_friction=0.0,
            angular_velocity_damping=5.0,
        )
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.current_state[:] = np.array([0.0, math.pi / 2, math.pi / 2, 1.25, 0.0, 0.0])
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.0)

        self.assertEqual(1.25, env.current_state[3])

    def test_angular_velocity_damping_applies_only_to_angular_velocity_indices(self):
        custom = env_config(
            angular_friction=0.0,
            cart_friction=0.0,
            angular_velocity_damping=5.0,
        )
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.current_state[:] = np.array([0.0, math.pi / 2, math.pi / 2, 0.0, 1.0, -2.0])
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.0)

        self.assertEqual(0.0, env.current_state[3])
        self.assertLess(abs(env.current_state[4]), 1.0)
        self.assertLess(abs(env.current_state[5]), 2.0)

    def test_finite_derivatives_and_next_state_for_zero_force_step(self):
        env = PendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")

        derivative = env.rhs(env.current_state, env.current_time, env.parameter_vals, lambda _state: 0.0)
        next_state, _reward, _terminated, _truncated, _info = env.step(0.0)

        self.assertTrue(np.all(np.isfinite(derivative)))
        self.assertTrue(np.all(np.isfinite(next_state)))

    def test_env_rejects_non_two_node_configuration(self):
        with self.assertRaises(ValueError):
            PendulumEnv(num_nodes=3, env_config=config)

        invalid_config = dict(config)
        invalid_config["num_nodes"] = 3
        with self.assertRaises(ValueError):
            PendulumEnv(env_config=invalid_config)

    def test_observation_and_physical_state_are_two_node_only(self):
        env = PendulumEnv(
            reward_manager=RewardManager(config),
            render_mode=None,
            env_config=config,
        )

        observation = env.reset(episode_mode="capture_vertical")
        physical = env.get_physical_state()

        self.assertEqual(2, env.n)
        self.assertEqual(52, len(observation))
        self.assertEqual((11,), physical.shape)

    def test_max_steps_is_truncation_not_termination(self):
        custom = dict(config)
        custom["max_steps"] = 1
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            max_steps=1,
            env_config=custom,
        )
        env.reset(episode_mode="down_to_up")

        _state, _reward, terminated, truncated, info = env.step(0.0)

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual("max_steps", info["termination_reason"])

    def test_cart_limit_is_penalized_without_immediate_termination(self):
        env = PendulumEnv(
            reward_manager=RewardManager(config),
            env_config=config,
        )
        env.reset(episode_mode="down_to_up")
        env.current_state[0] = 1.70
        derivative = np.zeros_like(env.current_state)
        derivative[0] = 20.0
        env.rhs = mock.Mock(return_value=derivative)

        _state, reward, terminated, truncated, info = env.step(0.0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["hit_cart_limit"])
        self.assertEqual(1, info["cart_limit_streak"])
        self.assertLess(reward, 0.0)
        self.assertLess(info["reward_components"]["cart_limit_step_penalty"], 0.0)
        self.assertIsNone(info["termination_reason"])

    def test_border_action_blocks_push_toward_wall_in_margin(self):
        custom = dict(config)
        custom["cart_border_action_margin"] = 0.10
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        cart_center_limit = env._cart_center_limit()
        env.current_state[0] = cart_center_limit * 0.99
        env.applied_force = 0.0
        max_action = float(custom["max_action"])

        def rhs(_state, _time, _parameters, force_function):
            derivative = np.zeros_like(env.current_state)
            derivative[0] = 5.0 * force_function(_state)
            return derivative

        env.rhs = rhs
        _state, _reward, terminated, truncated, info = env.step(max_action)
        self.assertLess(info["cart_border_action_scale"], 1.0)
        self.assertAlmostEqual(max_action, info["policy_action"])
        self.assertFalse(info["hit_cart_limit"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_border_action_keeps_pull_away_from_wall_at_full_strength(self):
        custom = dict(config)
        custom["cart_border_action_margin"] = 0.10
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        cart_center_limit = env._cart_center_limit()
        env.current_state[0] = cart_center_limit * 0.99
        max_action = float(custom["max_action"])
        action, scale = env._limit_action_near_cart_border(-max_action, cart_center_limit * 0.99)
        self.assertAlmostEqual(-max_action, action)
        self.assertAlmostEqual(1.0, scale)

    def test_cart_limit_streak_eventually_terminates(self):
        custom = dict(config)
        custom["cart_limit_termination_steps"] = 2
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )
        env.reset(episode_mode="down_to_up")
        env.current_state[0] = 1.70
        derivative = np.zeros_like(env.current_state)
        derivative[0] = 20.0
        env.rhs = mock.Mock(return_value=derivative)

        _state, _reward, terminated, truncated, info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _state, reward, terminated, truncated, info = env.step(0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual("cart_limit_streak", info["termination_reason"])
        self.assertLessEqual(
            info["reward_components"]["cart_failure_penalty"],
            custom["cart_failure_penalty"],
        )
        self.assertLess(reward, custom["cart_limit_step_penalty"])

    def test_immediate_wall_policy_scores_below_safe_baselines(self):
        custom = env_config(
            max_steps=30,
            cart_limit_termination_steps=1,
            angular_friction=0.0005,
            cart_friction=0.1,
        )
        max_action = float(custom["max_action"])
        rng = np.random.default_rng(123)

        initial_x = 1.65
        wall_return = self._rollout_return(custom, lambda _step, _state: max_action, initial_x=initial_x)
        zero_return = self._rollout_return(custom, lambda _step, _state: 0.0, initial_x=initial_x)
        random_safe_return = self._rollout_return(
            custom,
            lambda _step, _state: float(rng.uniform(-0.1 * max_action, 0.1 * max_action)),
            initial_x=initial_x,
        )
        sinus_return = self._rollout_return(
            custom,
            lambda step, _state: float(0.2 * max_action * math.sin(0.4 * step)),
            initial_x=initial_x,
        )

        self.assertLess(wall_return, zero_return)
        self.assertLess(wall_return, random_safe_return)
        self.assertLess(wall_return, sinus_return)

    def test_terminal_rail_components_reflect_full_failure_penalty(self):
        custom = env_config(cart_limit_termination_steps=1)
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="down_to_up")
        env.current_state[0] = 1.70
        derivative = np.zeros_like(env.current_state)
        derivative[0] = 20.0
        env.rhs = mock.Mock(return_value=derivative)

        _state, reward, terminated, _truncated, info = env.step(0.0)
        components = info["reward_components"]

        self.assertTrue(terminated)
        self.assertEqual(custom["cart_failure_penalty"], components["terminal_failure_penalty"])
        self.assertLess(components["safety_penalty"], 0.0)
        self.assertLess(reward, custom["cart_failure_penalty"] / 2.0)

    def test_capture_vertical_drop_penalty_is_paid_once(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )
        env.reset(episode_mode="capture_vertical")
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        _state, _warmup_reward, _warmup_terminated, _truncated, warmup_info = env.step(0.0)
        _state, first_reward, first_terminated, _truncated, first_info = env.step(0.0)
        _state, second_reward, second_terminated, _truncated, second_info = env.step(0.0)

        self.assertFalse(warmup_info["capture_drop"])
        self.assertFalse(first_terminated)
        self.assertFalse(second_terminated)
        self.assertTrue(first_info["capture_drop"])
        self.assertFalse(second_info["capture_drop"])
        self.assertIsNone(first_info["termination_reason"])
        remaining_fraction = 1.0 - env.capture_drop_step / custom["max_steps"]
        expected_drop_penalty = (
            custom["capture_drop_base_penalty"]
            + custom["capture_drop_remaining_penalty"] * remaining_fraction
        )
        self.assertAlmostEqual(
            expected_drop_penalty,
            first_info["reward_components"]["capture_drop_penalty"],
        )
        self.assertAlmostEqual(
            expected_drop_penalty,
            first_info["reward_components"]["terminal_failure_penalty"],
        )
        self.assertEqual(0.0, second_info["reward_components"]["capture_drop_penalty"])
        self.assertLess(first_reward, 0.0)

    def test_capture_vertical_truncates_after_post_drop_horizon(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        custom["capture_drop_truncation_steps"] = 2
        custom["capture_drop_terminates_episode"] = True
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )
        env.reset(episode_mode="capture_vertical")
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.0)
        _state, _reward, terminated, truncated, drop_info = env.step(0.0)
        self.assertTrue(drop_info["capture_drop"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _state, _reward, terminated, truncated, first_after_info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(first_after_info["capture_drop"])

        _state, _reward, terminated, truncated, end_info = env.step(0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual("capture_drop_failure", end_info["termination_reason"])

    def test_capture_vertical_recovery_cancels_post_drop_termination(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        custom["capture_drop_truncation_steps"] = 5
        custom["capture_drop_recovery_steps"] = 2
        custom["capture_drop_terminates_episode"] = True
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        env.step(0.0)
        _state, _reward, terminated, _truncated, drop_info = env.step(0.0)
        self.assertTrue(drop_info["capture_drop"])
        self.assertFalse(terminated)
        for _ in range(2):
            env.current_state[1] = math.pi / 2
            env.current_state[2] = math.pi / 2
            _state, _reward, terminated, _truncated, recovery_info = env.step(0.0)
            self.assertFalse(terminated)
        self.assertTrue(recovery_info["capture_drop_recovered"])
        self.assertIsNone(env.capture_drop_step)
        for _ in range(6):
            env.current_state[1] = math.pi / 2
            env.current_state[2] = math.pi / 2
            _state, _reward, terminated, _truncated, _info = env.step(0.0)
            self.assertFalse(terminated)

    def test_capture_vertical_redrop_terminates_without_second_penalty(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        custom["capture_drop_recovery_steps"] = 2
        custom["capture_drop_terminates_episode"] = True
        custom["capture_drop_redrop_terminates"] = True
        env = PendulumEnv(reward_manager=RewardManager(custom), env_config=custom)
        env.reset(episode_mode="capture_vertical")
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        env.step(0.0)
        env.step(0.0)
        for _ in range(2):
            env.current_state[1] = math.pi / 2
            env.current_state[2] = math.pi / 2
            env.step(0.0)
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        _state, reward, terminated, truncated, redrop_info = env.step(0.0)
        self.assertTrue(redrop_info["capture_redrop"])
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual("capture_drop_redrop", redrop_info["termination_reason"])
        self.assertEqual(0.0, redrop_info["reward_components"]["capture_drop_penalty"])
        self.assertGreaterEqual(reward, 0.0)

    def test_capture_vertical_continues_after_drop_when_termination_disabled(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        custom["capture_drop_truncation_steps"] = 2
        custom["capture_drop_terminates_episode"] = False
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )
        env.reset(episode_mode="capture_vertical")
        env.current_state[1] = -math.pi / 2
        env.current_state[2] = -math.pi / 2
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.0)
        _state, _reward, terminated, truncated, drop_info = env.step(0.0)
        self.assertTrue(drop_info["capture_drop"])
        self.assertFalse(terminated)

        for _ in range(4):
            _state, _reward, terminated, truncated, _info = env.step(0.0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)

    def test_down_to_up_does_not_get_capture_drop_penalty(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        env = PendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )
        env.reset(episode_mode="down_to_up")
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        _state, _reward, _terminated, _truncated, info = env.step(0.0)

        self.assertFalse(info["capture_drop"])
        self.assertEqual(0.0, info["reward_components"]["capture_drop_penalty"])

    def test_default_training_probabilities_are_not_rewritten(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)

        probabilities = trainer._episode_mode_probabilities(episode=10_000)

        self.assertEqual(config["episode_mode_probabilities"], probabilities)

    def test_stable_hold_does_not_terminate_episode(self):
        env = PendulumEnv(
            reward_manager=RewardManager(config),
            env_config=config,
        )
        env.reset(episode_mode="capture_vertical")
        components = {
            "reward": 0.0,
            "target_score": 1.0,
            "effective_target_score": 1.0,
            "in_target": 1.0,
            "end_y": 0.6,
        }
        env.reward_manager.evaluate_transition = mock.Mock(
            side_effect=[
                RewardResult(
                    reward=1.0,
                    components=dict(components),
                    capture_started=True,
                    hold_streak=1,
                    success=False,
                ),
                RewardResult(
                    reward=1.0,
                    components=dict(components),
                    capture_started=True,
                    hold_streak=2,
                    success=False,
                ),
                RewardResult(
                    reward=1.0,
                    components=dict(components),
                    capture_started=True,
                    hold_streak=3,
                    success=False,
                ),
            ]
        )
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        _state, _reward, terminated, truncated, info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["entered_success"])

        _state, reward, terminated, truncated, info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["entered_success"])
        self.assertFalse(env.success_achieved)
        self.assertAlmostEqual(1.0, reward, delta=0.01)

        _state, _reward, terminated, truncated, info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["entered_success"])
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
