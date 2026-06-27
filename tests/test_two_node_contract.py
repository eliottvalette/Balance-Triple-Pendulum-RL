import math
import unittest
from unittest import mock

import numpy as np
import torch

from config import config, validate_config
from metrics import MetricsTracker
from reward import RewardManager, RewardResult
from train import ReplayBuffer, PendulumTrainer
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

    def test_replay_buffer_rejects_short_batch(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.push(np.zeros(2), 0.0, 0.0, np.zeros(2), False, False)

        with self.assertRaisesRegex(ValueError, "batch"):
            buffer.sample(2)

    def test_replay_buffer_rejects_nonfinite_transition(self):
        buffer = ReplayBuffer(capacity=10)

        with self.assertRaises(FloatingPointError):
            buffer.push(np.array([np.nan]), 0.0, 0.0, np.zeros(1), False, False)

    def test_replay_buffer_requires_explicit_termination_and_truncation(self):
        buffer = ReplayBuffer(capacity=10)

        with self.assertRaises(TypeError):
            buffer.push(np.zeros(2), 0.0, 0.0, np.zeros(2), False)

    def test_replay_buffer_preserves_truncation_contract_and_bootstrap_mask(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.push(np.array([1.0]), 0.1, 1.0, np.array([2.0]), False, False)
        buffer.push(np.array([3.0]), 0.2, 2.0, np.array([4.0]), True, False)
        buffer.push(np.array([5.0]), 0.3, 3.0, np.array([6.0]), False, True)

        _states, _actions, _rewards, _next_states, terminated, truncated, bootstrap_masks = buffer.sample(3)

        self.assertEqual(1, int(np.sum(terminated)))
        self.assertEqual(1, int(np.sum(truncated)))
        self.assertEqual(1, int(np.sum(bootstrap_masks)))

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
        self.assertLess(result.components["hold_bonus"], 0.0)
        self.assertLess(result.reward, 0.0)

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
        self.assertEqual(53, len(observation))
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

    def test_td_target_uses_explicit_bootstrap_mask(self):
        trainer = object.__new__(PendulumTrainer)
        trainer.config = env_config(gamma=0.5, policy_noise=0.0, noise_clip=0.0)
        trainer.max_action = 1.0

        class ConstantActor:
            def __call__(self, states):
                return torch.zeros((states.shape[0], 1), dtype=states.dtype)

        class ConstantCritic:
            def __call__(self, states, actions):
                return (
                    torch.full((states.shape[0], 1), 10.0, dtype=states.dtype),
                    torch.full((states.shape[0], 1), 5.0, dtype=states.dtype),
                )

        trainer.actor_target = ConstantActor()
        trainer.critic_target = ConstantCritic()

        targets = trainer._compute_td_targets(
            rewards_tensor=torch.tensor([[1.0], [2.0], [3.0]]),
            next_states_tensor=torch.zeros((3, 4)),
            actions_tensor=torch.zeros((3, 1)),
            bootstrap_masks_tensor=torch.tensor([[1.0], [0.0], [0.0]]),
        )

        self.assertTrue(torch.allclose(targets, torch.tensor([[3.5], [2.0], [3.0]])))

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
        self.assertEqual(
            custom["capture_drop_penalty"],
            first_info["reward_components"]["capture_drop_penalty"],
        )
        self.assertEqual(0.0, second_info["reward_components"]["capture_drop_penalty"])
        self.assertLess(first_reward, second_reward)

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

    def test_config_rejects_invalid_sinus_probability_order(self):
        invalid = dict(config)
        invalid["swing_up_sinus_episode_probability_start"] = 0.2
        invalid["swing_up_sinus_episode_probability_end"] = 0.8

        with self.assertRaisesRegex(ValueError, "swing_up_sinus"):
            validate_config(invalid)

    def test_default_training_probabilities_are_not_rewritten(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)

        probabilities = trainer._episode_mode_probabilities(episode=10_000)

        self.assertEqual(config["episode_mode_probabilities"], probabilities)

    def test_swing_up_sinus_mode_uses_sinusoid_before_capture(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)
        with mock.patch.object(trainer, "select_action", return_value=-0.5) as actor_call:
            actions = []
            for step_index in range(120):
                action = trainer._select_collection_action(
                    np.zeros(trainer.state_dim),
                    step_index,
                    capture_started=False,
                    initial_pose_mode="down",
                    swing_period=60.0,
                    swing_phase=0.0,
                    use_sinus_swing_exploration=True,
                )
                actions.append(action)
        actor_call.assert_not_called()
        self.assertLess(min(actions), -0.1)
        self.assertGreater(max(actions), 0.1)

    def test_swing_up_actor_mode_uses_actor_before_capture(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)
        state = np.zeros(trainer.state_dim)
        with mock.patch.object(trainer, "select_action", return_value=0.2) as actor_call:
            action = trainer._select_collection_action(
                state,
                0,
                capture_started=False,
                initial_pose_mode="down",
                swing_period=60.0,
                swing_phase=0.0,
                use_sinus_swing_exploration=False,
            )
        actor_call.assert_called_once_with(state, noise_std=config["swing_up_exploration_noise"])
        self.assertEqual(0.2, action)

    def test_swing_up_sinus_probability_decays_over_window(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)
        start = cfg["swing_up_sinus_episode_probability_start"]
        end = cfg["swing_up_sinus_episode_probability_end"]
        decay = cfg["swing_up_sinus_episode_decay_episodes"]

        self.assertAlmostEqual(start, trainer._swing_up_sinus_episode_probability(0))
        self.assertAlmostEqual(end, trainer._swing_up_sinus_episode_probability(decay))
        self.assertAlmostEqual(end, trainer._swing_up_sinus_episode_probability(decay * 2))

    def test_capture_phase_uses_actor_after_swing_up(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = PendulumTrainer(cfg)
        state = np.zeros(trainer.state_dim)
        with mock.patch.object(trainer, "select_action", return_value=0.2) as actor_call:
            action = trainer._select_collection_action(
                state,
                0,
                capture_started=True,
                initial_pose_mode="down",
                swing_period=60.0,
                swing_phase=0.0,
                use_sinus_swing_exploration=True,
            )
        actor_call.assert_called_once_with(state, noise_std=config["swing_up_capture_noise"])
        self.assertEqual(0.2, action)

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
        self.assertEqual(1.0, reward)

        _state, _reward, terminated, truncated, info = env.step(0.0)
        self.assertFalse(terminated)
        self.assertFalse(info["entered_success"])
        self.assertFalse(truncated)


class TrainerContractTests(unittest.TestCase):
    def _trainer_config(self, **overrides):
        cfg = env_config(
            load_models=False,
            hidden_dim=16,
            batch_size=4,
            buffer_capacity=100,
            policy_noise=0.0,
            noise_clip=0.0,
            policy_delay=2,
            min_non_crash_transitions_for_actor_update=0,
            min_near_capture_transitions_for_actor_update=0,
        )
        cfg.update(overrides)
        return cfg

    def _fill_replay(self, trainer, count, *, termination_reason=None):
        state = np.zeros(trainer.state_dim, dtype=np.float32)
        next_state = np.ones(trainer.state_dim, dtype=np.float32) * 0.1
        for index in range(count):
            trainer.memory.push(
                state + index * 0.01,
                0.05,
                float(index),
                next_state + index * 0.01,
                False,
                False,
                metadata={
                    "episode_mode": "down_to_up",
                    "termination_reason": termination_reason,
                    "capture_started": index % 2 == 0,
                    "target_score": 0.8 if index % 2 == 0 else 0.1,
                },
            )

    @staticmethod
    def _parameters_changed(before, module):
        return any(
            not torch.allclose(previous, current)
            for previous, current in zip(before, module.parameters())
        )

    @staticmethod
    def _clone_parameters(module):
        return [param.detach().clone() for param in module.parameters()]

    def test_replay_buffer_sample_shapes_and_masks(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.push(np.zeros(3), 0.1, 1.0, np.ones(3), False, False)
        buffer.push(np.ones(3), -0.1, 2.0, np.zeros(3), True, False)

        states, actions, rewards, next_states, terminated, truncated, bootstrap_masks = buffer.sample(2)

        self.assertEqual((2, 3), states.shape)
        self.assertEqual((2, 1), actions.shape)
        self.assertEqual((2,), rewards.shape)
        self.assertEqual((2, 3), next_states.shape)
        self.assertEqual((2,), terminated.shape)
        self.assertEqual((2,), truncated.shape)
        self.assertEqual((2,), bootstrap_masks.shape)
        self.assertEqual(1.0, float(np.max(bootstrap_masks)))
        self.assertEqual(0.0, float(np.min(bootstrap_masks)))

    def test_target_networks_do_not_receive_gradients(self):
        trainer = PendulumTrainer(self._trainer_config(policy_delay=1))
        self._fill_replay(trainer, trainer.config["batch_size"])

        trainer.update_networks()

        self.assertTrue(all(param.grad is None for param in trainer.actor_target.parameters()))
        self.assertTrue(all(param.grad is None for param in trainer.critic_target.parameters()))

    def test_actor_respects_policy_delay_and_critic_updates_every_step(self):
        trainer = PendulumTrainer(self._trainer_config(policy_delay=2))
        self._fill_replay(trainer, trainer.config["batch_size"])
        actor_before = self._clone_parameters(trainer.actor_model)
        critic_before = self._clone_parameters(trainer.critic_model)

        first = trainer.update_networks()

        self.assertEqual(0.0, first["actor_loss"])
        self.assertFalse(self._parameters_changed(actor_before, trainer.actor_model))
        self.assertTrue(self._parameters_changed(critic_before, trainer.critic_model))

        actor_after_first = self._clone_parameters(trainer.actor_model)
        critic_after_first = self._clone_parameters(trainer.critic_model)
        second = trainer.update_networks()

        self.assertNotEqual(0.0, second["actor_loss"])
        self.assertTrue(self._parameters_changed(actor_after_first, trainer.actor_model))
        self.assertTrue(self._parameters_changed(critic_after_first, trainer.critic_model))

    def test_polyak_update_is_deterministic_and_slow(self):
        trainer = object.__new__(PendulumTrainer)
        trainer.config = env_config(polyak_tau=0.1)
        online = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with torch.no_grad():
            for param in online.parameters():
                param.fill_(1.0)
            for param in target.parameters():
                param.zero_()

        trainer._polyak_update(online, target)

        for param in target.parameters():
            self.assertTrue(torch.allclose(param, torch.full_like(param, 0.1)))

    def test_replay_buffer_reports_diagnostics(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.push(
            np.zeros(2),
            0.49,
            1.0,
            np.ones(2),
            False,
            False,
            metadata={
                "episode_mode": "down_to_up",
                "termination_reason": None,
                "capture_started": True,
                "target_score": 0.9,
            },
        )
        buffer.push(
            np.zeros(2),
            -0.49,
            -2.0,
            np.ones(2),
            True,
            False,
            metadata={
                "episode_mode": "capture_vertical",
                "termination_reason": "cart_limit_streak",
                "capture_started": False,
                "target_score": 0.1,
            },
        )

        diagnostics = buffer.diagnostics(max_action=0.5)

        self.assertEqual(2, diagnostics["size"])
        self.assertEqual(10, len(diagnostics["action_histogram"]))
        self.assertEqual({"down_to_up": 1, "capture_vertical": 1}, diagnostics["episode_mode_distribution"])
        self.assertEqual({"cart_limit_streak": 1}, diagnostics["termination_reasons"])
        self.assertEqual(1, diagnostics["non_crash_transitions"])
        self.assertEqual(1, diagnostics["near_capture_non_crash_transitions"])
        self.assertGreater(diagnostics["saturation_fraction"], 0.0)

    def test_replay_buffer_can_be_seeded_with_sinus_exploration(self):
        trainer = PendulumTrainer(self._trainer_config(batch_size=2))

        trainer.prefill_replay_buffer(strategy="sinus", num_steps=5, seed=11)

        self.assertEqual(5, len(trainer.memory))
        diagnostics = trainer.replay_diagnostics()
        self.assertEqual(5, diagnostics["size"])
        self.assertIn("down_to_up", diagnostics["episode_mode_distribution"])

    def test_actor_update_can_require_non_crash_transitions(self):
        trainer = PendulumTrainer(
            self._trainer_config(
                policy_delay=1,
                min_non_crash_transitions_for_actor_update=10,
            )
        )
        self._fill_replay(
            trainer,
            trainer.config["batch_size"],
            termination_reason="cart_limit_streak",
        )
        actor_before = self._clone_parameters(trainer.actor_model)
        critic_before = self._clone_parameters(trainer.critic_model)

        result = trainer.update_networks()

        self.assertEqual(0.0, result["actor_loss"])
        self.assertFalse(self._parameters_changed(actor_before, trainer.actor_model))
        self.assertTrue(self._parameters_changed(critic_before, trainer.critic_model))

    def test_actor_update_can_require_near_capture_transitions(self):
        trainer = PendulumTrainer(
            self._trainer_config(
                policy_delay=1,
                min_near_capture_transitions_for_actor_update=10,
            )
        )
        self._fill_replay(trainer, trainer.config["batch_size"])
        actor_before = self._clone_parameters(trainer.actor_model)
        critic_before = self._clone_parameters(trainer.critic_model)

        result = trainer.update_networks()

        self.assertEqual(0.0, result["actor_loss"])
        self.assertFalse(self._parameters_changed(actor_before, trainer.actor_model))
        self.assertTrue(self._parameters_changed(critic_before, trainer.critic_model))

    def test_actor_update_runs_after_near_capture_gate_is_satisfied(self):
        trainer = PendulumTrainer(
            self._trainer_config(
                policy_delay=1,
                min_near_capture_transitions_for_actor_update=2,
            )
        )
        self._fill_replay(trainer, trainer.config["batch_size"])
        actor_before = self._clone_parameters(trainer.actor_model)

        result = trainer.update_networks()

        self.assertNotEqual(0.0, result["actor_loss"])
        self.assertTrue(self._parameters_changed(actor_before, trainer.actor_model))


if __name__ == "__main__":
    unittest.main()
