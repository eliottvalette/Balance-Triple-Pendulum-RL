import math
import unittest
from unittest import mock

import numpy as np

from config import config, validate_config
from metrics import MetricsTracker
from reward import RewardManager, RewardResult
from train import ReplayBuffer, TriplePendulumTrainer
from tp_env import TriplePendulumEnv


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
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(RuntimeError, "before reset"):
            env.step(0.0)

    def test_out_of_range_action_raises_instead_of_being_clipped(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")

        with self.assertRaisesRegex(ValueError, "action"):
            env.step(config["max_action"] + 0.01)

    def test_action_near_rail_is_not_silently_replaced(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")
        env.current_state[0] = 1.66
        env.rhs = mock.Mock(return_value=np.zeros_like(env.current_state))

        env.step(0.2)

        self.assertEqual(0.2, env.previous_action)
        env.rhs.assert_called_once()

    def test_nonfinite_dynamics_raise_immediately(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)
        env.reset(episode_mode="down_to_up")
        env.rhs = mock.Mock(return_value=np.full_like(env.current_state, np.nan))

        with self.assertRaisesRegex(FloatingPointError, "non-finite derivative"):
            env.step(0.0)

    def test_empty_environment_config_is_not_replaced_by_global_config(self):
        with self.assertRaisesRegex(ValueError, "missing required config"):
            TriplePendulumEnv(env_config={})

    def test_environment_uses_injected_gravity(self):
        custom = dict(config)
        custom["gravity"] = 1.234
        env = TriplePendulumEnv(
            reward_manager=RewardManager(custom),
            env_config=custom,
        )

        self.assertEqual(1.234, env.parameter_vals[0])

    def test_invalid_phase_raises(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(ValueError, "phase"):
            env.reset(phase=0)

    def test_reset_requires_an_explicit_mode_or_phase(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)

        with self.assertRaisesRegex(ValueError, "episode_mode or phase"):
            env.reset()

    def test_replay_buffer_rejects_short_batch(self):
        buffer = ReplayBuffer(capacity=10)
        buffer.push(np.zeros(2), 0.0, 0.0, np.zeros(2), False)

        with self.assertRaisesRegex(ValueError, "batch"):
            buffer.sample(2)

    def test_replay_buffer_rejects_nonfinite_transition(self):
        buffer = ReplayBuffer(capacity=10)

        with self.assertRaises(FloatingPointError):
            buffer.push(np.array([np.nan]), 0.0, 0.0, np.zeros(1), False)

    def test_missing_saved_model_is_not_ignored(self):
        trainer = object.__new__(TriplePendulumTrainer)
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

    def test_stationary_bottom_has_zero_swing_up_reward(self):
        state = physical_state()

        result = self.manager.evaluate_transition(
            state,
            state,
            action=0.0,
            phase=1,
            capture_started=False,
            hold_streak=0,
        )

        self.assertAlmostEqual(0.0, result.reward)

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

    def test_hold_threshold_config_is_not_required(self):
        cfg = dict(config)
        cfg.pop("hold_required_steps", None)
        cfg.pop("success_bonus", None)

        self.assertIs(validate_config(cfg), cfg)


class TwoNodeEnvironmentTests(unittest.TestCase):
    def test_env_rejects_non_two_node_configuration(self):
        with self.assertRaises(ValueError):
            TriplePendulumEnv(num_nodes=3, env_config=config)

        invalid_config = dict(config)
        invalid_config["num_nodes"] = 3
        with self.assertRaises(ValueError):
            TriplePendulumEnv(env_config=invalid_config)

    def test_observation_and_physical_state_are_two_node_only(self):
        env = TriplePendulumEnv(
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
        env = TriplePendulumEnv(
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
        env = TriplePendulumEnv(
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
        env = TriplePendulumEnv(
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

    def test_capture_vertical_drop_penalty_is_paid_once(self):
        custom = dict(config)
        custom["capture_drop_grace_steps"] = 1
        env = TriplePendulumEnv(
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
        env = TriplePendulumEnv(
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
        trainer = TriplePendulumTrainer(cfg)

        probabilities = trainer._episode_mode_probabilities(episode=10_000)

        self.assertEqual(config["episode_mode_probabilities"], probabilities)

    def test_swing_up_sinus_mode_uses_sinusoid_before_capture(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = TriplePendulumTrainer(cfg)
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
        trainer = TriplePendulumTrainer(cfg)
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
        trainer = TriplePendulumTrainer(cfg)
        start = cfg["swing_up_sinus_episode_probability_start"]
        end = cfg["swing_up_sinus_episode_probability_end"]
        decay = cfg["swing_up_sinus_episode_decay_episodes"]

        self.assertAlmostEqual(start, trainer._swing_up_sinus_episode_probability(0))
        self.assertAlmostEqual(end, trainer._swing_up_sinus_episode_probability(decay))
        self.assertAlmostEqual(end, trainer._swing_up_sinus_episode_probability(decay * 2))

    def test_capture_phase_uses_actor_after_swing_up(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = TriplePendulumTrainer(cfg)
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
        env = TriplePendulumEnv(
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


if __name__ == "__main__":
    unittest.main()
