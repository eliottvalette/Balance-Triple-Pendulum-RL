import math
import unittest

import numpy as np

from config import config
from reward import RewardManager
from train import TriplePendulumTrainer
from tp_env import TriplePendulumEnv


class TwoNodeContractTests(unittest.TestCase):
    def test_env_rejects_non_two_node_configuration(self):
        with self.assertRaises(ValueError):
            TriplePendulumEnv(num_nodes=3)

        invalid_config = dict(config)
        invalid_config["num_nodes"] = 3
        with self.assertRaises(ValueError):
            TriplePendulumEnv(env_config=invalid_config)

    def test_observation_and_physical_state_are_two_node_only(self):
        env = TriplePendulumEnv(reward_manager=RewardManager(), render_mode=None)

        observation = env.reset(episode_mode="capture_vertical")
        physical_state = env.get_physical_state()

        self.assertEqual(2, env.n)
        self.assertEqual(53, len(observation))
        self.assertEqual((11,), physical_state.shape)

    def test_reward_uses_cart_velocity_separately_from_angular_velocity(self):
        reward_manager = RewardManager()
        physical_state = np.array(
            [
                0.0,
                math.pi / 2,
                math.pi / 2,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0 / 3.0,
                0.0,
                2.0 / 3.0,
            ],
            dtype=float,
        )

        _reward, components, _terminated = reward_manager.evaluate(
            physical_state,
            action=0.0,
            phase=1,
        )

        self.assertEqual(1.0, components["in_target"])
        self.assertAlmostEqual(0.0, components["velocity_penalty"])

    def test_capture_reward_penalizes_dropping_after_capture(self):
        reward_manager = RewardManager()
        fallen_center_state = np.array(
            [
                0.0,
                -math.pi / 2,
                -math.pi / 2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0 / 3.0,
                0.0,
                -2.0 / 3.0,
            ],
            dtype=float,
        )

        reward, components, _terminated = reward_manager.evaluate(
            fallen_center_state,
            action=0.0,
            phase=1,
            best_target_score=1.0,
        )

        self.assertGreater(components["capture_height_penalty"], 0.0)
        self.assertGreater(components["capture_lost_penalty"], 0.0)
        self.assertGreater(components["capture_rest_penalty"], 0.0)
        self.assertLess(reward, -2.0)

    def test_default_training_config_is_capture_vertical_only(self):
        cfg = dict(config)
        cfg["load_models"] = False
        trainer = TriplePendulumTrainer(cfg)

        probabilities = trainer._episode_mode_probabilities(episode=10_000)

        self.assertEqual(0.0, probabilities["down_to_up"])
        self.assertEqual(1.0, probabilities["capture_vertical"])
        self.assertEqual(0.0, probabilities["fold_to_up"])
        self.assertEqual(0.0, probabilities["up_to_fold"])


if __name__ == "__main__":
    unittest.main()
