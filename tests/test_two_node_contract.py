import math
import unittest

import numpy as np

from config import config
from reward import RewardManager
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


if __name__ == "__main__":
    unittest.main()
