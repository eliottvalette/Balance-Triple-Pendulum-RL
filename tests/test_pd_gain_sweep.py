import unittest

from scripts.run_pd_gain_sweep import aggregate, pd_action


class PDGainSweepTests(unittest.TestCase):
    def test_pd_action_is_clipped_to_max_action(self):
        physical_state = [0.0, -10.0, -10.0, 0.0, 0.0, 0.0]

        action = pd_action(
            physical_state,
            kp_angle=8.0,
            kd_angle=0.4,
            kp_cart=0.4,
            kd_cart=0.1,
            max_action=0.5,
        )

        self.assertEqual(-0.5, action)

    def test_aggregate_groups_by_gain_fields(self):
        rows = [
            self._row(gain_index=0, success=1.0, final_hold=1.0),
            self._row(gain_index=0, success=0.0, final_hold=0.0),
            self._row(gain_index=1, success=1.0, final_hold=0.5),
        ]

        summaries = aggregate(rows, ["gain_index"])

        by_gain = {row["gain_index"]: row for row in summaries}
        self.assertEqual(0.5, by_gain[0]["success_rate"])
        self.assertEqual(0.5, by_gain[0]["final_hold"])
        self.assertEqual(1.0, by_gain[1]["success_rate"])

    def _row(self, *, gain_index, success, final_hold):
        return {
            "gain_index": gain_index,
            "final_hold": final_hold,
            "success": success,
            "capture_drop": 0.0,
            "cart_hit": 0.0,
            "action_saturation": 0.0,
            "max_abs_cart_x": 0.0,
            "angular_speed_near_target": 0.0,
            "brief_target_contact": 1.0,
            "sustained_hold": 1.0,
            "drop_after_contact": 0.0,
            "action_abs_mean": 0.0,
            "episode_reward": 0.0,
            "episode_length": 200,
        }


if __name__ == "__main__":
    unittest.main()
