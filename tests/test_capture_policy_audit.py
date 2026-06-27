import unittest

from scripts.run_capture_policy_audit import compute_episode_metrics, policy_action


class CapturePolicyAuditTests(unittest.TestCase):
    def test_metrics_detect_drop_after_first_near_target_contact(self):
        rows = [
            self._step(target_score=0.20, in_target=False),
            self._step(target_score=0.80, in_target=True),
            self._step(target_score=0.82, in_target=True, capture_drop=True),
            self._step(target_score=0.10, in_target=False),
        ]

        metrics = compute_episode_metrics(rows, max_action=0.5)

        self.assertEqual(1.0, metrics["brief_target_contact"])
        self.assertEqual(1.0, metrics["drop_after_contact"])
        self.assertEqual(1.0, metrics["capture_drop"])

    def test_metrics_report_action_saturation_and_final_hold(self):
        rows = [
            self._step(action=0.5, in_target=True),
            self._step(action=-0.5, in_target=True),
            self._step(action=0.0, in_target=False),
            self._step(action=0.0, in_target=False),
            self._step(action=0.0, in_target=False),
        ]

        metrics = compute_episode_metrics(rows, max_action=0.5)

        self.assertEqual(0.4, metrics["action_saturation"])
        self.assertEqual(0.0, metrics["final_hold"])

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            policy_action("missing", [0.0] * 11, 0, None, 0.5)

    def _step(
        self,
        *,
        action=0.0,
        reward=0.0,
        cart_x=0.0,
        target_score=0.0,
        in_target=False,
        capture_drop=False,
        hit_cart_limit=False,
        angular_speed=0.0,
        termination_reason=None,
    ):
        return {
            "action": action,
            "reward": reward,
            "cart_x": cart_x,
            "target_score": target_score,
            "in_target": in_target,
            "capture_drop": capture_drop,
            "hit_cart_limit": hit_cart_limit,
            "angular_speed": angular_speed,
            "termination_reason": termination_reason,
        }


if __name__ == "__main__":
    unittest.main()
