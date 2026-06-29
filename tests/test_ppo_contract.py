import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.distributions import Normal

from config import config
from model import PendulumActorPolicy, PendulumValueCritic
from train import PendulumTrainer, RolloutBuffer, compute_gae


def trainer_config(**overrides):
    cfg = dict(config)
    cfg.update(
        load_models=False,
        hidden_dim=16,
        ppo_epochs=2,
        minibatch_size=4,
        render_training=False,
        render_first_episode=False,
        **overrides,
    )
    return cfg


class PPOContractTests(unittest.TestCase):
    def test_fixed_state_evaluation_preserves_training_env_and_global_rng(self):
        trainer = PendulumTrainer(trainer_config())
        trainer.env.reset(episode_mode="down_to_up", seed=91)
        training_state = trainer.env.current_state.copy()
        training_step = trainer.env.num_steps
        random.seed(2026)
        np.random.seed(2026)
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()

        trainer._actor_eval_on_fixed_capture_states()

        self.assertIsNot(trainer.env, trainer.eval_env)
        self.assertTrue(np.array_equal(training_state, trainer.env.current_state))
        self.assertEqual(training_step, trainer.env.num_steps)
        self.assertEqual(python_rng_state, random.getstate())
        restored_numpy_state = np.random.get_state()
        self.assertEqual(numpy_rng_state[0], restored_numpy_state[0])
        self.assertTrue(np.array_equal(numpy_rng_state[1], restored_numpy_state[1]))
        self.assertEqual(numpy_rng_state[2:], restored_numpy_state[2:])

    def test_policy_samples_bounded_actions_and_consistent_log_probs(self):
        torch.manual_seed(7)
        policy = PendulumActorPolicy(5, hidden_dim=16, max_action=0.5)
        states = torch.randn(32, 5)

        actions, log_probs, entropy = policy.sample(states)
        evaluated_log_probs, evaluated_entropy = policy.evaluate_actions(states, actions)

        self.assertEqual((32, 1), actions.shape)
        self.assertTrue(torch.all(actions.abs() <= 0.5))
        self.assertEqual((32,), log_probs.shape)
        self.assertEqual((32,), entropy.shape)
        self.assertTrue(torch.allclose(log_probs, evaluated_log_probs))
        self.assertTrue(torch.allclose(entropy, evaluated_entropy))

    def test_policy_log_prob_includes_tanh_and_action_scale_jacobian(self):
        policy = PendulumActorPolicy(
            3,
            hidden_dim=8,
            max_action=0.5,
            initial_log_std=-0.3,
        )
        states = torch.zeros(2, 3)
        raw_actions = torch.tensor([[0.4], [-0.7]])
        actions = policy.max_action * torch.tanh(raw_actions)

        raw_mean, std = policy(states)
        log_probs, _entropy = policy.evaluate_actions(states, actions)
        expected = (
            Normal(raw_mean, std).log_prob(raw_actions)
            - torch.log(torch.tensor(policy.max_action))
            - torch.log1p(-torch.tanh(raw_actions).square())
        ).sum(dim=-1)

        self.assertTrue(torch.allclose(log_probs, expected, atol=1e-6))

    def test_policy_deterministic_action_is_squashed_raw_mean(self):
        policy = PendulumActorPolicy(3, hidden_dim=8, max_action=0.5)
        states = torch.randn(4, 3)

        raw_mean, std = policy(states)
        expected_action = policy.max_action * torch.tanh(raw_mean)

        self.assertTrue(torch.allclose(policy.deterministic(states), expected_action))
        self.assertTrue(torch.all(policy.deterministic(states).abs() <= 0.5))
        self.assertTrue(torch.all(std > 0.0))

    def test_policy_squash_preserves_gradient_beyond_action_limit(self):
        policy = PendulumActorPolicy(3, hidden_dim=8, max_action=0.5)
        final_layer = policy.mean_network[-1]
        assert isinstance(final_layer, torch.nn.Linear)
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(1.0)

        action = policy.deterministic(torch.zeros(1, 3))
        action.sum().backward()

        self.assertLess(float(action.item()), policy.max_action)
        self.assertGreater(float(final_layer.bias.grad.item()), 0.0)

    def test_policy_evaluates_boundary_actions_without_non_finite_values(self):
        policy = PendulumActorPolicy(3, hidden_dim=8, max_action=0.5)
        states = torch.zeros(2, 3)
        actions = torch.tensor([[policy.max_action], [-policy.max_action]])

        log_probs, entropy = policy.evaluate_actions(states, actions)

        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        self.assertTrue(torch.all(torch.isfinite(entropy)))

    def test_value_critic_returns_one_value_per_state(self):
        critic = PendulumValueCritic(6, hidden_dim=8)

        values = critic(torch.randn(7, 6))

        self.assertEqual((7,), values.shape)

    def test_gae_stops_recursion_at_episode_boundary_and_bootstraps_truncation(self):
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        values = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        next_values = np.array([0.6, 0.7, 4.0], dtype=np.float32)
        dones = np.array([False, True, True])
        terminated = np.array([False, True, False])

        advantages, returns = compute_gae(
            rewards,
            values,
            next_values,
            dones,
            terminated,
            gamma=0.9,
            gae_lambda=0.8,
        )

        self.assertAlmostEqual(2.0 - 0.6, float(advantages[1]), places=6)
        self.assertAlmostEqual(3.0 + 0.9 * 4.0 - 0.7, float(advantages[2]), places=6)
        np.testing.assert_allclose(returns, advantages + values)

    def test_rollout_buffer_exposes_on_policy_batch_and_clears(self):
        buffer = RolloutBuffer()
        buffer.add(
            state=np.zeros(3, dtype=np.float32),
            action=0.1,
            reward=1.0,
            done=False,
            terminated=False,
            value=0.2,
            next_value=0.3,
            log_prob=-0.4,
        )
        buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

        batch = buffer.as_tensors()

        self.assertEqual((1, 3), batch["states"].shape)
        self.assertEqual((1, 1), batch["actions"].shape)
        self.assertEqual((1,), batch["old_log_probs"].shape)
        buffer.clear()
        self.assertEqual(0, len(buffer))

    def test_ppo_update_changes_policy_and_value_then_consumes_rollout(self):
        trainer = PendulumTrainer(trainer_config())
        state = np.zeros(trainer.state_dim, dtype=np.float32)
        for index in range(8):
            state_tensor = torch.as_tensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _entropy = trainer.actor_model.sample(state_tensor)
                value = trainer.critic_model(state_tensor)
            trainer.rollout_buffer.add(
                state=state + index * 0.01,
                action=float(action.item()),
                reward=float(index + 1),
                done=index == 7,
                terminated=index == 7,
                value=float(value.item()),
                next_value=0.0,
                log_prob=float(log_prob.item()),
            )
        trainer.rollout_buffer.compute_advantages(
            gamma=trainer.config["gamma"],
            gae_lambda=trainer.config["gae_lambda"],
        )
        actor_before = [parameter.detach().clone() for parameter in trainer.actor_model.parameters()]
        critic_before = [parameter.detach().clone() for parameter in trainer.critic_model.parameters()]

        metrics = trainer.update_ppo()

        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(actor_before, trainer.actor_model.parameters())
            )
        )
        self.assertTrue(
            any(
                not torch.allclose(before, after)
                for before, after in zip(critic_before, trainer.critic_model.parameters())
            )
        )
        self.assertTrue({"policy_loss", "value_loss", "entropy", "approx_kl"} <= set(metrics))
        self.assertEqual(0, len(trainer.rollout_buffer))

    def test_collect_rollout_stores_policy_statistics(self):
        trainer = PendulumTrainer(
            trainer_config(
                max_steps=4,
                episode_mode_probabilities={
                    "down_to_up": 0.0,
                    "capture_vertical": 1.0,
                    "fold_to_up": 0.0,
                    "up_to_fold": 0.0,
                },
            )
        )

        summary = trainer.collect_rollout(episode=0)

        self.assertEqual(summary["episode_length"], len(trainer.rollout_buffer))
        self.assertEqual(summary["episode_length"], len(trainer.rollout_buffer.log_probs))
        self.assertEqual(summary["episode_length"], len(trainer.rollout_buffer.values))
        self.assertTrue(np.all(np.abs(trainer.rollout_buffer.actions) <= trainer.max_action))

    def test_checkpoint_metadata_is_ppo_and_rejects_incompatible_algorithm(self):
        trainer = PendulumTrainer(trainer_config())
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "checkpoint"
            trainer.save_models(str(prefix), episode=3)
            metadata_path = Path(f"{prefix}_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual("ppo", metadata["algorithm"])
            self.assertEqual("tanh_squashed_gaussian", metadata["policy_distribution"])

            metadata["algorithm"] = "incompatible"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with mock.patch("train.MODEL_LOAD_PREFIX", str(prefix)):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    trainer.load_models()

    def test_checkpoint_rejects_legacy_clamped_gaussian_policy(self):
        trainer = PendulumTrainer(trainer_config())
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "checkpoint"
            trainer.save_models(str(prefix), episode=3)
            metadata_path = Path(f"{prefix}_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            del metadata["policy_distribution"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            with mock.patch("train.MODEL_LOAD_PREFIX", str(prefix)):
                with self.assertRaisesRegex(ValueError, "policy_distribution"):
                    trainer.load_models()


if __name__ == "__main__":
    unittest.main()
