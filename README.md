# Double Pendulum PPO Control

## Table Of Contents

1. [Overview](#1-overview)
2. [Running The Project](#2-running-the-project)
3. [Environment](#3-environment)
4. [Reward And Termination Contract](#4-reward-and-termination-contract)
5. [Reinforcement Learning Formulation](#5-reinforcement-learning-formulation)
6. [Checkpoints And Compatibility](#6-checkpoints-and-compatibility)
7. [Metrics And Audits](#7-metrics-and-audits)
8. [Project Layout](#8-project-layout)
9. [Caveats](#9-caveats)

---

## 1. Overview

This repository trains a model-free controller for a two-link pendulum mounted on a cart.

The current training stack is an on-policy PPO-style actor-critic system:

- stochastic scalar-action actor;
- state-value critic \(V(s)\);
- rollout buffer containing only samples from the current policy;
- old log-probability storage;
- GAE returns;
- clipped PPO objective;
- entropy regularization;
- optional approximate-KL early stop;
- no long replay buffer, target networks, action-value critic, or off-policy deterministic actor-critic update.

The control objective depends on the sampled episode mode. The main tasks are swing-up, vertical capture, and controlled transitions between upright and folded target poses.

---

## 2. Running The Project

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training:

```bash
python train.py
```

Run the interactive environment:

```bash
python tp_env.py
```

Interactive controls:

| Key | Action |
|---|---|
| Left / Right | apply manual cart force within `[-max_action, max_action]` |
| `Space` | reset to `capture_vertical` |
| `P` | toggle phase and reset |
| `S` | print state and reward diagnostics |
| `M` | toggle control by the saved PPO actor from `models/saved/checkpoint_*` |

The Pygame window displays `Control: MANUAL (M)` or `Control: SAVED MODEL (M)`. The window title also changes when model control is toggled.
Interactive resets use the exact centered pose with no initialization noise.

Run the test suite:

```bash
python -m unittest discover tests
```

Run experiment scripts:

```bash
python scripts/run_ablation.py
python scripts/run_pd_gain_sweep.py
python scripts/run_capture_policy_audit.py
python scripts/run_capture_constant_action_audit.py
```

Important defaults in `config.py`:

| Key | Default | Meaning |
|---|---:|---|
| `num_episodes` | `10000` | training episodes |
| `max_steps` | `1000` | rollout horizon per episode |
| `load_models` | `True` | resume from `models/checkpoint_*` |
| `render_training` | `True` | show Pygame rendering during selected episodes |
| `render_every_episodes` | `50` | render cadence |
| `max_action` | `0.2` | scalar force limit |

Set `load_models=False` in `config.py` when training from scratch.

---

## 3. Environment

### 3.1 Physical Model

The environment is a fixed two-node pendulum on a one-dimensional cart. It is implemented in `tp_env.py` with SymPy mechanics for the equations of motion and NumPy for step integration.

The physical state is:

```math
\mathbf{x}_t =
\begin{bmatrix}
x & q_1 & q_2 & \dot{x} & \dot{q}_1 & \dot{q}_2
\end{bmatrix}^{\top}.
```

The control input is a horizontal cart force:

```math
u_t \in [-u_{\max}, u_{\max}],
\qquad
u_{\max} = \texttt{max\_action}.
```

The symbolic dynamics solve:

```math
M(q)\ddot{q} = F(q,\dot{q},u),
```

then integrate with fixed step:

```math
\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t\, f(\mathbf{x}_t,u_t),
\qquad
\Delta t = 0.01.
```

The applied force is smoothed before entering the dynamics:

```math
f_t = f_{t-1} + 0.1\,(u_t - f_{t-1}).
```

Cart motion is bounded by the rail limits. Actions pointing further into a rail are attenuated near the border, and repeated rail contact can terminate the episode.

### 3.2 Episode Modes

The configured modes are:

| Mode | Initial Pose | Phase Behavior |
|---|---|---|
| `down_to_up` | pendulum down | swing up to vertical |
| `capture_vertical` | near upright | maintain vertical capture |
| `fold_to_up` | folded target | switch to upright target |
| `up_to_fold` | upright target | switch to folded target |

Mode sampling uses `episode_mode_probabilities` from `config.py`. Adaptive curriculum can be enabled with `adaptive_curriculum_enabled`.

### 3.3 State Representation

The policy observation is:

```math
\mathbf{s}_t \in \mathbb{R}^{52}.
```

It combines normalized physical state, Cartesian kinematics, phase context, target geometry, and progress memory.

| Block | Dim | Content |
|---|---:|---|
| Cart and angular dynamics | 8 | \(x,\dot{x},\sin q_i,\cos q_i,\dot{q}_i\) |
| Link positions | 4 | \((x_1,y_1,x_2,y_2)\) |
| Link velocities | 4 | \((\dot{x}_1,\dot{y}_1,\dot{x}_2,\dot{y}_2)\) |
| Previous control | 2 | smoothed force, previous action |
| Phase context | 6 | phase, capture flag, initial pose, normalized time, time-to-switch, switched flag |
| Active target errors | 3 | active height, horizontal, and shape errors |
| Upright target errors | 3 | phase-1 height, horizontal, alignment errors |
| Folded target errors | 4 | phase-2 link and endpoint errors |
| Active target points | 4 | desired \((x_1,y_1,x_2,y_2)\) for current phase |
| Active target deltas | 4 | active target minus current points |
| Next target points | 4 | desired points for the opposite phase |
| Next target deltas | 4 | next target minus current points |
| Motion and hold memory | 2 | endpoint speed, hold progress |

### 3.4 Action Space

The action is a scalar continuous force:

```math
a_t \in [-a_{\max}, a_{\max}].
```

During training the actor samples from a tanh-squashed Gaussian. During evaluation it uses the deterministic mean action.

---

## 4. Reward And Termination Contract

Rewards are computed in `reward.py` from the transition between physical states.

### 4.1 Target Score

For the upright target:

```math
S_{\mathrm{up}} =
\exp\left(
-8e_h^2 -4e_{y_1}^2 -2e_x^2 -2e_{\mathrm{align}}
\right),
```

where:

```math
e_h = h_{\max} - y_2,
\qquad
e_{y_1} = l - y_1,
\qquad
e_x = x_2 - x,
\qquad
e_{\mathrm{align}} = 1 - \cos(q_1-q_2).
```

For the folded target:

```math
S_{\mathrm{fold}} =
\exp\left(
-8e_{y_1}^2 -10y_2^2 -1.5e_x^2
\right).
```

### 4.2 Swing-Up Potential

Before capture, the reward uses potential shaping:

```math
r_t^{\mathrm{swing}} =
\gamma\Phi(s_{t+1}) - \Phi(s_t),
```

with:

```math
\Phi(s) =
C_{\mathrm{cart}}(x)
\left(
w_E E_{\mathrm{score}}(s)
+ w_H H_{\mathrm{score}}(s)
+ w_C
\right).
```

`RewardManager.verify_cart_termination_is_suboptimal()` checks that the rail terminal penalty is strictly worse than the lower bound of the shaped continuation value.

### 4.3 Capture Reward

After capture, or in folded phase, the reward favors target quality, slow angular motion, cart safety, low action magnitude, and sustained hold:

```math
r_t^{\mathrm{capture}} =
w_Q Q_t
+ b_{\mathrm{entry}}
+ b_{\mathrm{hold}}
+ w_M S_t
- w_D \max(0, S_{t-1}-S_t)
+ b_{\mathrm{target}}
+ b_{\mathrm{center}}.
```

The quality term is:

```math
Q_t =
S_t
\cdot
\frac{1}{1 + (\omega_t / \omega_{\max})^2}
\cdot
C_{\mathrm{cart}}(x_t)
\cdot
\frac{1}{1 + (a_t/a_{\max})^2}.
```

### 4.4 Regularization And Safety

The environment adds penalties for low inactive poses, low energy, sleeping below the track, rail proximity, rail hits, action magnitude, action delta, and saturation:

```math
r_t =
r_t^{\mathrm{task}}
+ r_t^{\mathrm{safety}}
- \lambda_a \left(\frac{a_t}{a_{\max}}\right)^2
- \lambda_{\Delta a}
\left(\frac{a_t-a_{t-1}}{a_{\max}}\right)^2
- \lambda_{\mathrm{sat}}\mathbf{1}_{|a_t|\ge 0.95a_{\max}}.
```

### 4.5 Termination

An episode terminates on:

| Reason | Condition |
|---|---|
| `cart_limit_streak` | repeated cart rail contact for `cart_limit_termination_steps` |
| `capture_drop_redrop` | recovered capture drops again when redrop termination is enabled |
| `capture_drop_failure` | capture is not recovered within `capture_drop_truncation_steps` |

An episode truncates with reason `max_steps` when the horizon is reached without terminal failure.

---

## 5. Reinforcement Learning Formulation

### 5.1 Actor

`PendulumActorPolicy` maps state to a Normal distribution in raw action space:

```math
\mu_\theta(s),\sigma_\theta =
\pi_\theta(s),
\qquad
z_t \sim \mathcal{N}(\mu_\theta(s_t),\sigma_\theta^2).
```

The executed action is tanh-squashed and scaled:

```math
a_t = a_{\max}\tanh(z_t).
```

The log-probability includes the tanh and action-scale Jacobian:

```math
\log \pi_\theta(a_t|s_t)
=
\log \mathcal{N}(z_t;\mu_\theta,\sigma_\theta)
-
\log\left|
a_{\max}(1-\tanh^2 z_t)
\right|.
```

The policy uses a learned global `log_std` parameter, clamped before exponentiation.

### 5.2 Critic

`PendulumValueCritic` estimates one scalar value per state:

```math
V_\phi(s_t) \in \mathbb{R}.
```

There is no action-value critic in the training loop.

### 5.3 Rollout Buffer

`RolloutBuffer` stores one current-policy rollout:

```text
state, action, reward, done, terminated, value, next_value, log_prob
```

It is consumed by `update_ppo()` and then cleared. It is not a long-term replay buffer.

### 5.4 GAE

Advantages are computed backward:

```math
\delta_t =
r_t + \gamma(1-\mathbb{1}_{\mathrm{terminated}})V(s_{t+1}) - V(s_t),
```

```math
A_t =
\delta_t
+ \gamma\lambda(1-\mathbb{1}_{\mathrm{done}})A_{t+1}.
```

Returns are:

```math
R_t = A_t + V(s_t).
```

True terminations disable bootstrapping. Time-limit truncations keep the final bootstrap value but stop recursive advantage propagation.

### 5.5 PPO Update

For stored old log-probabilities:

```math
\rho_t(\theta) =
\exp\left(
\log\pi_\theta(a_t|s_t)
-
\log\pi_{\mathrm{old}}(a_t|s_t)
\right).
```

The clipped policy objective is:

```math
L^{\mathrm{clip}}(\theta) =
\mathbb{E}_t
\left[
\min\left(
\rho_t A_t,
\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
```

The actor minimizes:

```math
\mathcal{L}_{\mathrm{actor}}
=
-L^{\mathrm{clip}}(\theta)
- c_H\,\mathcal{H}[\pi_\theta].
```

The critic minimizes:

```math
\mathcal{L}_{\mathrm{critic}}
=
c_V\,\mathbb{E}_t\left[
\left(V_\phi(s_t)-R_t\right)^2
\right].
```

Gradients are clipped with `max_grad_norm`. If `target_kl` is not `None`, PPO epochs stop early when the approximate KL exceeds that threshold.

---

## 6. Checkpoints And Compatibility

Models are saved as separate actor, critic, optimizer, and metadata files:

```text
models/interrupted_actor.pth
models/interrupted_critic.pth
models/interrupted_actor_optimizer.pth
models/interrupted_critic_optimizer.pth
models/interrupted_metadata.json
```

Training resumes from:

```text
models/checkpoint_actor.pth
models/checkpoint_critic.pth
models/checkpoint_actor_optimizer.pth
models/checkpoint_critic_optimizer.pth
models/checkpoint_metadata.json
```

Checkpoint metadata must declare:

```json
{
  "algorithm": "ppo",
  "policy_distribution": "tanh_squashed_gaussian"
}
```

Legacy off-policy checkpoints are intentionally incompatible. Loading fails when metadata does not match the PPO actor-critic contract.

---

## 7. Metrics And Audits

Training records scalar metrics through `MetricsTracker` and can generate plots under `results/`.

Core PPO metrics:

| Metric | Meaning |
|---|---|
| `policy_loss` | clipped PPO policy loss |
| `value_loss` | value regression MSE |
| `entropy` | policy entropy |
| `approx_kl` | approximate KL from old to updated policy |
| `clip_fraction` | fraction of samples outside PPO clip range |
| `kl_early_stop` | whether target-KL stopped the update |

Episode diagnostics include reward, length, hold scores, target score, action statistics, transition direction, and termination reason.

Audit outputs are written as CSV/JSON artifacts in `results/`, for example:

```text
results/ablation_*/episodes.csv
results/ablation_*/summary.csv
results/capture_policy_audit/steps.csv
results/pd_gain_sweep/summary.csv
```

---

## 8. Project Layout

| Path | Purpose |
|---|---|
| `train.py` | PPO trainer, rollout collection, GAE, PPO update, checkpoint IO |
| `model.py` | stochastic actor and value critic |
| `tp_env.py` | double-pendulum cart environment and Pygame rendering |
| `reward.py` | reward shaping, target scores, safety checks |
| `config.py` | hyperparameters, environment constants, validation |
| `metrics.py` | metric storage and plot generation |
| `tests/` | PPO, environment, and audit contract tests |
| `scripts/` | ablation and capture diagnostics |
| `models/` | checkpoint files |
| `results/` | generated metrics and experiment artifacts |

---

## 9. Caveats

- The environment is fixed to two pendulum nodes.
- `train.py` reads hyperparameters directly from `config.py`; there is no command-line training interface.
- Rendering uses Pygame and may require a graphical session.
- Reward shaping is intentionally explicit and component-heavy to keep failures auditable.
- Existing checkpoint files must match the current PPO metadata contract before loading.
