# Triple Pendulum Reinforcement Learning

**Overview**

This project explores control of a triple pendulum mounted on a cart using deep reinforcement learning (RL). The environment models the system dynamics with symbolic equations derived via Kane's method. The agent learns to apply a horizontal force to the cart to keep the pendulum in specified configurations.

**Pendulum Dynamics**

Let $q_i$ denote the angles of the links with respect to the vertical and $u_i$ their angular velocities. The state vector is $s = [x, q_1, q_2, q_3, u_1, u_2, u_3]$ where $x$ is the cart position. The dynamics are obtained from the generalized coordinates and velocities. In matrix form
\[
M(q) \dot{u} = F(q, u) + B f,
\]
where $M$ is the mass matrix, $F$ collects gravitational and friction terms and $f$ is the applied force. The right-hand side is computed symbolically, then evaluated numerically during simulation. The step update is
\[
\dot{s} = [\dot{x}, u_1, u_2, u_3, \dot{u}_1, \dot{u}_2, \dot{u}_3],\qquad s_{t+1} = s_t + \dot{s}\,\Delta t.
\]

**Reward Shaping**

The reward combines several components:

- *Upright reward*: encourages the end node to exceed a height threshold. If the pendulum remains upright for $n$ steps, the reward is scaled by $\min(\alpha^{n/10}, r_{\max})$ with base $\alpha=1.15$.
- *Cart position penalty*: $w_x x^2$ discourages drifting.
- *Alignment penalty*: $w_a\bigl(1-\cos(q_1-q_2) + 1-\cos(q_2-q_3)\bigr)/2$ promotes aligned segments.
- *Stability penalty*: based on angular velocities and node velocity.
- *MSE penalty*: distance from a predefined target state.

The final reward is roughly
\[
R = R_{\text{upright}} - w_x x^2 - w_{\text{align}} A - w_{\text{stab}} S - w_{\text{mse}}E,
\]
where $A,S,E$ denote alignment, stability and mean squared error terms.

**Observation Features**

Beyond the raw positions and velocities, the environment appends engineered features such as relative angles, kinetic energy approximations and indicators on whether the pendulum has remained upright. These features help the policy to infer long-term trends.

**RL Strategy**

The agent follows an actor--critic approach inspired by Deep Deterministic Policy Gradient (DDPG). The actor $\pi_\theta(s)$ outputs a continuous action in $[-1,1]$. The critic $Q_\phi(s,a)$ estimates expected returns. Experience tuples are stored in a replay buffer and sampled to update the networks. Target values are computed as
\[
Q_\text{target} = r + \gamma (1-d)\, Q_\phi(s', \pi_\theta(s')).
\]
Gradients are clipped to stabilize training.

Exploration uses Ornstein–Uhlenbeck noise
\[
\xi_{t+1} = \xi_t + \theta(\mu-\xi_t)\,\Delta t + \sigma \sqrt{\Delta t}\,\mathcal N(0,1),
\]
combined with an $\varepsilon$-greedy strategy during early episodes. The noise encourages smooth actions while random episodes help escaping local optima.

**Network Architecture**

Both actor and critic networks employ layer normalization and multiple fully connected layers. Skip connections allow the actor to reuse the input state at deeper layers. The critic receives the action concatenated with the state.

**Metrics**

Training statistics such as episode reward and losses are logged and plotted. Reward components can be visualized individually to understand which terms drive learning progress.

