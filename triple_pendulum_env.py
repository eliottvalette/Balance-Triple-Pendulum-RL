# triple_pendulum_env.py

import gym
import numpy as np
from gym import spaces
import pygame
import math

class TriplePendulumEnv(gym.Env):
    """
    Custom Gym environment for controlling a cart holding a triple pendulum,
    where the agent applies a horizontal force to stabilize the pendulum upright.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super(TriplePendulumEnv, self).__init__()

        # -----------------------
        # Environment parameters
        # -----------------------
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pend = 0.1   # mass of each pendulum link
        self.length = 0.5      # length of each pendulum link
        self.cart_friction = 0.0
        self.pend_friction = 0.0

        # Maximum force magnitude the agent can apply
        self.force_mag = 10.0

        # Time step
        self.tau = 0.02

        # Limits for cart position and velocity
        self.x_threshold = 2.4
        self.x_dot_threshold = 10.0

        # Angular limits (radians)
        self.theta_threshold_radians = math.pi

        # --------------
        # Action / State
        # --------------
        # Action: continuous cliped force applied to the cart
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)

        # Observation:
        #   [x, x_dot, theta1, theta_dot1, theta2, theta_dot2, theta3, theta_dot3]
        high = np.array([
            self.x_threshold * 2.0,
            self.x_dot_threshold * 2.0,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Internal state
        self.state = None

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 600
        self.cart_y_pos = 300   # y-position of cart in the rendered view
        self.pixels_per_meter = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Small random initial angles near upright
        self.state = np.array([
            0.0,                        # cart x
            0.0,                        # cart velocity
            np.random.uniform(-3, 3),   # theta1
            0.2,                            # theta_dot1
            np.random.uniform(-3, 3),   # theta2
            0.2,                            # theta_dot2
            np.random.uniform(-3, 3),   # theta3
            0.2                             # theta_dot3
        ], dtype=np.float32)

        if self.render_mode == "human":
            self._render_init()

        return self.state, {}

    def step(self, action):
        # Unpack the state
        x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot = self.state
        force = np.clip(action[0], -self.force_mag, self.force_mag)

        # -------------
        # Equations of motion (simplified)
        # -------------
        sin1, cos1 = math.sin(th1), math.cos(th1)
        sin2, cos2 = math.sin(th2), math.cos(th2)
        sin3, cos3 = math.sin(th3), math.cos(th3)

        # Effective mass for the cart (sum of cart mass + projection of pendulums)
        total_mass = self.mass_cart + 3 * self.mass_pend

        # Torques on each pendulum due to gravity
        torque1 = -self.mass_pend * self.gravity * self.length * sin1
        torque2 = -self.mass_pend * self.gravity * self.length * sin2
        torque3 = -self.mass_pend * self.gravity * self.length * sin3

        # Angular accelerations (assuming each link is pinned at the top)
        # This is an approximation that doesn't consider link-to-link coupling properly.
        alpha1 = torque1 / (self.mass_pend * (self.length**2))
        alpha2 = torque2 / (self.mass_pend * (self.length**2))
        alpha3 = torque3 / (self.mass_pend * (self.length**2))

        # Coupled effect on the cart from pendulums
        # For a more realistic approach, you'd sum up the horizontal components of
        # each pendulum’s acceleration.  We'll do a rough approach:
        horizontal_force_pend1 = self.mass_pend * self.length * (th1_dot**2 * sin1 + alpha1 * cos1)
        horizontal_force_pend2 = self.mass_pend * self.length * (th2_dot**2 * sin2 + alpha2 * cos2)
        horizontal_force_pend3 = self.mass_pend * self.length * (th3_dot**2 * sin3 + alpha3 * cos3)
        net_pendulum_force = horizontal_force_pend1 + horizontal_force_pend2 + horizontal_force_pend3

        # Net force on the cart
        net_force_on_cart = force + net_pendulum_force - self.cart_friction * x_dot

        # Cart acceleration
        x_ddot = net_force_on_cart / total_mass

        # Pendulum friction (very simplified)
        alpha1 -= self.pend_friction * th1_dot
        alpha2 -= self.pend_friction * th2_dot
        alpha3 -= self.pend_friction * th3_dot

        # ----------------------
        # Integrate forward
        # ----------------------
        x       = x       + self.tau * x_dot
        x_dot   = x_dot   + self.tau * x_ddot
        th1     = th1     + self.tau * th1_dot
        th1_dot = th1_dot + self.tau * alpha1
        th2     = th2     + self.tau * th2_dot
        th2_dot = th2_dot + self.tau * alpha2
        th3     = th3     + self.tau * th3_dot
        th3_dot = th3_dot + self.tau * alpha3

        self.state = (x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot)

        # ----------------------
        # Check termination
        # ----------------------
        terminated = bool(
            abs(x) > self.x_threshold
        )
        
        # ----------------------
        # Calculate reward
        # ----------------------
        # Reward for keeping angles near 0 (upright)
        # Also small penalty for cart displacement
        upright_reward = 3.0 - (abs(th1) + abs(th2) + abs(th3))
        cart_penalty = 0.05 * abs(x)
        reward = float(upright_reward - cart_penalty)

        if terminated:
            # If the episode ends early, penalty can be added
            reward -= 5.0

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._render_init()

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Current state
        x, _, th1, _, th2, _, th3, _ = self.state

        # Convert cart x (meters) to pixels
        cart_x_px = int(self.screen_width / 2 + x * self.pixels_per_meter)
        cart_y_px = int(self.cart_y_pos)

        # Draw cart
        cart_w, cart_h = 50, 30
        cart_rect = pygame.Rect(
            cart_x_px - cart_w//2,
            cart_y_px - cart_h//2,
            cart_w,
            cart_h
        )
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)

        # Helper function to draw each link
        def draw_link(origin_x, origin_y, angle, color):
            # Convert from upright reference
            # End of the pendulum link
            end_x = origin_x + self.length * self.pixels_per_meter * math.sin(angle)
            end_y = origin_y + self.length * self.pixels_per_meter * math.cos(angle)
            pygame.draw.line(
                surface= self.screen,
                color = color,
                start_pos = (origin_x, origin_y),
                end_pos = (end_x, end_y),
                width = 3
            )
            return end_x, end_y

        # First link pivot is at the cart center (top)
        pivot1_x, pivot1_y = cart_x_px, cart_y_px - cart_h//2
        end1_x, end1_y = draw_link(pivot1_x, pivot1_y, th1, (255, 0, 0))

        # Second link pivot is at the end of link 1
        end2_x, end2_y = draw_link(end1_x, end1_y, th2, (0, 255, 0))

        # Third link pivot is at the end of link 2
        end3_x, end3_y = draw_link(end2_x, end2_y, th3, (0, 0, 255))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _render_init(self):
        pygame.init()
        pygame.display.set_caption("Triple Pendulum Environment")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
