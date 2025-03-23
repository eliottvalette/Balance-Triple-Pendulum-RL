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

    def __init__(self, render_mode=None, test_mode = False):
        super(TriplePendulumEnv, self).__init__()

        # -----------------------
        # Environment parameters
        # -----------------------
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pend1 = 0.1  # Mass of first pendulum
        self.mass_pend2 = 0.1  # Mass of second pendulum
        self.mass_pend3 = 0.1  # Mass of third pendulum
        self.length = 0.5
        self.cart_friction = 0.5
        self.pend_friction = 0.1
        self.air_resistance = 0.1
        
        # Reduce force magnitude
        self.force_mag = 20.0
        
        # Increase simulation substeps for better stability
        self.sub_steps = 8
        self.tau = 0.02 / self.sub_steps
        self.constraint_iterations = 3  # Iterations for constraints
        
        # Inertia for each rod (simplified as thin rods)
        self.inertia_pend1 = (1/12) * self.mass_pend1 * self.length**2
        self.inertia_pend2 = (1/12) * self.mass_pend2 * self.length**2
        self.inertia_pend3 = (1/12) * self.mass_pend3 * self.length**2
        
        # Limits for cart position and velocity
        self.x_threshold = 3.2
        self.x_dot_threshold = 10.0

        # Angular limits (radians)
        self.theta_threshold_radians = math.pi

        # --------------
        # Action / State
        # --------------
        # Action: continuous cliped force applied to the cart
        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)

        # Observation:
        #   [x, x_dot, x_ddot,  # Cart state
        #    th1, th1_dot, th1_ddot,  # First pendulum
        #    th2, th2_dot, th2_ddot,  # Second pendulum
        #    th3, th3_dot, th3_ddot,  # Third pendulum
        #    p1_x, p1_y, p2_x, p2_y, p3_x, p3_y,  # Pendulum positions
        #    v1_x, v1_y, v2_x, v2_y, v3_x, v3_y]  # Pendulum velocities
        high = np.array([
            self.x_threshold * 2.0,        # x
            self.x_dot_threshold * 2.0,    # x_dot
            self.x_dot_threshold * 2.0,    # x_ddot
            np.finfo(np.float32).max,      # theta1
            np.finfo(np.float32).max,      # theta1_dot
            np.finfo(np.float32).max,      # theta1_ddot
            np.finfo(np.float32).max,      # theta2
            np.finfo(np.float32).max,      # theta2_dot
            np.finfo(np.float32).max,      # theta2_ddot
            np.finfo(np.float32).max,      # theta3
            np.finfo(np.float32).max,      # theta3_dot
            np.finfo(np.float32).max,      # theta3_ddot
            self.x_threshold * 2.0,        # p1_x
            self.x_threshold * 2.0,        # p1_y
            self.x_threshold * 2.0,        # p2_x
            self.x_threshold * 2.0,        # p2_y
            self.x_threshold * 2.0,        # p3_x
            self.x_threshold * 2.0,        # p3_y
            self.x_dot_threshold * 2.0,    # v1_x
            self.x_dot_threshold * 2.0,    # v1_y
            self.x_dot_threshold * 2.0,    # v2_x
            self.x_dot_threshold * 2.0,    # v2_y
            self.x_dot_threshold * 2.0,    # v3_x
            self.x_dot_threshold * 2.0,    # v3_y
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
        
        # Font for metrics display
        self.font = None
        self.test_mode = test_mode
        
        # Reward display
        self.current_reward = 0.0
        self.reward_components = {}

    def reset(self):
        self.state = [
            0.0,                        # cart x
            0.0,                        # cart velocity
            0.0,                        # cart acceleration
            np.random.uniform(-1, 1),   # theta1
            np.random.uniform(-0.2, 0.2), # theta_dot1
            0.0,                        # theta_ddot1
            np.random.uniform(-1, 1),   # theta2
            np.random.uniform(-0.2, 0.2), # theta_dot2
            0.0,                        # theta_ddot2
            np.random.uniform(-1, 1),   # theta3
            np.random.uniform(-0.2, 0.2), # theta_dot3
            0.0,                        # theta_ddot3
            0.0, 0.0,                   # p1_x, p1_y
            0.0, 0.0,                   # p2_x, p2_y
            0.0, 0.0,                   # p3_x, p3_y
            0.0, 0.0,                   # v1_x, v1_y
            0.0, 0.0,                   # v2_x, v2_y
            0.0, 0.0                    # v3_x, v3_y
        ]

        if self.render_mode == "human":
            self._render_init()
        
        return self.state, {}

    def apply_constraints(self, x, th1, th2, th3):
        """Applique les contraintes de distance entre les pendules"""
        l = self.length
        
        # Position du chariot
        p0 = np.array([x, 0])
        
        # Positions initiales des pendules
        p1 = p0 + l * np.array([np.sin(th1), np.cos(th1)])
        p2 = p1 + l * np.array([np.sin(th2), np.cos(th2)])
        p3 = p2 + l * np.array([np.sin(th3), np.cos(th3)])
        
        # Applique les contraintes plusieurs fois pour plus de stabilité
        for _ in range(self.constraint_iterations):
            # Contrainte entre p0 et p1
            diff = p1 - p0
            dist = np.linalg.norm(diff)
            if abs(dist - l) > 1e-6:
                correction = (dist - l) / dist
                p1 = p0 + diff * (1 - correction)
            
            # Contrainte entre p1 et p2
            diff = p2 - p1
            dist = np.linalg.norm(diff)
            if abs(dist - l) > 1e-6:
                correction = (dist - l) / dist
                p2 = p1 + diff * (1 - correction)
            
            # Contrainte entre p2 et p3
            diff = p3 - p2
            dist = np.linalg.norm(diff)
            if abs(dist - l) > 1e-6:
                correction = (dist - l) / dist
                p3 = p2 + diff * (1 - correction)
        
        # Calcule les nouveaux angles
        th1_new = np.arctan2(p1[0] - p0[0], p1[1] - p0[1])
        th2_new = np.arctan2(p2[0] - p1[0], p2[1] - p1[1])
        th3_new = np.arctan2(p3[0] - p2[0], p3[1] - p2[1])
        
        return th1_new, th2_new, th3_new

    def step(self, action):
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        # Store initial state
        x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state[:12]
        
        # Total system mass calculation for proper momentum conservation
        total_mass = self.mass_cart + self.mass_pend1 + self.mass_pend2 + self.mass_pend3
        
        # Run multiple substeps for better stability
        for _ in range(self.sub_steps):
            # Unpack current state
            x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state[:12]
            
            # Calculate pendulum positions
            p1_x = x + self.length * np.sin(th1)
            p1_y = -self.length * np.cos(th1)
            p2_x = p1_x + self.length * np.sin(th2)
            p2_y = p1_y + self.length * np.cos(th2)
            p3_x = p2_x + self.length * np.sin(th3)
            p3_y = p2_y + self.length * np.cos(th3)
            
            # Calculate pendulum velocities
            v1_x = x_dot + self.length * th1_dot * np.cos(th1)
            v1_y = self.length * th1_dot * np.sin(th1)
            v2_x = v1_x + self.length * th2_dot * np.cos(th2)
            v2_y = v1_y + self.length * th2_dot * np.sin(th2)
            v3_x = v2_x + self.length * th3_dot * np.cos(th3)
            v3_y = v2_y + self.length * th3_dot * np.sin(th3)
            
            # Calculate forces from pendulum mass on cart
            # Compute centripetal forces from pendulum rotations
            f1_x = self.mass_pend1 * self.length * (th1_dot**2) * np.sin(th1)
            f2_x = self.mass_pend2 * self.length * (th2_dot**2) * np.sin(th2)
            f3_x = self.mass_pend3 * self.length * (th3_dot**2) * np.sin(th3)
            
            # Compute tangential forces from pendulum angular accelerations
            f1_tang = self.mass_pend1 * self.length * th1_ddot * np.cos(th1)
            f2_tang = self.mass_pend2 * self.length * th2_ddot * np.cos(th2)
            f3_tang = self.mass_pend3 * self.length * th3_ddot * np.cos(th3)
            
            # Cart friction
            friction = -self.cart_friction * x_dot
            
            # Total force on cart
            net_force = force + friction + f1_x + f2_x + f3_x + f1_tang + f2_tang + f3_tang
            
            # Cart acceleration (F = ma)
            # Include the effective mass of pendulums for momentum conservation
            x_ddot = net_force / total_mass
            
            # Calculate pendulum joint forces (tension in the links)
            # This is simplified - a full constraint-based system would be more accurate
            tension1 = self.mass_pend1 * (self.gravity * np.sin(th1) + x_ddot * np.cos(th1))
            tension2 = self.mass_pend2 * (self.gravity * np.sin(th2) + x_ddot * np.cos(th2))
            tension3 = self.mass_pend3 * (self.gravity * np.sin(th3) + x_ddot * np.cos(th3))
            
            # Apply air resistance to each pendulum node
            v1_speed = np.sqrt(v1_x**2 + v1_y**2)
            v2_speed = np.sqrt(v2_x**2 + v2_y**2)
            v3_speed = np.sqrt(v3_x**2 + v3_y**2)
            
            air_resistance1 = -self.air_resistance * v1_speed * np.array([v1_x, v1_y])
            air_resistance2 = -self.air_resistance * v2_speed * np.array([v2_x, v2_y])
            air_resistance3 = -self.air_resistance * v3_speed * np.array([v3_x, v3_y])
            
            # Convert air resistance to angular forces
            th1_air = np.cross([0, 0, 1], [air_resistance1[0], air_resistance1[1], 0])[2] / self.length
            th2_air = np.cross([0, 0, 1], [air_resistance2[0], air_resistance2[1], 0])[2] / self.length
            th3_air = np.cross([0, 0, 1], [air_resistance3[0], air_resistance3[1], 0])[2] / self.length
            
            # Pendulum angular accelerations with proper physics
            g = self.gravity
            
            # First pendulum angular acceleration
            th1_ddot = (-g * np.sin(th1) - x_ddot * np.cos(th1) + th1_air) / self.length
            
            # Second pendulum angular acceleration influenced by first pendulum
            th2_ddot = (-g * np.sin(th2) - x_ddot * np.cos(th2) + th2_air) / self.length
            
            # Third pendulum angular acceleration influenced by second pendulum
            th3_ddot = (-g * np.sin(th3) - x_ddot * np.cos(th3) + th3_air) / self.length
            
            # Add damping with momentum preservation
            th1_ddot -= self.pend_friction * th1_dot
            th2_ddot -= self.pend_friction * (th2_dot - th1_dot)  # Relative to first node
            th3_ddot -= self.pend_friction * (th3_dot - th2_dot)  # Relative to second node
            
            # Semi-implicit Euler integration
            dt = self.tau
            
            # Update velocities first
            x_dot_new = x_dot + x_ddot * dt
            th1_dot_new = th1_dot + th1_ddot * dt
            th2_dot_new = th2_dot + th2_ddot * dt
            th3_dot_new = th3_dot + th3_ddot * dt
            
            # Update positions
            x_new = x + x_dot_new * dt
            th1_temp = th1 + th1_dot_new * dt
            th2_temp = th2 + th2_dot_new * dt
            th3_temp = th3 + th3_dot_new * dt
            
            # Apply constraints to maintain rigid connections
            th1_new, th2_new, th3_new = self.apply_constraints(x_new, th1_temp, th2_temp, th3_temp)
            
            # Recalculate pendulum positions and velocities with updated positions
            p1_x = x_new + self.length * np.sin(th1_new)
            p1_y = -self.length * np.cos(th1_new)
            p2_x = p1_x + self.length * np.sin(th2_new)
            p2_y = p1_y + self.length * np.cos(th2_new)
            p3_x = p2_x + self.length * np.sin(th3_new)
            p3_y = p2_y + self.length * np.cos(th3_new)
            
            # Calculate pendulum velocities
            v1_x = x_dot_new + self.length * th1_dot_new * np.cos(th1_new)
            v1_y = self.length * th1_dot_new * np.sin(th1_new)
            v2_x = v1_x + self.length * th2_dot_new * np.cos(th2_new)
            v2_y = v1_y + self.length * th2_dot_new * np.sin(th2_new)
            v3_x = v2_x + self.length * th3_dot_new * np.cos(th3_new)
            v3_y = v2_y + self.length * th3_dot_new * np.sin(th3_new)
            
            # Update state with all information
            self.state = np.array([
                x_new, x_dot_new, x_ddot,  # Cart state
                th1_new, th1_dot_new, th1_ddot,  # First pendulum
                th2_new, th2_dot_new, th2_ddot,  # Second pendulum
                th3_new, th3_dot_new, th3_ddot,  # Third pendulum
                p1_x, p1_y, p2_x, p2_y, p3_x, p3_y,  # Pendulum positions
                v1_x, v1_y, v2_x, v2_y, v3_x, v3_y  # Pendulum velocities
            ], dtype=np.float32)
        
        # Check termination
        x_new, x_dot_new = self.state[0], self.state[1]
        terminated = bool(
            abs(x_new) > self.x_threshold or
            abs(x_dot_new) > self.x_dot_threshold
        )
        
        return np.array(self.state, dtype=np.float32), terminated

    def calculate_reward(self):
        """Calculate the reward based on the current state"""
        # Extract relevant state variables
        x, x_dot = self.state[0], self.state[1]
        th1, th2, th3 = self.state[3], self.state[6], self.state[9]
        
        # Basic reward for staying alive
        reward = 1.0
        
        # Penalty for cart distance from center
        x_penalty = -0.1 * abs(x)
        
        # Penalty for cart velocity
        x_dot_penalty = -0.1 * abs(x_dot)
        
        # Reward for keeping pendulums upright (pi is upright)
        # Use cosine to make it smooth - closer to upright = higher reward
        upright_reward = 3.0 * (np.cos(th1 - np.pi) + np.cos(th2 - np.pi) + np.cos(th3 - np.pi)) / 3.0
        
        # Penalty for non-alignment of the pendulums (they should all point in same direction)
        non_alignement_penalty = -0.5 * (abs(th1 - th2) + abs(th2 - th3)) 
        
        # Total reward
        total_reward = reward + x_penalty + x_dot_penalty + upright_reward + non_alignement_penalty
        
        # Store reward components for visualization
        self.current_reward = total_reward
        self.reward_components = {
            "reward": reward,
            "x_penalty": x_penalty,
            "x_dot_penalty": x_dot_penalty, 
            "upright_reward": upright_reward,
            "non_alignement_penalty": non_alignement_penalty
        }
        
        return total_reward

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._render_init()

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Current state
        x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state[:12]

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
            end_x = origin_x + self.length * self.pixels_per_meter * math.sin(angle)
            end_y = origin_y + self.length * self.pixels_per_meter * math.cos(angle)
            pygame.draw.line(
                surface=self.screen,
                color=color,
                start_pos=(origin_x, origin_y),
                end_pos=(end_x, end_y),
                width=3
            )
            return end_x, end_y

        # Draw pendulum links
        pivot1_x, pivot1_y = cart_x_px, cart_y_px - cart_h//2
        end1_x, end1_y = draw_link(pivot1_x, pivot1_y, th1, (255, 0, 0))
        end2_x, end2_y = draw_link(end1_x, end1_y, th2, (0, 255, 0))
        end3_x, end3_y = draw_link(end2_x, end2_y, th3, (0, 0, 255))

        # Display metrics
        if self.font is None:
            self.font = pygame.font.Font(None, 24)

        # Convert angles to degrees for display
        th1_deg = math.degrees(th1)
        th2_deg = math.degrees(th2)
        th3_deg = math.degrees(th3)

        # Create metric texts
        metrics = [
            f"Cart Position: {x:.2f}m",
            f"Cart Velocity: {x_dot:.2f}m/s",
            f"Cart Acceleration: {x_ddot:.2f}m/s²",
            f"Angle 1: {th1_deg:.1f}°",
            f"Angle 2: {th2_deg:.1f}°",
            f"Angle 3: {th3_deg:.1f}°",
            f"Total Reward: {self.current_reward:.2f}"
        ]

        # Add reward components if available
        if self.reward_components:
            metrics.extend([
                f"Reward: {self.reward_components['reward']:.2f}",
                f"Alignement Penalty: {self.reward_components['non_alignement_penalty']:.2f}",
                f"Upright Reward: {self.reward_components['upright_reward']:.2f}",
                f"Cart Penalty: {self.reward_components['x_penalty']:.2f}",
                f"Velocity Penalty: {self.reward_components['x_dot_penalty']:.2f}"
            ])

        # Display metrics
        for i, metric in enumerate(metrics):
            text = self.font.render(metric, True, (0, 0, 0))
            self.screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _render_init(self):
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Triple Pendulum Environment")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        pygame.display.flip()

    def apply_brake(self):
        """
        Apply more physically realistic emergency brake to the cart
        """
        if self.state is not None:
            # Get current velocity
            current_velocity = self.state[1]
            
            # Apply strong braking force but preserve momentum physics
            if abs(current_velocity) > 0.01:
                braking_direction = -1 if current_velocity > 0 else 1
                # Apply a very strong braking force - but not instant zeroing of velocity
                braking_force = abs(current_velocity) * 5.0  # Strong proportional braking
                braking_force = min(braking_force, self.force_mag)
                return np.array([braking_direction * braking_force], dtype=np.float32)
        
        return np.array([0.0], dtype=np.float32)
