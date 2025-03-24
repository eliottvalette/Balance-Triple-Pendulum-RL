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
        self.gravity = 1.0 # Gravity (9.81)
        self.mass_cart = 1.0
        self.mass_pend1 = 0.1  # Mass of first pendulum
        self.mass_pend2 = 0.1  # Mass of second pendulum
        self.mass_pend3 = 0.1  # Mass of third pendulum
        self.length = 0.5
        self.cart_friction = 0.5
        self.pend_friction = 0.1
        self.air_resistance = 0.5  # Base air resistance coefficient
        self.air_density = 1.225  # kg/m^3 (density of air)
        self.drag_coefficient = 0.47  # Drag coefficient for a sphere
        self.reference_area = 0.01  # m^2 (reference area for drag calculation)
        self.velocity_threshold = 5.0  # m/s (threshold for increased drag)
        self.max_drag_coefficient = 2.0  # Maximum drag coefficient at high velocities
        
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
        high = np.array([
            self.x_threshold * 2.0,        # x
            self.x_dot_threshold * 2.0,    # x_dot
            np.finfo(np.float32).max,      # x_ddot
            np.finfo(np.float32).max,      # theta1
            np.finfo(np.float32).max,      # theta1_dot
            np.finfo(np.float32).max,      # theta1_ddot
            np.finfo(np.float32).max,      # theta2
            np.finfo(np.float32).max,      # theta2_dot
            np.finfo(np.float32).max,      # theta2_ddot
            np.finfo(np.float32).max,      # theta3
            np.finfo(np.float32).max,      # theta3_dot
            np.finfo(np.float32).max       # theta3_ddot
        ], dtype=np.float32)

        self.observation_space = 21

        # Internal state
        self.state_for_simu = None
        self.consecutive_upright_steps = 0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.cart_y_pos = 300   # y-position of cart in the rendered view
        self.pixels_per_meter = 100
        self.tick = 30
        self.clock = pygame.time.Clock()

        # Font for metrics display
        self.font = None

        # Reward display
        self.current_reward = 0.0
        self.reward_components = {}

    def reset(self):
        self.state_for_simu = [
            0.0,                          # cart x
            0.0,                          # cart velocity
            0.0,                          # cart acceleration
            np.pi, # theta1 (vertical)
            0.0, # theta_dot1 (vertical)
            0.0,                          # theta_ddot1 (vertical)
            np.pi, # theta2 (horizontal)
            0.0, # theta_dot2
            0.0,                          # theta_ddot2
            np.pi, # theta3
            0.0, # theta_dot3
            0.0                           # theta_ddot3
        ]

        if self.render_mode == "human" and self.screen is None:
            self._render_init()
        
        # Create a copy of the state to avoid directly sharing state_for_simu
        observation = np.array(self.state_for_simu, dtype=np.float32)
        
        return observation, {}

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
        x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state_for_simu
        
        # Total system mass calculation for proper momentum conservation
        total_mass = self.mass_cart + self.mass_pend1 + self.mass_pend2 + self.mass_pend3
        
        # Run multiple substeps for better stability
        for _ in range(self.sub_steps):
            # Unpack current state
            x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state_for_simu
            
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
            
            # Calculate dynamic drag coefficient based on velocity
            def get_drag_coefficient(velocity):
                if velocity < self.velocity_threshold:
                    return self.drag_coefficient
                else:
                    # Increase drag coefficient with velocity
                    excess_velocity = velocity - self.velocity_threshold
                    return min(self.drag_coefficient + (excess_velocity * 0.3), self.max_drag_coefficient)
            
            # Calculate air resistance using the drag equation
            def calculate_air_resistance(velocity, speed):
                if speed < 1e-6:
                    return np.zeros(2)
                drag_coef = get_drag_coefficient(speed)
                # F = 1/2 * ρ * v^2 * Cd * A
                force_magnitude = 0.5 * self.air_density * speed**2 * drag_coef * self.reference_area
                # Apply force in opposite direction of velocity
                return -force_magnitude * np.array([velocity[0]/speed, velocity[1]/speed])
            
            # Apply air resistance to each node
            air_resistance1 = calculate_air_resistance([v1_x, v1_y], v1_speed)
            air_resistance2 = calculate_air_resistance([v2_x, v2_y], v2_speed)
            air_resistance3 = calculate_air_resistance([v3_x, v3_y], v3_speed)
            
            # Convert air resistance to angular forces with increased effect
            th1_air = np.cross([0, 0, 1], [air_resistance1[0], air_resistance1[1], 0])[2] / (self.length * 0.5)
            th2_air = np.cross([0, 0, 1], [air_resistance2[0], air_resistance2[1], 0])[2] / (self.length * 0.5)
            th3_air = np.cross([0, 0, 1], [air_resistance3[0], air_resistance3[1], 0])[2] / (self.length * 0.5)
            
            # Add additional angular damping at high velocities
            def get_angular_damping(velocity):
                if velocity < self.velocity_threshold:
                    return 0.0
                else:
                    excess_velocity = velocity - self.velocity_threshold
                    return excess_velocity * 0.2  # Linear increase in damping
            
            # Apply additional angular damping
            th1_air -= get_angular_damping(v1_speed) * th1_dot
            th2_air -= get_angular_damping(v2_speed) * th2_dot
            th3_air -= get_angular_damping(v3_speed) * th3_dot
            
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
            
            # Apply maximum angular velocity limits to prevent unrealistic movement
            max_angular_velocity = 15.0  # radians per second
            th1_dot_new = np.clip(th1_dot_new, -max_angular_velocity, max_angular_velocity)
            th2_dot_new = np.clip(th2_dot_new, -max_angular_velocity, max_angular_velocity)
            th3_dot_new = np.clip(th3_dot_new, -max_angular_velocity, max_angular_velocity)
            
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
            
            # Update state with all information (only 12 core variables now)
            self.state_for_simu = np.array([
                x_new, x_dot_new, x_ddot,  # Cart state
                th1_new, th1_dot_new, th1_ddot,  # First pendulum
                th2_new, th2_dot_new, th2_ddot,  # Second pendulum
                th3_new, th3_dot_new, th3_ddot   # Third pendulum
            ], dtype=np.float32)
        
        x_new, x_dot_new = self.state_for_simu[0], self.state_for_simu[1]
        
        # Set velocity to zero at the boundary to prevent bouncing
        if (x_new == self.x_threshold and x_dot_new > 0) or (x_new == -self.x_threshold and x_dot_new < 0):
            x_dot_new = 0
        
        # Update the state with clipped position and adjusted velocity
        self.state_for_simu[0] = x_new
        self.state_for_simu[1] = x_dot_new

        # Only terminate if velocity exceeds threshold
        terminated = bool(abs(x_dot_new) > self.x_dot_threshold or abs(x_new) >= 3)

        # Check for NaN values in state
        if np.isnan(np.sum(self.state_for_simu)):
            print('state:', self.state_for_simu)
            raise ValueError("Warning: NaN detected in state")
        
        # Create a copy of the state for the observation to avoid directly sharing state_for_simu
        observation = np.array(self.state_for_simu, dtype=np.float32)
        
        # Return observation, reward, terminated, and info dictionary
        return observation, terminated

    def get_rich_state(self, state):
        """
        Get a rich state representation for the environment.
        INDEX, VALUE, DESCRIPTION
        0, x, cart position
        1, x_dot, cart velocity
        2, x_ddot, cart acceleration
        3, th1, first pendulum angle
        4, th1_dot, first pendulum velocity
        5, th1_ddot, first pendulum acceleration
        6, th2, second pendulum angle
        7, th2_dot, second pendulum velocity
        8, th2_ddot, second pendulum acceleration
        9, th3, third pendulum angle
        10, th3_dot, third pendulum velocity
        11, th3_ddot, third pendulum acceleration
        12, pivot1_x, first pivot position
        13, pivot1_y, first pivot position
        14, end1_x, first end position
        15, end1_y, first end position
        16, end2_x, second end position
        17, end2_y, second end position
        18, end3_x, third end position
        19, end3_y, third end position
        20, close_to_left, binary if cart is close to left edge of the screen
        21, close_to_right, binary if cart is close to right edge of the screen
        22, normalized_consecutive_upright_steps, normalized number of consecutive upright steps
        23, is_long_upright, binary if number of consecutive upright steps is greater than 60
        
        Returns:
            list: A list containing all relevant state variables.
        """

        # Create a copy of the state to avoid modifying state_for_simu
        processed_state = np.array(state, dtype=np.float32)

        # Extract relevant state variables
        x, x_dot = processed_state[0], processed_state[1]
        th1, th2, th3 = processed_state[3], processed_state[6], processed_state[9]

        cart_x_px = int(self.screen_width / 2 + x * self.pixels_per_meter)
        cart_y_px = int(self.cart_y_pos)

        def calculate_following_node(origin_x, origin_y, angle):
            link_length = self.length * self.pixels_per_meter
            end_x = origin_x + link_length * math.sin(angle)
            end_y = origin_y + link_length * math.cos(angle)
            return end_x, end_y

        # Get pendulum nodes positions in pixels refering to draw_link in pygame render
        pivot1_x, pivot1_y = cart_x_px, cart_y_px
        end1_x, end1_y = calculate_following_node(pivot1_x, pivot1_y, th1)
        end2_x, end2_y = calculate_following_node(end1_x, end1_y, th2)
        end3_x, end3_y = calculate_following_node(end2_x, end2_y, th3)

        # normalize points to be between 0 and 1
        pivot1_x, pivot1_y = pivot1_x / self.screen_width, pivot1_y / self.screen_height
        end1_x, end1_y = end1_x / self.screen_width, end1_y / self.screen_height
        end2_x, end2_y = end2_x / self.screen_width, end2_y / self.screen_height
        end3_x, end3_y = end3_x / self.screen_width, end3_y / self.screen_height

        # Binary warning if close to the edge of the screen
        close_to_left = (x < -2.4)
        close_to_right = (x > 2.4)

        # Add consecutive upright steps
        upright_reward_points = 0.2 * (2.25 - end1_y - end2_y - end3_y)
        upright_reward_angles = 0.2 * (abs(th1) + abs(th2) + abs(th3))
        upright_reward = upright_reward_points + upright_reward_angles
        is_upright = (upright_reward > 1.4)
        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0
        normalized_consecutive_upright_steps = min(self.consecutive_upright_steps / 100, 1)
        is_long_upright = self.consecutive_upright_steps > 60

        # Create model-ready state (only include necessary information)
        model_state = np.concatenate([
            processed_state[:12],  # Original 12 state variables
            [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y, close_to_left, close_to_right, normalized_consecutive_upright_steps, is_long_upright]  # Additional visual information
        ])

        return model_state

    def render(self, episode = None):
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._render_init()

        # Define colors
        BACKGROUND_COLOR = (240, 240, 245)
        CART_COLOR = (50, 50, 60)
        TRACK_COLOR = (180, 180, 190)
        PENDULUM1_COLOR = (220, 60, 60)
        PENDULUM2_COLOR = (60, 180, 60)
        PENDULUM3_COLOR = (60, 60, 220)
        TEXT_COLOR = (40, 40, 40)
        GRID_COLOR = (210, 210, 215)
        
        # Clear screen with a light background
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw grid
        grid_spacing = 50
        for x in range(0, self.screen_width, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.screen_width, y))
        
        # Draw track
        track_height = 5
        track_width = self.x_threshold * 2 * self.pixels_per_meter
        pygame.draw.rect(
            self.screen,
            TRACK_COLOR,
            pygame.Rect(
                self.screen_width // 2 - track_width // 2,
                self.cart_y_pos + 15,
                track_width,
                track_height
            )
        )
        
        # Mark center of track
        center_mark_height = 10
        pygame.draw.line(
            self.screen,
            (100, 100, 110),
            (self.screen_width // 2, self.cart_y_pos + 15),
            (self.screen_width // 2, self.cart_y_pos + 15 + center_mark_height),
            2
        )

        # Current state
        x, x_dot, x_ddot, th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot = self.state_for_simu

        # Convert cart x (meters) to pixels
        cart_x_px = int(self.screen_width / 2 + x * self.pixels_per_meter)
        cart_y_px = int(self.cart_y_pos)

        # Draw cart with rounded corners and 3D effect
        cart_w, cart_h = 60, 30
        cart_rect = pygame.Rect(
            cart_x_px - cart_w//2,
            cart_y_px - cart_h//2,
            cart_w,
            cart_h
        )
        
        # Draw cart shadow
        shadow_offset = 3
        shadow_rect = pygame.Rect(
            cart_rect.left + shadow_offset,
            cart_rect.top + shadow_offset,
            cart_rect.width,
            cart_rect.height
        )
        pygame.draw.rect(self.screen, (180, 180, 190), shadow_rect, border_radius=5)
        
        # Draw main cart
        pygame.draw.rect(self.screen, CART_COLOR, cart_rect, border_radius=5)
        
        # Add a highlight to the cart for 3D effect
        highlight_rect = pygame.Rect(
            cart_rect.left + 3,
            cart_rect.top + 3,
            cart_rect.width - 6,
            10
        )
        pygame.draw.rect(self.screen, (80, 80, 90), highlight_rect, border_radius=3)

        # Helper function to draw each link with thickness and joints
        def draw_link(origin_x, origin_y, angle, color):
            link_length = self.length * self.pixels_per_meter
            end_x = origin_x + link_length * math.sin(angle)
            end_y = origin_y + link_length * math.cos(angle)
            
            # Draw link with thickness
            pygame.draw.line(
                surface=self.screen,
                color=color,
                start_pos=(origin_x, origin_y),
                end_pos=(end_x, end_y),
                width=6
            )
            
            # Draw joint at start and end
            pygame.draw.circle(self.screen, (30, 30, 30), (int(origin_x), int(origin_y)), 7)
            pygame.draw.circle(self.screen, (90, 90, 100), (int(origin_x), int(origin_y)), 5)
            
            # End joint is drawn by the next link or at the end
            return end_x, end_y

        # Draw pendulum links
        pivot1_x, pivot1_y = cart_x_px, cart_y_px - cart_h//2
        end1_x, end1_y = draw_link(pivot1_x, pivot1_y, th1, PENDULUM1_COLOR)
        end2_x, end2_y = draw_link(end1_x, end1_y, th2, PENDULUM2_COLOR)
        end3_x, end3_y = draw_link(end2_x, end2_y, th3, PENDULUM3_COLOR)
        
        # Draw final joint at the end of last pendulum
        pygame.draw.circle(self.screen, (30, 30, 30), (int(end3_x), int(end3_y)), 7)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(end3_x), int(end3_y)), 5)

        # Draw info panel background (left panel for metrics)
        panel_width = 240
        panel_height = 220
        panel_x = 10
        panel_y = 10
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (230, 230, 235), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 205), panel_rect, border_radius=10, width=2)
        
        # Draw reward panel at top right if we have reward components
        if self.reward_components:
            reward_panel_width = 300
            reward_panel_height = 180
            reward_panel_x = self.screen_width - reward_panel_width - 10
            reward_panel_y = 10
            reward_panel_rect = pygame.Rect(reward_panel_x, reward_panel_y, reward_panel_width, reward_panel_height)
            pygame.draw.rect(self.screen, (230, 230, 235), reward_panel_rect, border_radius=10)
            pygame.draw.rect(self.screen, (200, 200, 205), reward_panel_rect, border_radius=10, width=2)
            
            # Title for reward panel
            reward_title_font = pygame.font.Font(None, 28)
            reward_title = reward_title_font.render("Reward Components", True, (60, 60, 70))
            self.screen.blit(reward_title, (reward_panel_x + 10, reward_panel_y + 10))
            
            # Separator for reward panel
            pygame.draw.line(
                self.screen, 
                (200, 200, 205), 
                (reward_panel_x + 10, reward_panel_y + 40), 
                (reward_panel_x + reward_panel_width - 10, reward_panel_y + 40), 
                2
            )
        
        # Initialize font if not done yet
        if self.font is None:
            self.font = pygame.font.Font(None, 24)
        
        # Title for the metrics panel
        title_font = pygame.font.Font(None, 28)
        title = title_font.render("Triple Pendulum", True, (60, 60, 70))
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Draw separator
        pygame.draw.line(
            self.screen, 
            (200, 200, 205), 
            (panel_x + 10, panel_y + 40), 
            (panel_x + panel_width - 10, panel_y + 40), 
            2
        )

        # Convert angles to degrees for display
        th1_deg = math.degrees(th1)
        th2_deg = math.degrees(th2)
        th3_deg = math.degrees(th3)


        # Create metrics with colored indicators
        metrics = [
            {"text": f"Cart Position: {x:.2f}m", "color": TEXT_COLOR},
            {"text": f"Cart Velocity: {x_dot:.2f}m/s", "color": TEXT_COLOR},
            {"text": f"Angle 1: {th1_deg:.1f}°", "color": PENDULUM1_COLOR},
            {"text": f"Angle 2: {th2_deg:.1f}°", "color": PENDULUM2_COLOR},
            {"text": f"Angle 3: {th3_deg:.1f}°", "color": PENDULUM3_COLOR},
            {"text": f"Episode: {episode if episode is not None else 'None'}", "color": TEXT_COLOR}
        ]
        
        # Add total reward to metrics panel
        if self.reward_components:
            metrics.append({"text": f"Total Reward: {self.current_reward:.2f}", "color": (80, 80, 200)})
            # Visualize reward components with bars
            reward_components = [
                {"name": "Base", "value": self.reward_components.get('reward', 0), "color": (100, 100, 200)},
                {"name": "Upright", "value": self.reward_components.get('upright_reward', 0), "color": (80, 180, 80)},
                {"name": "Position", "value": self.reward_components.get('x_penalty', 0), "color": (200, 80, 80)},
                {"name": "Alignment", "value": self.reward_components.get('non_alignement_penalty', 0), "color": (180, 130, 80)},
                {"name": "MSE", "value": self.reward_components.get('mse_penalty', 0), "color": (180, 130, 80)}
            ]
            
            # Add stability penalty if it exists
            if 'stability_penalty' in self.reward_components:
                reward_components.append({"name": "Stability", "value": self.reward_components.get('stability_penalty', 0), "color": (150, 80, 150)})
            
            # Add velocity penalty if it exists
            if 'x_dot_penalty' in self.reward_components:
                reward_components.append({"name": "Velocity", "value": self.reward_components.get('x_dot_penalty', 0), "color": (180, 80, 180)})

        # Display metrics in left panel
        for i, metric in enumerate(metrics):
            text = self.font.render(metric["text"], True, metric["color"])
            self.screen.blit(text, (panel_x + 15, panel_y + 50 + i * 25))
        
        # Draw controls help at the bottom
        controls_y = self.screen_height - 30
        controls_text = "Controls: ← → (Move)  |  SPACE (Reset)  |  B (Brake)  |  S (Stop)  |  ↑↓ (Force)"
        controls = self.font.render(controls_text, True, (100, 100, 110))
        controls_x = (self.screen_width - controls.get_width()) // 2
        self.screen.blit(controls, (controls_x, controls_y))

        # Draw reward component bars in the right panel if we have reward information
        if self.reward_components:
            bar_y = reward_panel_y + 50
            bar_width = reward_panel_width - 40
            bar_height = 16
            bar_spacing = 28
            
            max_bar_value = 3.0  # Scale for visualization
            
            for comp in reward_components:
                name_text = self.font.render(comp["name"], True, TEXT_COLOR)
                self.screen.blit(name_text, (reward_panel_x + 15, bar_y))
                
                # Calculate the center point for the bar (this will be zero point)
                center_x = reward_panel_x + 100 + (bar_width - 130) / 2
                usable_width = bar_width - 130
                
                # Background bar
                pygame.draw.rect(
                    self.screen,
                    (220, 220, 225),
                    pygame.Rect(center_x - usable_width/2, bar_y + 5, usable_width, bar_height),
                    border_radius=4
                )
                
                # Draw center line to mark zero
                pygame.draw.line(
                    self.screen, 
                    (150, 150, 155), 
                    (center_x, bar_y + 3), 
                    (center_x, bar_y + bar_height + 7), 
                    2
                )
                
                # Value bar
                value = comp["value"]
                
                if value != 0:
                    if value > 0:
                        bar_length = min(value / max_bar_value * (usable_width/2), usable_width/2)
                        bar_x = center_x
                        bar_color = comp["color"]
                    else:
                        bar_length = min(abs(value) / max_bar_value * (usable_width/2), usable_width/2)
                        bar_x = center_x - bar_length
                        bar_color = (200, 90, 90)  # Red for negative values
                    
                    # Ensure bar_length is always a valid number and at least 1 pixel
                    bar_length = max(1, int(bar_length))
                    
                    pygame.draw.rect(
                        self.screen,
                        bar_color,
                        pygame.Rect(int(bar_x), int(bar_y + 5), bar_length, bar_height),
                        border_radius=4
                    )
                
                # Value text
                value_text = self.font.render(f"{value:.2f}", True, TEXT_COLOR)
                self.screen.blit(value_text, (reward_panel_x + bar_width - 30, bar_y + 5))
                
                bar_y += bar_spacing

        pygame.display.flip()
        # Control frame rate
        self.clock.tick(self.tick)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def _render_init(self):
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Triple Pendulum Simulation")
        
        # Only create a new screen if it doesn't exist
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            
            # Set up a nicer font if available
            try:
                self.font = pygame.font.SysFont("Arial", 18)
            except:
                self.font = pygame.font.Font(None, 24)
            
            pygame.display.flip()

    def apply_brake(self):
        """
        Apply more physically realistic emergency brake to the cart
        """
        if self.state_for_simu is not None:
            # Get current velocity
            current_velocity = self.state_for_simu[1]
            
            # Apply strong braking force but preserve momentum physics
            if abs(current_velocity) > 0.01:
                braking_direction = -1 if current_velocity > 0 else 1
                # Apply a very strong braking force - but not instant zeroing of velocity
                braking_force = abs(current_velocity) * 5.0  # Strong proportional braking
                braking_force = min(braking_force, self.force_mag)
                return np.array([braking_direction * braking_force], dtype=np.float32)
        
        return np.array([0.0], dtype=np.float32)
