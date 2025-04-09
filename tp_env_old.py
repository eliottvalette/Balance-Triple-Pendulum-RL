# tp_env.py

import gym
import numpy as np
from gym import spaces
import pygame
import math
from reward import RewardManager
import random as rd
class TriplePendulumEnv(gym.Env):
    """
    Custom Gym environment for controlling a cart holding a pendulum with configurable number of nodes,
    where the agent applies a horizontal force to stabilize the pendulum upright.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, reward_manager : RewardManager, render_mode=None, num_nodes=3):
        super(TriplePendulumEnv, self).__init__()
        self.num_nodes = num_nodes

        # -----------------------
        # Environment parameters
        # -----------------------
        self.gravity = 0.1
        self.mass_cart = 1.0
        self.mass_pend1 = 0.2  # Mass of first pendulum
        self.mass_pend2 = 0.2  # Mass of second pendulum
        self.mass_pend3 = 0.2  # Mass of third pendulum
        self.length = 0.5
        self.cart_friction = 0.5
        self.pend_friction = 0.1
        self.air_resistance = 0.5  # Base air resistance coefficient
        self.air_density = 1.225  # kg/m^3 (density of air)
        self.drag_coefficient = 0.47  # Drag coefficient for a sphere
        self.reference_area = 0.01  # m^2 (reference area for drag calculation)
        self.velocity_threshold = 5.0  # m/s (threshold for increased drag)
        self.max_drag_coefficient = 2.0  # Maximum drag coefficient at high velocities
        
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

        self.observation_space = 34

        # Internal state
        self.state_for_simu = None
        self.consecutive_upright_steps = 0

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.screen_width = 800
        self.screen_height = 700
        self.cart_y_pos = self.screen_height // 2  # y-position of cart in the rendered view
        self.pixels_per_meter = 100
        self.tick = 30
        self.clock = pygame.time.Clock()

        # Font for metrics display
        self.font = None

        # Reward display
        self.current_reward = 0.0
        self.reward_components = {}
        self.reward_manager = reward_manager


    def reset(self):
        # Initialize state with zeros for all nodes
        self.state_for_simu = [0.0] * 12  # 3 for cart + 9 for pendulums
        
        # Set initial angles for active nodes
        rd_angle = rd.uniform(-1, 1)
        for i in range(self.num_nodes):
            self.state_for_simu[3 + i*3] = rd_angle  # Initial angle
            self.state_for_simu[4 + i*3] = 0.0  # Initial angular velocity
            self.state_for_simu[5 + i*3] = 0.0  # Initial angular acceleration

        if self.render_mode == "human" and self.screen is None:
            self._render_init()
        
        # Create a copy of the state to avoid directly sharing state_for_simu
        observation = np.array(self.state_for_simu, dtype=np.float32)
        
        return observation, {}

    def apply_constraints(self, x, *angles):
        """Applique les contraintes de distance entre les pendules"""
        l = self.length
        
        # Position du chariot
        p0 = np.array([x, 0])
        
        # Positions initiales des pendules
        positions = [p0]
        for i in range(len(angles)):
            th = angles[i]
            prev_pos = positions[-1]
            new_pos = prev_pos + l * np.array([np.sin(th), np.cos(th)])
            positions.append(new_pos)
        
        # Applique les contraintes plusieurs fois pour plus de stabilité
        for _ in range(self.constraint_iterations):
            for i in range(1, len(positions)):
                # Contrainte entre les positions i-1 et i
                diff = positions[i] - positions[i-1]
                dist = np.linalg.norm(diff)
                if abs(dist - l) > 1e-6:
                    correction = (dist - l) / dist
                    positions[i] = positions[i-1] + diff * (1 - correction)
        
        # Calcule les nouveaux angles
        new_angles = []
        for i in range(1, len(positions)):
            diff = positions[i] - positions[i-1]
            new_angles.append(np.arctan2(diff[0], diff[1]))
        
        return new_angles

    def step(self, action):
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        
        # Store initial state
        x, x_dot, x_ddot = self.state_for_simu[0:3]
        pendulum_states = []
        for i in range(self.num_nodes):
            pendulum_states.append(self.state_for_simu[3 + i*3:6 + i*3])
        
        # Detect direction change
        direction_changed = (force > 0 and x_dot < 0) or (force < 0 and x_dot > 0)
        
        # Total system mass calculation for proper momentum conservation
        total_mass = self.mass_cart
        for i in range(self.num_nodes):
            total_mass += getattr(self, f'mass_pend{i+1}')
        
        # Run multiple substeps for better stability
        for substep in range(self.sub_steps):
            # Unpack current state
            x, x_dot, x_ddot = self.state_for_simu[0:3]
            pendulum_states = []
            for i in range(self.num_nodes):
                pendulum_states.append(self.state_for_simu[3 + i*3:6 + i*3])
            
            # If direction changed and this is the first substep, force velocity to zero
            if direction_changed and substep == 0:
                x_dot = x_dot * 0.6
                x_ddot = x_ddot * 0.6
            
            # Calculate pendulum positions and velocities
            positions = []
            velocities = []
            p_x, p_y = x, 0
            v_x, v_y = x_dot, 0
            
            for i in range(self.num_nodes):
                th, th_dot, th_ddot = pendulum_states[i]
                p_x = p_x + self.length * np.sin(th)
                p_y = p_y - self.length * np.cos(th)
                positions.append((p_x, p_y))
                
                v_x = v_x + self.length * th_dot * np.cos(th)
                v_y = v_y + self.length * th_dot * np.sin(th)
                velocities.append((v_x, v_y))
            
            # Calculate forces from pendulum mass on cart
            net_force = force - self.cart_friction * x_dot
            
            for i in range(self.num_nodes):
                th, th_dot, th_ddot = pendulum_states[i]
                mass = getattr(self, f'mass_pend{i+1}')
                
                # Centripetal force
                f_x = mass * self.length * (th_dot**2) * np.sin(th)
                # Tangential force
                f_tang = mass * self.length * th_ddot * np.cos(th)
                
                net_force += f_x + f_tang
            
            # Cart acceleration (F = ma)
            x_ddot = net_force / total_mass
            
            # Calculate pendulum angular accelerations
            new_pendulum_states = []
            for i in range(self.num_nodes):
                th, th_dot, th_ddot = pendulum_states[i]
                v_x, v_y = velocities[i]
                
                # Calculate air resistance
                speed = np.sqrt(v_x**2 + v_y**2)
                if speed < 1e-6:
                    air_resistance = np.zeros(2)
                else:
                    drag_coef = self.drag_coefficient if speed < self.velocity_threshold else min(
                        self.drag_coefficient + (speed - self.velocity_threshold) * 0.3,
                        self.max_drag_coefficient
                    )
                    force_magnitude = 0.5 * self.air_density * speed**2 * drag_coef * self.reference_area
                    air_resistance = -force_magnitude * np.array([v_x/speed, v_y/speed])
                
                # Convert air resistance to angular force
                th_air = np.cross([0, 0, 1], [air_resistance[0], air_resistance[1], 0])[2] / (self.length * 0.5)
                
                # Add angular damping
                if speed > self.velocity_threshold:
                    th_air -= (speed - self.velocity_threshold) * 0.2 * th_dot
                
                # Calculate angular acceleration
                th_ddot = (-self.gravity * np.sin(th) - x_ddot * np.cos(th) + th_air) / self.length
                
                # Add damping with momentum preservation
                if i == 0:
                    th_ddot -= self.pend_friction * th_dot
                else:
                    th_ddot -= self.pend_friction * (th_dot - pendulum_states[i-1][1])
                
                new_pendulum_states.append([th, th_dot, th_ddot])
            
            # Semi-implicit Euler integration
            dt = self.tau
            
            # Update velocities first
            x_dot_new = x_dot + x_ddot * dt
            for i in range(self.num_nodes):
                th, th_dot, th_ddot = new_pendulum_states[i]
                th_dot_new = th_dot + th_ddot * dt
                th_dot_new = np.clip(th_dot_new, -15.0, 15.0)  # Limit angular velocity
                new_pendulum_states[i][1] = th_dot_new
            
            # Update positions
            x_new = x + x_dot_new * dt
            for i in range(self.num_nodes):
                th, th_dot, th_ddot = new_pendulum_states[i]
                new_pendulum_states[i][0] = th + th_dot * dt
            
            # Apply constraints to maintain rigid connections
            if self.num_nodes > 1:
                th_list = [new_pendulum_states[i][0] for i in range(self.num_nodes)]
                th_list = self.apply_constraints(x_new, *th_list)
                for i in range(self.num_nodes):
                    new_pendulum_states[i][0] = th_list[i]
            
            # Update state
            self.state_for_simu[0] = x_new
            self.state_for_simu[1] = x_dot_new
            self.state_for_simu[2] = x_ddot
            
            for i in range(self.num_nodes):
                self.state_for_simu[3 + i*3:6 + i*3] = new_pendulum_states[i]
        
        # Set velocity to zero at the boundary to prevent bouncing
        if (x_new == self.x_threshold and x_dot_new > 0) or (x_new == -self.x_threshold and x_dot_new < 0):
            x_dot_new = 0
        
        # Update the state with clipped position and adjusted velocity
        self.state_for_simu[0] = x_new
        self.state_for_simu[1] = x_dot_new

        # Only terminate if velocity exceeds threshold
        terminated = bool(abs(x_dot_new) > self.x_dot_threshold or abs(x_new) >= 3)
        
        # Create a copy of the state for the observation to avoid directly sharing state_for_simu
        observation = np.array(self.state_for_simu, dtype=np.float32)
        
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
        24, reward, full combined reward
        25, upright_reward, upright_reward
        26, x_penalty, x penalty
        27, non_alignement_penalty, non alignement penalty
        28, stability_penalty, stability penalty
        29, mse_penalty, mse penalty
        30, have_been_upright_once, binary if pendulum has been upright at least once
        31, came_back_down, binary if pendulum has come back down after being upright
        32, steps_double_down, number of steps since coming back down
        33, force_terminated, binary if episode should be terminated
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

        # ADD reward components if Na, put 0
        if self.reward_components:
            reward = self.reward_components['reward']
            upright_reward = self.reward_components['upright_reward']
            x_penalty = self.reward_components['x_penalty']
            non_alignement_penalty = self.reward_components['non_alignement_penalty']
            stability_penalty = self.reward_components['stability_penalty']
            mse_penalty = self.reward_components['mse_penalty']
        else:
            reward = 0
            upright_reward = 0
            x_penalty = 0
            non_alignement_penalty = 0
            stability_penalty = 0
            mse_penalty = 0

        # Get reward manager state
        have_been_upright_once = float(self.reward_manager.have_been_upright_once)
        came_back_down = float(self.reward_manager.came_back_down)
        steps_double_down = self.reward_manager.steps_double_down / 150.0  # Normalize to [0,1]
        force_terminated = float(self.reward_manager.force_terminated)

        # Create model-ready state (only include necessary information)
        model_state = np.concatenate([
            processed_state[:12],  # Original 12 state variables
            [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y, close_to_left, close_to_right, normalized_consecutive_upright_steps, is_long_upright],  # Additional visual information 
            [reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty, mse_penalty],
            [have_been_upright_once, came_back_down, steps_double_down, force_terminated]  # Reward manager state
        ])

        return model_state

    def render(self, episode = None, epsilon = 0):
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._render_init()

        # Define colors
        BACKGROUND_COLOR = (240, 240, 245)
        CART_COLOR = (50, 50, 60)
        TRACK_COLOR = (180, 180, 190)
        PENDULUM_COLORS = [
            (220, 60, 60),   # Red for first pendulum
            (60, 180, 60),   # Green for second pendulum
            (60, 60, 220)    # Blue for third pendulum
        ]
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
        x, x_dot, x_ddot = self.state_for_simu[0:3]
        pendulum_states = []
        for i in range(self.num_nodes):
            pendulum_states.append(self.state_for_simu[3 + i*3:6 + i*3])

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
        pivot_x, pivot_y = cart_x_px, cart_y_px - cart_h//2
        for i in range(self.num_nodes):
            th = pendulum_states[i][0]
            pivot_x, pivot_y = draw_link(pivot_x, pivot_y, th, PENDULUM_COLORS[i])
        
        # Draw final joint at the end of last pendulum
        pygame.draw.circle(self.screen, (30, 30, 30), (int(pivot_x), int(pivot_y)), 7)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(pivot_x), int(pivot_y)), 5)

        # Draw info panel background (left panel for metrics)
        panel_width = 240
        panel_height = 220
        panel_x = 10
        panel_y = 10
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.screen, (230, 230, 235), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (200, 200, 205), panel_rect, border_radius=10, width=2)
        
        # Draw reward panel at top right if we have reward components
        rich_state = self.get_rich_state(self.state_for_simu)
        self.reward_components = self.reward_manager.get_reward_components(rich_state, 0)
        
        if self.reward_components:
            reward_panel_width = 300
            reward_panel_height = 240
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
        title = title_font.render(f"{self.num_nodes}-Node Pendulum", True, (60, 60, 70))
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
        metrics = [
            {"text": f"Cart Position: {x:.2f}m", "color": TEXT_COLOR},
            {"text": f"Cart Velocity: {x_dot:.2f}m/s", "color": TEXT_COLOR}
        ]
        
        for i in range(self.num_nodes):
            th = math.degrees(pendulum_states[i][0])
            metrics.append({"text": f"Angle {i+1}: {th:.1f}°", "color": PENDULUM_COLORS[i]})
        
        metrics.extend([
            {"text": f"Episode: {episode if episode is not None else 'None'}", "color": TEXT_COLOR},
            {"text": f"Epsilon: {epsilon*100:.2f}%", "color": TEXT_COLOR}
        ])
        
        # Add total reward to metrics panel
        if self.reward_components:
            metrics.append({"text": f"Total Reward: {self.current_reward:.2f}", "color": (80, 80, 200)})
        
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
