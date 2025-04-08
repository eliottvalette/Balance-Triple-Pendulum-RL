import numpy as np
import random as rd
from config import config

class RewardManager:
    def __init__(self):
        # -----------------------
        # Configuration
        # -----------------------
        self.num_nodes = config['num_nodes']
        self.length = 0.5  # Pendulum length
        
        # -----------------------
        # Reward weights
        # -----------------------
        self.cart_position_weight = 0.10
        self.termination_penalty = 100.0
        self.alignement_weight = 0.15
        self.upright_weight = 3
        self.stability_weight = 0.02  # Weight for the stability reward
        self.mse_weight = 0.5  # Weight for the MSE penalty
        
        # -----------------------
        # Upright tracking parameters
        # -----------------------
        self.upright_threshold = 0.45  # Threshold for considering pendulum upright
        self.consecutive_upright_steps = 0  # Track consecutive steps with pendulum upright
        self.exponential_base = 1.15  # Base for exponential reward
        self.max_exponential = 5.0  # Maximum exponential multiplier
        self.have_been_upright_once = False
        self.came_back_down = False
        self.steps_double_down = 0
        self.force_terminated = False
        
        # -----------------------
        # State tracking
        # -----------------------
        self.old_points_positions = None
        self.cached_velocity = 0
        
        # -----------------------
        # Target state
        # -----------------------
        self.aim_position_state = [0.0, 0.0, 0.0, np.pi,
                                 0.0, 0.0, np.pi, 0.0,
                                 0.0, np.pi, 0.0, 0.0,
                                 0.5, 0.5, 0.5, 0.41,
                                 0.5, 0.33333333, 0.5, 0.25,
                                 0.0, 0.0, 0.0, 0.0]

    def calculate_reward(self, state, terminated, current_step):
        """
        Calculate the reward based on the current state and termination status.
        
        Args:
            state (tuple): A tuple containing:
                - Cart state: x, x_dot, x_ddot (indices 0-2)
                - Pendulum states: th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot (indices 3-11)
                - Pendulum node positions: pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y (indices 12-19)
            terminated (bool): Whether the episode has terminated due to constraints violation
        
        Returns:
            float: The calculated reward value
        """
        # Extract state components
        x, x_dot, x_ddot = state[0:3]  # Cart state
        th1, th1_dot, th1_ddot = state[3:6]  # First pendulum angles and derivatives
        th2, th2_dot, th2_ddot = state[6:9]  # Second pendulum angles and derivatives
        th3, th3_dot, th3_ddot = state[9:12] # Third pendulum angles and derivatives
        pivot1_x, pivot1_y = state[12:14]    # First pivot position
        end1_x, end1_y = state[14:16]        # First end position
        end2_x, end2_y = state[16:18]        # Second end position
        end3_x, end3_y = state[18:20]        # Third end position

        end_node_y = state[13 + self.num_nodes * 2]

        if self.old_points_positions is None:
            self.old_points_positions = [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y]
        
        # x close to 0
        x_penalty = self.cart_position_weight * (abs(x)) **2

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node - negate p*_y values because in the physics simulation,
        # negative y means upward (which is what we want to reward)
        # The physics uses a reference frame where positive y is downward
        upright_reward_points = self.upright_weight * (1.25 - end1_y - end2_y - end3_y)
        upright_reward_angles = self.upright_weight * (abs(th1) + abs(th2) + abs(th3)) * 0.2
        upright_reward = upright_reward_points + upright_reward_angles - 2.5

        # Check if pendulum is upright
        is_upright = (end_node_y < self.upright_threshold)

        # Update consecutive upright steps
        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0

        # Calculate exponential reward multiplier
        if self.consecutive_upright_steps > 40:
            exponential_multiplier = min(
                self.exponential_base ** (self.consecutive_upright_steps / 50),  # Divide by 100 to make it grow more slowly
                self.max_exponential
            )
        else:
            exponential_multiplier = 0.5

        # Apply exponential multiplier to upright reward
        upright_reward *= exponential_multiplier

        # Stability reward: penalize high angular velocities and accelerations of points of the pendulum
        angular_velocity_penalty = (th1_dot**2 + th2_dot**2 + th3_dot**2) / 3.0

        # Calculate velocity of points of the pendulum
        points_velocity = ((abs(end3_x - self.old_points_positions[6]) + abs(end3_y - self.old_points_positions[7]))) ** 0.2
        if points_velocity == 0:
            points_velocity = self.cached_velocity
        else:
            self.cached_velocity = points_velocity

        self.old_points_positions = [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y]

        stability_penalty = self.stability_weight * (angular_velocity_penalty + points_velocity)

        # Calculate MSE penalty with special handling for angles
        mse_sum = 0
        # Handle non-angular components (positions and velocities)
        for i in range(len(state)):
            if i in [3, 6, 9]: # angles
                mse_sum += np.sqrt((abs(state[i]) - self.aim_position_state[i]) ** 2)
            elif i in [0, 12, 13, 14, 15, 16, 17, 18, 19] : # absolute positions
                mse_sum += (state[i] - self.aim_position_state[i]) ** 2
        
        mse_penalty = self.mse_weight * (mse_sum / len(state))

        # Penalties are negative, rewards are positive
        reward = upright_reward - x_penalty - non_alignement_penalty - stability_penalty - mse_penalty
        # reward = upright_reward - (x_penalty + non_alignement_penalty + stability_penalty + mse_penalty) * 0.01

        if not self.have_been_upright_once and end_node_y < self.upright_threshold:
            self.have_been_upright_once = True

        if self.have_been_upright_once and end_node_y > self.upright_threshold:
            self.came_back_down = True
        
        if self.have_been_upright_once and self.came_back_down:
            reward = -10
            self.steps_double_down += 1

        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty

        self.force_terminated = False
        if self.steps_double_down > 150:
            self.force_terminated = True
                   
        return reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty, mse_penalty, self.force_terminated
    
    def get_reward_components(self, state, current_step):
        reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty, mse_penalty, force_terminated = self.calculate_reward(state, False, current_step)
        return {
            'x_penalty': x_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'stability_penalty': stability_penalty,
            'mse_penalty': mse_penalty,
            'reward': reward,
            'force_terminated': force_terminated
        }

    def reset(self):
        """Reset the reward manager state"""
        self.old_points_positions = None
        self.consecutive_upright_steps = 0
        self.have_been_upright_once = False
        self.came_back_down = False
        self.steps_double_down = 0
        self.force_terminated = False
        self.cached_velocity = 0