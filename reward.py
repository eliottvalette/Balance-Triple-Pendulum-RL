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
        self.cart_position_weight = 0.30
        self.termination_penalty = 10_000
        self.alignement_weight = 2.0
        self.upright_weight = 1.5
        self.stability_weight = 0.02  # Weight for the stability reward
        self.mse_weight = 0.0  # Weight for the MSE penalty
        
        # -----------------------
        # Upright tracking parameters
        # -----------------------
        self.upright_threshold = 0.20  # Threshold for considering pendulum upright
        self.consecutive_upright_steps = 0  # Track consecutive steps with pendulum upright
        self.exponential_base = 1.15  # Base for exponential reward
        self.max_exponential = 3.0  # Maximum exponential multiplier
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
        
        # -----------------------
        # DEBUG
        # -----------------------
        self.old_angles = [0.0, 0.0, 0.0]
        self.old_alignement_penalty = None

    def calculate_reward(self, state, terminated, current_step):
        """
        Calculate the reward based on the current state and termination status.
        
        Args:
            state: [x, q1, q2, q3, u1, u2, u3, f, x1, y1, x2, y2, x3, y3]
            terminated (bool): Whether the episode has terminated due to constraints violation
        
        Returns:
            float: The calculated reward value
        """
        # ----------------------- SET UPS -----------------------
        x = state[0]
        q1 = state[1]
        q2 = state[2]
        q3 = state[3]
        u1 = state[4]
        u2 = state[5]
        u3 = state[6]
        f = state[7]
        x1 = state[8]
        y1 = state[9]
        x2 = state[10]
        y2 = state[11]
        x3 = state[12]
        y3 = state[13]

        end_node_y = y3 if self.num_nodes == 3 else y2 if self.num_nodes == 2 else y1

        if self.old_points_positions is None:
            self.old_points_positions = [x1, y1, x2, y2, x3, y3]
        
        # ----------------------- CART POSITION REWARD -----------------------
        x_penalty = self.cart_position_weight * (abs(x)) **2

        # ----------------------- ANGLES ALIGNEMENT REWARD -----------------------
        non_alignement_penalty = self.alignement_weight * (((1 - np.cos(q1 - q2)) + (1 - np.cos(q2 - q3))) / 2)

        # ----------------------- UPRIGHTNESS REWARD -----------------------
        # Uprightness of each node - negate p*_y values because in the physics simulation,
        # negative y means upward (which is what we want to reward)
        # The physics uses a reference frame where positive y is downward
        upright_reward_points = self.upright_weight * (y1 + y2 + y3)
        upright_reward_angles = self.upright_weight * (np.sin(q1) + np.sin(q2) + np.sin(q3)) * 0.2 
        upright_reward = upright_reward_points + upright_reward_angles

        # Check if pendulum is upright
        is_upright = (end_node_y > self.upright_threshold)

        # Update consecutive upright steps
        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0

        # Calculate exponential reward multiplier
        if self.consecutive_upright_steps > 40:
            exponential_multiplier = min(
                self.exponential_base ** (self.consecutive_upright_steps / 10),  # Divide by 100 to make it grow more slowly
                self.max_exponential
            )
        else:
            exponential_multiplier = 0.5

        # Apply exponential multiplier to upright reward
        if upright_reward > 0:
            upright_reward *= exponential_multiplier

        # ----------------------- STABILITY REWARD -----------------------
        angular_velocity_penalty = (u1**2 + u2**2 + u3**2) / 3.0

        # ----------------------- POINTS VELOCITY -----------------------
        points_velocity = ((abs(x3 - self.old_points_positions[4]) + abs(y3 - self.old_points_positions[5]))) ** 0.2
        if points_velocity == 0:
            points_velocity = self.cached_velocity
        else:
            self.cached_velocity = points_velocity

        self.old_points_positions = [x1, y1, x2, y2, x3, y3]

        stability_penalty = self.stability_weight * (angular_velocity_penalty + points_velocity)

        # ----------------------- MSE PENALTY -----------------------
        mse_sum = 0
        # Handle non-angular components (positions and velocities)
        for i in range(len(state)):
            if i in [3, 6, 9]: # angles
                mse_sum += np.sqrt((abs(state[i]) - self.aim_position_state[i]) ** 2)
            elif i in [0, 12, 13, 14, 15, 16, 17, 18, 19] : # absolute positions
                mse_sum += (state[i] - self.aim_position_state[i]) ** 2
        
        mse_penalty = self.mse_weight * (mse_sum / len(state))

        # ----------------------- REWARD -----------------------
        reward = upright_reward - x_penalty - non_alignement_penalty - stability_penalty - mse_penalty
        # reward = upright_reward - (x_penalty + non_alignement_penalty + stability_penalty + mse_penalty) * 0.01

        if not self.have_been_upright_once and end_node_y > self.upright_threshold:
            self.have_been_upright_once = True

        if self.have_been_upright_once and end_node_y < self.upright_threshold:
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