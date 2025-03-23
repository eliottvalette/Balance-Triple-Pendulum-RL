import numpy as np
import random as rd
class RewardManager:
    def __init__(self):
        self.cart_position_weight = 3
        self.termination_penalty = 10.0
        self.alignement_weight = 0.01
        self.upright_weight = 1.0

    def calculate_reward(self, state, terminated):
        """
        Calculate the reward based on the current state and termination status.
        
        Args:
            state (tuple): A tuple containing:
                - Cart state: x, x_dot, x_ddot
                - Pendulum states: th1, th1_dot, th1_ddot, th2, th2_dot, th2_ddot, th3, th3_dot, th3_ddot
                - Pendulum positions: p1_x, p1_y, p2_x, p2_y, p3_x, p3_y
                - Pendulum velocities: v1_x, v1_y, v2_x, v2_y, v3_x, v3_y
            terminated (bool): Whether the episode has terminated due to constraints violation
        
        Returns:
            float: The calculated reward value
        """
        # Extract state components
        x, x_dot, x_ddot = state[0:3]  # Cart state
        th1, th1_dot, th1_ddot = state[3:6]  # First pendulum
        th2, th2_dot, th2_ddot = state[6:9]  # Second pendulum
        th3, th3_dot, th3_ddot = state[9:12]  # Third pendulum
        p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = state[12:18]  # Pendulum positions
        v1_x, v1_y, v2_x, v2_y, v3_x, v3_y = state[18:24]  # Pendulum velocities
        
        # x close to 0
        x_penalty = self.cart_position_weight * (abs(x) / 3.2) ** 2

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node
        upright_reward = self.upright_weight * (np.cos(th1 - np.pi) + np.cos(th2 - np.pi) + np.cos(th3 - np.pi)) / 3.0

        # Penalties are negative, rewards are positive
        reward = upright_reward - x_penalty - non_alignement_penalty
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward, upright_reward, x_penalty, non_alignement_penalty

    def get_reward_components(self, state):
        reward, upright_reward, x_penalty, non_alignement_penalty = self.calculate_reward(state, False)
        return {
            'x_penalty': x_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'reward': reward
        } 