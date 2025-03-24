import numpy as np
import random as rd
class RewardManager:
    def __init__(self):
        self.cart_position_weight = 0.20
        self.termination_penalty = 100.0
        self.alignement_weight = 0.1
        self.upright_weight = 0.5
        self.stability_weight = 0.5  # Weight for the stability reward
        self.old_state = None
        self.length = 0.5  # Pendulum length
        self.consecutive_upright_steps = 0  # Track consecutive steps with pendulum upright
        self.upright_threshold = 1.3  # Threshold for considering pendulum upright
        self.exponential_base = 1.15  # Base for exponential reward
        self.max_exponential = 5.0  # Maximum exponential multiplier
        self.old_points_positions = None
        self.cached_velocity = 0

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

        if self.old_points_positions is None :
            self.old_points_positions = [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y]
        
        # x close to 0
        x_penalty = self.cart_position_weight * (abs(x)) **2

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node - negate p*_y values because in the physics simulation,
        # negative y means upward (which is what we want to reward)
        # The physics uses a reference frame where positive y is downward
        upright_reward_points = self.upright_weight * (2.25 - end1_y - end2_y - end3_y)
        upright_reward_angles = self.upright_weight * (abs(th1) + abs(th2) + abs(th3))
        upright_reward = upright_reward_points + upright_reward_angles

        # Check if pendulum is upright
        is_upright = (upright_reward > self.upright_threshold)

        # Update consecutive upright steps
        if is_upright:
            self.consecutive_upright_steps += 1
        else:
            self.consecutive_upright_steps = 0

        # Calculate exponential reward multiplier
        if self.consecutive_upright_steps > 60:
            exponential_multiplier = min(
                self.exponential_base ** (self.consecutive_upright_steps / 50),  # Divide by 100 to make it grow more slowly
                self.max_exponential
            )
        else:
            exponential_multiplier = 0.38

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

        # Penalties are negative, rewards are positive
        reward = upright_reward - x_penalty - non_alignement_penalty - stability_penalty
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty

    def get_reward_components(self, state, current_step):
        reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty = self.calculate_reward(state, False, current_step)
        return {
            'x_penalty': x_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'stability_penalty': stability_penalty,
            'reward': reward
        } 