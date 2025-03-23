import numpy as np

class RewardManager:
    def __init__(self):
        self.cart_position_weight = 0.05
        self.velocity_weight = 0.01
        self.termination_penalty = 5.0
        self.alignement_weight = 0.05
        self.upright_weight = 3.0

    def calculate_reward(self, state, terminated):
        """
        Calculate the reward based on the current state and termination status.
        
        Args:
            state (tuple): A tuple of (x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)
                - x: position of the cart (range is [-2.4, 2.4])
                - x_dot: velocity of the cart (range is [-10, 10])
                - th1, th2, th3: angles of the three pendulum segments in radians (range is [-π, π])
                - th1_dot, th2_dot, th3_dot: angular velocities of the pendulum segments (range is unbounded)
                - p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y: positions of the four nodes (range is 
                  x: [-2.4 ± length, 2.4 ± length], y: [0, length*3])
            terminated (bool): Whether the episode has terminated due to constraints violation
        
        Returns:
            float: The calculated reward value
        """
        x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot = state
        
        # x close to 0
        x_penalty = self.cart_position_weight * abs(x)
        
        # x_dot close to 0
        x_dot_penalty = self.velocity_weight * abs(x_dot)

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node
        # Measure how close each pendulum segment is to vertical position
        upright_reward = 2 - self.upright_weight * (
            (np.cos(th1) + 1) / 2 +  # Map from [-1,1] to [0,1]
            (np.cos(th2) + 1) / 2 + 
            (np.cos(th3) + 1) / 2
        ) / 3  # Average across all three segments

        # Penalties are negative, rewards are positive
        reward = - x_penalty - x_dot_penalty - non_alignement_penalty + upright_reward
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward, upright_reward, x_penalty, x_dot_penalty, non_alignement_penalty

    def get_reward_components(self, state):
        reward, upright_reward, x_penalty, x_dot_penalty, non_alignement_penalty = self.calculate_reward(state, False)
        return {
            'x_penalty': x_penalty,
            'x_dot_penalty': x_dot_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'reward': reward
        } 