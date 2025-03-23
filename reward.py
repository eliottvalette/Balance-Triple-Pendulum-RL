import numpy as np

class RewardManager:
    def __init__(self):
        self.cart_position_weight = 0.5
        self.velocity_weight = 0.05
        self.termination_penalty = 10.0
        self.alignement_weight = 0.2
        self.upright_weight = 1.0
        self.acceleration_weight = 0.05
        self.energy_weight = 0.1

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
        x_penalty = self.cart_position_weight * abs(x)
        
        # x_dot close to 0
        x_dot_penalty = self.velocity_weight * abs(x_dot)
        
        # Penalize high acceleration
        acceleration_penalty = self.acceleration_weight * abs(x_ddot)

        # angles alignement 
        non_alignement_penalty = self.alignement_weight * (abs(th1-th2) + abs(th2-th3) + abs(th3-th1)) / np.pi

        # Uprightness of each node
        # Measure how close each pendulum segment is to vertical position (upward)
        # When pendulum is upright, cos(θ) is -1, when pointing down, cos(θ) is 1
        # We convert to a [0,1] scale where 1 is fully upright
        upright_reward = self.upright_weight * (
            (1 - np.cos(th1)) / 2 +  # Map from [-1,1] to [0,1]
            (1 - np.cos(th2)) / 2 + 
            (1 - np.cos(th3)) / 2
        ) / 3  # Average across all three segments
        
        # Energy penalty to encourage smooth motion
        # Calculate total kinetic and potential energy
        kinetic_energy = 0.5 * (v1_x**2 + v1_y**2 + v2_x**2 + v2_y**2 + v3_x**2 + v3_y**2)
        potential_energy = abs(p1_y) + abs(p2_y) + abs(p3_y)  # Height from ground
        energy_penalty = self.energy_weight * (kinetic_energy + potential_energy)

        # Penalties are negative, rewards are positive
        reward = upright_reward - x_penalty - x_dot_penalty - acceleration_penalty - non_alignement_penalty - energy_penalty
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward, upright_reward, x_penalty, x_dot_penalty, non_alignement_penalty, acceleration_penalty, energy_penalty

    def get_reward_components(self, state):
        reward, upright_reward, x_penalty, x_dot_penalty, non_alignement_penalty, acceleration_penalty, energy_penalty = self.calculate_reward(state, False)
        return {
            'x_penalty': x_penalty,
            'x_dot_penalty': x_dot_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'acceleration_penalty': acceleration_penalty,
            'energy_penalty': energy_penalty,
            'reward': reward
        } 