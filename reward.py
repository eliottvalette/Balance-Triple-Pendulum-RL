import numpy as np

class RewardManager:
    def __init__(self):
        self.upright_weight = 3.0
        self.cart_position_weight = 0.05
        self.velocity_weight = 0.01
        self.termination_penalty = 5.0

    def calculate_reward(self, state, terminated):
        x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot = state
        
        # Main reward components
        upright_reward = self.upright_weight - (abs(th1) + abs(th2) + abs(th3))
        cart_penalty = self.cart_position_weight * abs(x)
        velocity_penalty = self.velocity_weight * abs(x_dot)
        
        # Calculate total reward
        reward = float(upright_reward - cart_penalty - velocity_penalty)
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty
            
        return reward

    def get_reward_components(self, state):
        """Returns individual reward components for logging"""
        x, x_dot, th1, th1_dot, th2, th2_dot, th3, th3_dot = state
        return {
            'upright_reward': self.upright_weight - (abs(th1) + abs(th2) + abs(th3)),
            'cart_penalty': self.cart_position_weight * abs(x),
            'velocity_penalty': self.velocity_weight * abs(x_dot)
        } 