import numpy as np
import random as rd
from config import config

class RewardManager:
    def __init__(self):
        # -----------------------
        # Configuration
        # -----------------------
        self.num_nodes = config['num_nodes']
        self.length = 0.33  # Pendulum length
        
        # -----------------------
        # Reward weights
        # -----------------------
        self.cart_position_weight = 0.20
        self.nodes_position_weight = 0.20
        self.termination_penalty = 1
        self.alignement_weight = 2.0
        self.upright_weight = 0.21 
        self.stability_weight = 0.02  # Weight for the stability reward
        self.mse_weight = 0.3  # Weight for the MSE penalty
        
        # -----------------------
        # Upright tracking parameters
        # -----------------------
        self.max_height = 0.33 * self.num_nodes # Threshold for considering pendulum upright
        self.consecutive_upright_steps = 0  # Track consecutive steps with pendulum upright
        self.exponential_base = 1.15  # Base for exponential reward
        self.max_exponential = 3.0  # Maximum exponential multiplier
        self.have_been_upright_once = False
        self.came_back_down = False
        self.steps_double_down = 0
        self.force_terminated = False
        self.first_step_above_threshold = None

        # -----------------------
        # Real Reward Components
        # -----------------------
        self.threshold_ratio = 0.90  # 90% of pendulum length (as per the formula)
        self.time_over_threshold = 0
        self.prev_output = None
        self.output_deltas = []

        # -----------------------
        # State tracking
        # -----------------------
        self.old_points_positions = None
        self.cached_velocity = 0
        self.update_step = 0
        self.hera_update_step = 0
        self.old_heraticness_penalty = 0
        self.previous_action = None
        self.old_points_positions = None
        
        # -----------------------
        # Target state
        # -----------------------
        self.aim_position_state = [ 0.0,  np.pi/2,  np.pi/2,  np.pi/2,
                                    0.0, 0.0, 0.0, 0.0,
                                    0.0, 0.33, 0.0, 0.66,
                                    0.0, 1.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0,
                                    0.5,  1.0,  0.0,  0.0,
                                    0.0,  1.0,  1.0]
        
        # -----------------------
        # DEBUG
        # -----------------------
        self.old_angles = [0.0, 0.0, 0.0]
        self.old_alignement_penalty = None

    def calculate_reward(self, state, terminated, current_step, action):
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

        end_node_x = x3 if self.num_nodes == 3 else x2 if self.num_nodes == 2 else x1
        end_node_y = y3 if self.num_nodes == 3 else y2 if self.num_nodes == 2 else y1

        # ----------------------- REWARD COMPONENTS -----------------------
        if current_step == 0 and self.update_step == 0:
            self.old_points_positions = [x, x1, y1, x2, y2, x3, y3]
            self.update_step = current_step
        
        # ----------------------- CART AND NODES POSITION REWARD -----------------------
        x_cart_penalty = self.cart_position_weight * (abs(x)) **2
        x_nodes_penalty = self.nodes_position_weight * (abs(end_node_x)) **2

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
        is_upright = (end_node_y > self.max_height * self.threshold_ratio)

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
        points_velocity = ((abs(x3 - self.old_points_positions[5]) + abs(y3 - self.old_points_positions[6]))) ** 0.2
        if points_velocity == 0:
            points_velocity = self.cached_velocity
        else:
            self.cached_velocity = points_velocity

        stability_penalty = self.stability_weight * (angular_velocity_penalty + points_velocity)

        # ----------------------- MSE PENALTY -----------------------
        mse_sum = 0
        for idx, component in enumerate(self.aim_position_state):
            if idx in [0, 9, 11, 13]:
                importance_coef = 5.0
            else:
                importance_coef = 0.0
            if idx < len(state):
                mse_sum += (state[idx] - component) ** 2 * importance_coef
        
        mse_penalty = self.mse_weight * (mse_sum / len(state))

        # ----------------------- FAKE REWARD -----------------------
        fake_reward = upright_reward - mse_penalty - non_alignement_penalty - stability_penalty

        # ----------------------- REAL REWARD -----------------------
        threshold = self.max_height * self.threshold_ratio

        # Track time over threshold
        if end_node_y > threshold:
            self.time_over_threshold += 1
        else:
            self.time_over_threshold = 0

        # Track smoothness with exponential moving average of variation
        if self.prev_output is not None:
            delta = abs(end_node_y - self.prev_output)
            self.smoothed_variation = 0.99 * self.smoothed_variation + 0.1 * delta
        else:
            self.smoothed_variation = 0.0

        self.prev_output = end_node_y

        # ----------------------- BORDER PENALTY -----------------------
        border_penalty = 0.0
        if x < -1.6 or x > 1.6:
            border_penalty = 1

        # ----------------------- HERATICNESS PENALTY -----------------------
        heraticness_penalty = 0.0
        # Sauvegarder l'action précédente pour l'affichage
        old_action = self.previous_action
        
        if action is None or action == 0.0:
            pass
        elif self.previous_action is None:
            # Initialisation
            self.previous_action = action
        else:
            # Si l'action est différente de la précédente, calculer la pénalité
            if action != self.previous_action:
                # Convertir en valeur scalaire en utilisant float()
                heraticness_penalty = float(abs(self.previous_action - action))
            else:
                # Sinon, utiliser l'ancienne pénalité
                heraticness_penalty = self.old_heraticness_penalty
            
            # Toujours mettre à jour l'action précédente pour la prochaine fois
            self.previous_action = action
        
        # Mettre à jour l'ancienne pénalité pour la prochaine fois
        self.old_heraticness_penalty = heraticness_penalty
        
        # Compute the score
        reward = self.time_over_threshold / (1 + self.smoothed_variation) + max(end_node_y * 5, 0) 

        # Normalize reward
        reward = (1 + (reward / 25) * ((2 * np.pi) ** (-0.5) * np.exp(-(x) ** 2)) / 5) ** 2 - border_penalty - x_nodes_penalty - heraticness_penalty * 0.1
        
        # Apply termination penalty
        if terminated:
            reward -= self.termination_penalty

        if end_node_y < self.max_height * self.threshold_ratio * 0.9:
            self.force_terminated = True
            reward -= 3
        
        components_dict = {
            'reward': reward,
            'x_penalty': x_cart_penalty,
            'non_alignement_penalty': non_alignement_penalty,
            'upright_reward': upright_reward,
            'stability_penalty': stability_penalty,
            'mse_penalty': mse_penalty,
            'heraticness_penalty': heraticness_penalty,
        }

        return reward, components_dict, self.force_terminated

    def reset(self):
        """Reset the reward manager state"""
        self.consecutive_upright_steps = 0
        self.have_been_upright_once = False
        self.came_back_down = False
        self.steps_double_down = 0
        self.force_terminated = False
        self.cached_velocity = 0
        self.update_step = 0
        self.hera_update_step = 0
        self.smoothed_smoothness = 0.0
        self.prev_output = None
        self.time_over_threshold = 0
        self.output_deltas = []
        self.old_heraticness_penalty = 0
        self.previous_action = None
        self.smoothed_variation = 0.0
