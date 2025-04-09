# tp_env.py

import gym
import numpy as np
import pygame
from gym import spaces


class TriplePendulumEnv(gym.Env):
    """
    Environnement Gym simulant un pendule triple inversé.
    Le système consiste en trois pendules en série attachés à un point fixe.
    L'agent contrôle un couple appliqué sur le premier pendule pour maintenir le système en position verticale (inversée).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, max_steps=500):
        super(TriplePendulumEnv, self).__init__()
        # Paramètres physiques
        self.L = 1.0            # Longueur de chaque pendule (m)
        self.g = 9.81           # Accélération due à la pesanteur (m/s²)
        self.k = 2.0            # Constante de couplage entre les pendules
        self.b = 0.1            # Coefficient d'amortissement
        self.dt = 0.05          # Pas de temps pour l'intégration (s)
        self.max_torque = 5.0   # Couple maximal appliqué sur le premier pendule

        # Espaces d'action et d'observation
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        # L'état est défini par [theta1, theta2, theta3, omega1, omega2, omega3]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # Contrôle du nombre d'étapes
        self.max_steps = max_steps
        self.current_step = 0

        self.render_mode = render_mode
        self.state = None

        # Paramètres d'affichage
        self.screen = None
        self.screen_width = 800
        self.screen_height = 600
        self.pivot = (self.screen_width // 2, 100)  # Point d'attache des pendules
        self.clock = None

    def reset(self):
        """
        Réinitialise l'environnement et retourne l'observation initiale.
        """
        self.current_step = 0
        # Initialisation proche de la position verticale inversée, avec un léger bruit
        theta1 = 0.0 + np.random.uniform(-0.05, 0.05)
        theta2 = 0.0 + np.random.uniform(-0.05, 0.05)
        theta3 = 0.0 + np.random.uniform(-0.05, 0.05)
        omega1 = 0.0
        omega2 = 0.0
        omega3 = 0.0
        self.state = np.array([theta1, theta2, theta3, omega1, omega2, omega3], dtype=np.float32)
        if self.render_mode == "human":
            self._init_render()
        return self.state, {}

    def step(self, action):
        """
        Applique une action (couple) et intègre la dynamique du système.
        Renvoie (observation, terminé).
        """
        torque = np.clip(action[0], -self.max_torque, self.max_torque)
        self.state = self._rk4_step(self.state, torque)
        self.current_step += 1

        # Calcul d'une récompense : inciter à rapprocher les angles de 0 (position verticale inversée)
        reward = - (self.state[0]**2 + self.state[1]**2 + self.state[2]**2) - 0.01 * (torque**2)

        # Critère d'arrêt : si trop d'étapes ou si un angle s'éloigne trop de la verticale
        done = bool(
            self.current_step >= self.max_steps or
            np.abs(self.state[0]) > np.pi/2 or
            np.abs(self.state[1]) > np.pi/2 or
            np.abs(self.state[2]) > np.pi/2
        )

        return self.state, done

    def _rk4_step(self, state, torque):
        """
        Intègre les équations de la dynamique à l'aide de la méthode RK4.
        """
        dt = self.dt

        def dynamics(x, torque):
            theta1, theta2, theta3, omega1, omega2, omega3 = x
            dtheta1 = omega1
            dtheta2 = omega2
            dtheta3 = omega3
            # Equations de la dynamique non linéaire avec couplage et amortissement
            domega1 = (self.g / self.L) * np.sin(theta1) - self.k * (theta1 - theta2) - self.b * omega1 + torque
            domega2 = (self.g / self.L) * np.sin(theta2) - self.k * ((theta2 - theta1) + (theta2 - theta3)) - self.b * omega2
            domega3 = (self.g / self.L) * np.sin(theta3) - self.k * (theta3 - theta2) - self.b * omega3
            return np.array([dtheta1, dtheta2, dtheta3, domega1, domega2, domega3], dtype=np.float32)

        k1 = dynamics(state, torque)
        k2 = dynamics(state + 0.5 * dt * k1, torque)
        k3 = dynamics(state + 0.5 * dt * k2, torque)
        k4 = dynamics(state + dt * k3, torque)
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return new_state

    def render(self, episode=None, epsilon=0):
        """
        Affiche l'état actuel du pendule triple à l'aide de pygame.
        """
        if self.render_mode != "human":
            return
        if self.screen is None:
            self._init_render()

        # Efface l'écran
        self.screen.fill((240, 240, 245))

        # Calcul des positions des points d'attache des pendules
        theta1, theta2, theta3, _, _, _ = self.state
        L_pixels = self.L * 200  # Echelle : 200 pixels par mètre

        # Position du premier point (pivot)
        x0, y0 = self.pivot

        # Premier pendule
        x1 = x0 + L_pixels * np.sin(theta1)
        y1 = y0 - L_pixels * np.cos(theta1)
        pygame.draw.line(self.screen, (50, 50, 60), (int(x0), int(y0)), (int(x1), int(y1)), 6)
        pygame.draw.circle(self.screen, (30, 30, 30), (int(x0), int(y0)), 8)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(x0), int(y0)), 5)

        # Deuxième pendule
        x2 = x1 + L_pixels * np.sin(theta2)
        y2 = y1 - L_pixels * np.cos(theta2)
        pygame.draw.line(self.screen, (60, 180, 60), (int(x1), int(y1)), (int(x2), int(y2)), 6)
        pygame.draw.circle(self.screen, (30, 30, 30), (int(x1), int(y1)), 8)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(x1), int(y1)), 5)

        # Troisième pendule
        x3 = x2 + L_pixels * np.sin(theta3)
        y3 = y2 - L_pixels * np.cos(theta3)
        pygame.draw.line(self.screen, (60, 60, 220), (int(x2), int(y2)), (int(x3), int(y3)), 6)
        pygame.draw.circle(self.screen, (30, 30, 30), (int(x2), int(y2)), 8)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(x2), int(y2)), 5)
        pygame.draw.circle(self.screen, (30, 30, 30), (int(x3), int(y3)), 8)
        pygame.draw.circle(self.screen, (90, 90, 100), (int(x3), int(y3)), 5)

        # Affichage des informations (épisode, étape, epsilon)
        font = pygame.font.Font(None, 24)
        info_text = f"Episode: {episode if episode is not None else '-'}    Steps: {self.current_step}    Epsilon: {epsilon:.2f}"
        text_surface = font.render(info_text, True, (40, 40, 40))
        self.screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Simulation du Pendule Triple Inversé")
        self.clock = pygame.time.Clock()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
