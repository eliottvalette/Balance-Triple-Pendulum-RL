import gym
import numpy as np
import pygame
import math
from reward import RewardManager

# Pour le modèle symbolique
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols

class TriplePendulumEnv(gym.Env):
    """
    Environnement Gym pour un pendule triple utilisant la physique du modèle.
    L'interface (méthodes reset, step, render, etc.) reste inchangée.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, reward_manager: RewardManager, render_mode=None, num_nodes=3):
        super(TriplePendulumEnv, self).__init__()
        self.reward_manager = reward_manager
        self.render_mode = render_mode

        # --- Configuration graphique ---
        self.screen = None
        self.screen_width = 800
        self.screen_height = 700
        self.cart_y_pos = self.screen_height // 2    # Position verticale pour le rendu
        self.pixels_per_meter = 100
        self.tick = 30
        self.clock = pygame.time.Clock()
        self.font = None

        # --- Configuration physique (inspirée de tp_env_new) ---
        self.n = num_nodes            # Nombre de pendules (q: q0 = position, q1,...,q_n = angles)
        self.dt = 0.01                # Pas de temps pour l'intégration
        self.current_time = 0.0
        self.current_state = None     # Etat sous la forme [q0, q1, ... q_n, u0, u1, ... u_n]
        self.applied_force = 0.0      # Force appliquée sur le chariot

        # Paramètres pour la modélisation symbolique
        self.friction_coefficient = 0.1
        # Coordonnées généralisées : q[0] = position, q[1:] = angles
        self.q = dynamicsymbols(f'q:{self.n+1}')
        self.u = dynamicsymbols(f'u:{self.n+1}')  # vitesses associées
        self.f_sym = dynamicsymbols('f')          # force appliquée sur le chariot

        # Symboles physiques : masse et longueur
        self.m = symbols(f'm:{self.n+1}')         # m[0] : masse du chariot, m[1:] : masses des pendules
        self.l = symbols(f'l:{self.n}')           # longueurs des bras (pendules)
        self.g, self.t = symbols('g t')

        # Construction du modèle symbolique et fonctions numériques
        self._setup_symbolic_model()
        self._setup_numeric_evaluation()

        # Valeurs numériques des paramètres :
        # Pour le chariot : g = 9.81, masse = 0.01/3
        # Pour chaque pendule : longueur = 1./3, masse = 0.01/3
        self.parameter_vals = [9.81, 0.01/3]
        for i in range(self.n):
            self.parameter_vals += [1./3, 0.01/3]

        # Définition de l'action : force continue sur le chariot
        self.force_mag = 20.0
        self.action_space = gym.spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)

        # Définition de l'observation (taille indicative pour compatibilité)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)

    def _setup_symbolic_model(self):
        """
        Construit le modèle symbolique via la méthode de Kane.
        """
        I = ReferenceFrame('I')
        O = Point('O')
        O.set_vel(I, 0)

        # Chariot
        P0 = Point('P0')
        P0.set_pos(O, self.q[0] * I.x)
        P0.set_vel(I, self.u[0] * I.x)
        Pa0 = Particle('Pa0', P0, self.m[0])

        frames = [I]
        points = [P0]
        particles = [Pa0]

        force_cart = self.f_sym * I.x
        weight_cart = -self.m[0] * self.g * I.y
        friction_cart = -self.friction_coefficient * self.u[0] * I.x
        forces = [(P0, force_cart + weight_cart + friction_cart)]
        kindiffs = [self.q[0].diff(self.t) - self.u[0]]

        for i in range(self.n):
            Bi = I.orientnew(f'B{i}', 'Axis', [self.q[i+1], I.z])
            Bi.set_ang_vel(I, self.u[i+1] * I.z)
            frames.append(Bi)

            Pi = points[-1].locatenew(f'P{i+1}', self.l[i] * Bi.x)
            Pi.v2pt_theory(points[-1], I, Bi)
            points.append(Pi)

            Pai = Particle(f'Pa{i+1}', Pi, self.m[i+1])
            particles.append(Pai)

            weight = -self.m[i+1] * self.g * I.y
            friction = -self.friction_coefficient * self.u[i+1] * I.z
            forces.append((Pi, weight + friction))
            kindiffs.append(self.q[i+1].diff(self.t) - self.u[i+1])

        self.kane = KanesMethod(I, q_ind=self.q, u_ind=self.u, kd_eqs=kindiffs)
        self.fr = self.kane._form_fr(forces)
        self.frstar = self.kane._form_frstar(particles)

    def _setup_numeric_evaluation(self):
        """
        Crée les fonctions numériques pour la matrice d'inertie et le vecteur de forces.
        """
        dynamic = list(self.q) + list(self.u)
        dynamic.append(self.f_sym)
        dummy_symbols = [Dummy() for _ in dynamic]
        dummy_dict = dict(zip(dynamic, dummy_symbols))
        kindiff_dict = self.kane.kindiffdict()
        M = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
        F = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)
        self.M_func = lambdify(dummy_symbols + self._get_parameter_symbols(), M, modules='numpy')
        self.F_func = lambdify(dummy_symbols + self._get_parameter_symbols(), F, modules='numpy')

    def _get_parameter_symbols(self):
        """
        Retourne la liste des symboles de paramètres dans l'ordre attendu.
        """
        params = [self.g, self.m[0]]
        for i in range(self.n):
            params += [self.l[i], self.m[i+1]]
        return params

    def reset(self):
        """
        Initialise l'état du système :
          - q[0] est la position initiale du chariot (0.0)
          - q[1:] sont les angles initiaux (ici -pi/2 pour avoir le pendule vers le bas)
          - u correspond aux vitesses (toutes faibles)
        """
        from numpy import ones, hstack, pi
        position_initial = 0.0
        angles_init = -pi/2
        q_init = np.array([position_initial] + [angles_init] * self.n)
        u_init = 1e-3 * np.ones(self.n + 1)
        self.current_state = np.hstack((q_init, u_init))
        self.current_time = 0.0

        if self.render_mode == "human" and self.screen is None:
            self._render_init()

        return self.current_state, {}

    def rhs(self, x, t, args, controller=None):
        """
        Calcule la dérivée de l'état à partir du modèle numérique.
        """
        u_input = controller(x) if controller else 0.0
        arguments = np.hstack((x, u_input, args))
        from numpy.linalg import solve
        M = self.M_func(*arguments)
        F = self.F_func(*arguments)
        dx = np.array(solve(M, F)).flatten()
        return dx

    def step(self, action):
        """
        Effectue un pas de simulation par intégration d'Euler explicite.
        L'action correspond à la force appliquée sur le chariot.
        """
        force = np.clip(action[0], -self.force_mag, self.force_mag)
        dx = self.rhs(self.current_state, self.current_time, self.parameter_vals, lambda x: force)
        self.current_state = self.current_state + dx * self.dt
        self.current_time += self.dt

        terminated = bool(abs(self.current_state[0]) > 3.2)
        return self.current_state, terminated

    def get_rich_state(self, state):
        """
        Renvoie une représentation riche de l'état.
        L'état se compose de :
          - state[0] : position du chariot
          - state[1:self.n+1] : angles des pendules
          - state[self.n+1:] : vitesses (du chariot et des pendules)
        Pour le rendu, la conversion des angles utilise cos et sin.
        """
        x = state[0]
        x_dot = state[1] if len(state) > 1 else 0.0
        x_ddot = 0.0  # Non calculé explicitement ici

        # Extraction des angles
        pendulum_angles = state[1:self.n+1]
        pendulum_states = [[angle, 0.0, 0.0] for angle in pendulum_angles]

        # Conversion en pixels
        cart_x_px = int(self.screen_width / 2 + x * self.pixels_per_meter)
        cart_y_px = int(self.cart_y_pos)

        def calculate_following_node(origin_x, origin_y, angle):
            # Conversion conforme au modèle : x += link_length*cos(angle), y += link_length*sin(angle)
            link_length = (1./3) * self.pixels_per_meter
            end_x = origin_x + link_length * math.cos(angle)
            end_y = origin_y + link_length * math.sin(angle)
            return end_x, end_y

        pivot1_x, pivot1_y = cart_x_px, cart_y_px
        th1 = pendulum_states[0][0] if pendulum_states else 0.0
        end1_x, end1_y = calculate_following_node(pivot1_x, pivot1_y, th1)
        if self.n > 1:
            th2 = pendulum_states[1][0]
            end2_x, end2_y = calculate_following_node(end1_x, end1_y, th2)
        else:
            end2_x, end2_y = (0, 0)
        if self.n > 2:
            th3 = pendulum_states[2][0]
            end3_x, end3_y = calculate_following_node(end2_x, end2_y, th3)
        else:
            end3_x, end3_y = (0, 0)

        pivot1_x, pivot1_y = pivot1_x / self.screen_width, pivot1_y / self.screen_height
        end1_x, end1_y = end1_x / self.screen_width, end1_y / self.screen_height
        end2_x, end2_y = end2_x / self.screen_width, end2_y / self.screen_height
        end3_x, end3_y = end3_x / self.screen_width, end3_y / self.screen_height

        close_to_left = (x < -2.4)
        close_to_right = (x > 2.4)
        normalized_consecutive_upright_steps = 0.0
        is_long_upright = False

        # Récompenses et pénalités (mis à zéro ici)
        reward = 0
        upright_reward = 0
        x_penalty = 0
        non_alignement_penalty = 0
        stability_penalty = 0
        mse_penalty = 0
        have_been_upright_once = 0
        came_back_down = 0
        steps_double_down = 0
        force_terminated = 0

        model_state = np.concatenate([
            [x, x_dot, x_ddot],
            np.array([val for sub in pendulum_states for val in sub]),
            [pivot1_x, pivot1_y, end1_x, end1_y, end2_x, end2_y, end3_x, end3_y,
             close_to_left, close_to_right, normalized_consecutive_upright_steps, is_long_upright],
            [reward, upright_reward, x_penalty, non_alignement_penalty, stability_penalty, mse_penalty],
            [have_been_upright_once, came_back_down, steps_double_down, force_terminated]
        ])
        return model_state

    def render(self, episode=None, epsilon=0):
        """
        Rend la simulation via pygame.
        """
        if self.render_mode != "human":
            return

        if self.screen is None:
            self._render_init()

        # Couleurs et paramètres graphiques
        BACKGROUND_COLOR = (240, 240, 245)
        CART_COLOR = (50, 50, 60)
        TRACK_COLOR = (180, 180, 190)
        PENDULUM_COLORS = [
            (220, 60, 60),
            (60, 180, 60),
            (60, 60, 220)
        ]
        TEXT_COLOR = (40, 40, 40)
        GRID_COLOR = (210, 210, 215)

        self.screen.fill(BACKGROUND_COLOR)
        grid_spacing = 50
        for x in range(0, self.screen_width, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.screen_width, y))

        track_height = 5
        track_width = 3.2 * 2 * self.pixels_per_meter
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
        center_mark_height = 10
        pygame.draw.line(
            self.screen,
            (100, 100, 110),
            (self.screen_width // 2, self.cart_y_pos + 15),
            (self.screen_width // 2, self.cart_y_pos + 15 + center_mark_height),
            2
        )

        # Récupère l'état courant
        x = self.current_state[0]
        cart_x_px = int(self.screen_width / 2 + x * self.pixels_per_meter)
        cart_y_px = int(self.cart_y_pos)
        cart_w, cart_h = 60, 30
        cart_rect = pygame.Rect(
            cart_x_px - cart_w // 2,
            cart_y_px - cart_h // 2,
            cart_w,
            cart_h
        )
        shadow_offset = 3
        shadow_rect = pygame.Rect(
            cart_rect.left + shadow_offset,
            cart_rect.top + shadow_offset,
            cart_rect.width,
            cart_rect.height
        )
        pygame.draw.rect(self.screen, (180, 180, 190), shadow_rect, border_radius=5)
        pygame.draw.rect(self.screen, CART_COLOR, cart_rect, border_radius=5)
        highlight_rect = pygame.Rect(
            cart_rect.left + 3,
            cart_rect.top + 3,
            cart_rect.width - 6,
            10
        )
        pygame.draw.rect(self.screen, (80, 80, 90), highlight_rect, border_radius=3)

        # Fonction locale pour dessiner un lien de pendule
        def draw_link(origin_x, origin_y, angle, color):
            link_length = (1./3) * self.pixels_per_meter
            # Respecte la physique du modèle : utilisation de cos pour x, sin pour y
            end_x = origin_x + link_length * math.cos(angle)
            end_y = origin_y + link_length * math.sin(angle)
            pygame.draw.line(
                self.screen,
                color,
                (origin_x, origin_y),
                (end_x, end_y),
                6
            )
            pygame.draw.circle(self.screen, (30, 30, 30), (int(origin_x), int(origin_y)), 7)
            pygame.draw.circle(self.screen, (90, 90, 100), (int(origin_x), int(origin_y)), 5)
            return end_x, end_y

        pivot_x, pivot_y = cart_x_px, cart_y_px - cart_h // 2
        # Dessin des pendules en chaîne
        for i in range(self.n):
            th = self.current_state[1 + i]  # q[1+i] = angle du ième pendule
            pivot_x, pivot_y = draw_link(pivot_x, pivot_y, th, PENDULUM_COLORS[i % len(PENDULUM_COLORS)])

        pygame.display.flip()
        self.clock.tick(self.tick)

    def _render_init(self):
        """
        Initialise la fenêtre d'affichage avec pygame.
        """
        if not pygame.get_init():
            pygame.init()
        pygame.display.set_caption("Triple Pendulum Simulation")
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            try:
                self.font = pygame.font.SysFont("Arial", 18)
            except:
                self.font = pygame.font.Font(None, 24)
            pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    def apply_brake(self):
        """
        Applique un freinage d'urgence sur le chariot.
        """
        if self.current_state is not None:
            current_velocity = self.current_state[1]  # u0 représente la vitesse du chariot
            if abs(current_velocity) > 0.01:
                braking_direction = -1 if current_velocity > 0 else 1
                braking_force = abs(current_velocity) * 5.0
                braking_force = min(braking_force, self.force_mag)
                return np.array([braking_direction * braking_force], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)
