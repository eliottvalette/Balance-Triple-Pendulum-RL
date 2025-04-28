import numpy as np
import random as rd
import pygame
from numpy.linalg import solve
from numpy import pi, cos, sin, hstack, zeros, ones
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
from config import config
import sys

GRAVITY = config['gravity']
DT = 0.01

class TriplePendulumEnv:
    def __init__(self, reward_manager=None, render_mode=None, num_nodes=config['num_nodes'], arm_length=1./3, bob_mass=0.01/3, friction_coefficient=config['friction_coefficient']):
        self.reward_manager = reward_manager
        self.render_mode = render_mode
        self.n = num_nodes
        self.arm_length = arm_length
        self.bob_mass = bob_mass
        self.friction_coefficient = friction_coefficient
        self.cart_friction = 0.1

        # Paramètre de simulation pas-à-pas
        self.dt = DT  # Durée d'un pas de simulation
        self.current_state = None
        self.current_time = 0.0
        self.applied_force = 0.0

        # -----------------------------
        # Modèle symbolique
        # -----------------------------
        self.positions = dynamicsymbols(f'q:{num_nodes + 1}')   # Coordonnées généralisées
        self.velocities = dynamicsymbols(f'u:{num_nodes + 1}')  # Vitesses généralisées
        self.force = dynamicsymbols('f')                        # Force appliquée au chariot

        self.masses = symbols(f'm:{num_nodes + 1}')             # Masses
        self.lengths = symbols(f'l:{num_nodes}')                # Longueurs
        self.gravity, self.time = symbols('g t')                # Gravité et temps
        self.friction_symbol = symbols('b')

        self._setup_symbolic_model()
        self._setup_numeric_evaluation()
        
        # Pygame initialization
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.scale = 200  # Pixels par unité de longueur
        
        # Limites pour le chariot
        self.window_width = 4.0
        self.xmin, self.xmax = -self.window_width / 2, self.window_width / 2
        
        # Dimensions du chariot
        self.cart_width, self.cart_height = 0.4 * self.scale, 0.2 * self.scale
        
        # Couleurs
        self.BACKGROUND_COLOR = (240, 240, 245)
        self.CART_COLOR = (50, 50, 60)
        self.TRACK_COLOR = (180, 180, 190)
        self.PENDULUM_COLORS = [(220, 60, 60), (60, 180, 60)]
        self.TEXT_COLOR = (60, 60, 70)
        self.GRID_COLOR = (210, 210, 215)
    
    def _render_init(self):
        self._init_pygame()

    def _setup_symbolic_model(self):
        inertial_frame = ReferenceFrame('I')
        origin = Point('O')
        origin.set_vel(inertial_frame, 0)

        cart_point = Point('P0')
        cart_point.set_pos(origin, self.positions[0] * inertial_frame.x)
        cart_point.set_vel(inertial_frame, self.velocities[0] * inertial_frame.x)
        cart_particle = Particle('Pa0', cart_point, self.masses[0])

        frames = [inertial_frame]
        points = [cart_point]
        particles = [cart_particle]

        force_cart = self.force * inertial_frame.x
        weight_cart = -self.masses[0] * self.gravity * inertial_frame.y
        friction_cart = -self.cart_friction * self.velocities[0] * inertial_frame.x
        forces = [(cart_point, force_cart + weight_cart + friction_cart)]
        kindiffs = [self.positions[0].diff(self.time) - self.velocities[0]]

        for i in range(self.n):
            pendulum_frame = inertial_frame.orientnew(f'B{i}', 'Axis', [self.positions[i + 1], inertial_frame.z])
            pendulum_frame.set_ang_vel(inertial_frame, self.velocities[i + 1] * inertial_frame.z)
            frames.append(pendulum_frame)

            pendulum_point = points[-1].locatenew(f'P{i + 1}', self.lengths[i] * pendulum_frame.x)
            pendulum_point.v2pt_theory(points[-1], inertial_frame, pendulum_frame)
            points.append(pendulum_point)

            pendulum_particle = Particle(f'Pa{i + 1}', pendulum_point, self.masses[i + 1])
            particles.append(pendulum_particle)

            weight = -self.masses[i + 1] * self.gravity * inertial_frame.y
            friction = -self.friction_symbol * self.velocities[i + 1] * inertial_frame.z
            forces.append((pendulum_point, weight + friction))
            kindiffs.append(self.positions[i + 1].diff(self.time) - self.velocities[i + 1])

        self.kane = KanesMethod(inertial_frame, q_ind=self.positions, u_ind=self.velocities, kd_eqs=kindiffs)
        self.fr = self.kane._form_fr(forces)
        self.frstar = self.kane._form_frstar(particles)

    def _setup_numeric_evaluation(self):
        parameters = [self.gravity, self.masses[0]]
        self.parameter_vals = [GRAVITY, self.bob_mass]

        for i in range(self.n):
            parameters += [self.lengths[i], self.masses[i + 1]]
            self.parameter_vals += [self.arm_length, self.bob_mass]

        parameters.append(self.friction_symbol)
        self.parameter_vals.append(self.friction_coefficient)

        dynamic = self.positions + self.velocities
        dynamic.append(self.force)
        dummy_symbols = [Dummy() for _ in dynamic]
        dummy_dict = dict(zip(dynamic, dummy_symbols))
        kindiff_dict = self.kane.kindiffdict()

        mass_matrix = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
        forcing_vector = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)

        self.M_func = lambdify(dummy_symbols + parameters, mass_matrix)
        self.F_func = lambdify(dummy_symbols + parameters, forcing_vector)

    def rhs(self, state, time, args, controller=None):
        control_input = controller(state) if controller else 0.0
        arguments = hstack((state, control_input, args))
        state_derivative = np.array(solve(self.M_func(*arguments), self.F_func(*arguments))).T[0]
        return state_derivative

    def reset(self):
        # Initialisation de l'état
        position_initiale_chariot = 0.0
        rd_angle = pi/2 # rd.uniform(-pi, pi)
        angles_initiaux = [rd_angle] + [rd_angle] * (len(self.positions) - 2)
        vitesses_initiales = 0.0
        state = hstack((
            position_initiale_chariot,
            angles_initiaux,
            vitesses_initiales * ones(len(self.velocities))
        ))
        self.current_state = state.copy()  # On stocke l'état courant
        self.current_time = 0.0            # Réinitialisation du temps courant
        self.dt = DT
        self.num_steps = 0
        if self.reward_manager is not None:
            self.reward_manager.reset()
        return state

    def _init_pygame(self):
        if not self.pygame_initialized:
            pygame.init()
            self.width, self.height = 800, 600
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Triple Pendule Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 16)
            self.pygame_initialized = True

    def _convert_to_screen_coords(self, x, y):
        # Convertit les coordonnées physiques en coordonnées d'écran
        screen_x = self.width // 2 + int(x * self.scale)
        screen_y = self.height // 2 - int(y * self.scale)  # Y inversé dans pygame
        return screen_x, screen_y
    
    def _calculate_base_state(self):
        """
        Méthode interne pour calculer l'état de base sans risque de récursion infinie.
        Calcule juste les positions des pendules sans les composants de récompense.
        """
        if self.current_state is None:
            return None
        
        non_adapted_state = self.current_state
        
        if self.n == 1:
            adapted_state = [non_adapted_state[0], non_adapted_state[1], 0, 0, non_adapted_state[2], 0, 0, non_adapted_state[3]]
        elif self.n == 2:
            adapted_state = [non_adapted_state[0], non_adapted_state[1], non_adapted_state[2], 0, non_adapted_state[3], non_adapted_state[4], 0, non_adapted_state[5]]
        elif self.n == 3:
            adapted_state = non_adapted_state
        
        # Calculer les positions x et y de tous les noeuds
        cart_position = adapted_state[0]
        
        # Point d'attache sur le chariot
        attach_x = cart_position
        attach_y = 0
        
        # Position de la première masse (après le premier bras)
        position_x1 = attach_x + self.arm_length * np.cos(adapted_state[1])
        position_y1 = attach_y + self.arm_length * np.sin(adapted_state[1])
        
        # Position de la deuxième masse (après le deuxième bras)
        position_x2 = position_x1 + self.arm_length * np.cos(adapted_state[2])
        position_y2 = position_y1 + self.arm_length * np.sin(adapted_state[2])
        
        # Position de la troisième masse (après le troisième bras)
        position_x3 = position_x2 + self.arm_length * np.cos(adapted_state[3])
        position_y3 = position_y2 + self.arm_length * np.sin(adapted_state[3])
        
        # Retourne un état de base avec uniquement les positions
        return adapted_state, position_x1, position_y1, position_x2, position_y2, position_x3, position_y3
    
    def get_state(self, action):
        """
        Renvoie l'état courant du système enrichi avec des combinaisons de features.
        """
        if self.current_state is None:
            return None

        # Obtenir l'état de base (positions sans récompenses)
        base_result = self._calculate_base_state()
        if base_result is None:
            return None

        adapted_state, position_x1, position_y1, position_x2, position_y2, position_x3, position_y3 = base_result

        # ---------------- Reward Components -----------------------
        _, reward_components, _ = self.reward_manager.calculate_reward (
            np.hstack((adapted_state, position_x1, position_y1, position_x2, position_y2, position_x3, position_y3)),
            False,
            self.num_steps,
            action
        )
        
        x_penalty = reward_components['x_penalty']
        upright_reward = reward_components['upright_reward']
        non_alignement_penalty = reward_components['non_alignement_penalty']
        stability_penalty = reward_components['stability_penalty']
        mse_penalty = reward_components['mse_penalty']
        consecutive_upright_steps = self.reward_manager.consecutive_upright_steps / 150
        have_been_upright_once = self.reward_manager.have_been_upright_once
        came_back_down = self.reward_manager.came_back_down
        steps_double_down = self.reward_manager.steps_double_down / 150
        time_over_threshold = self.reward_manager.time_over_threshold / 150
        smoothed_variation = self.reward_manager.smoothed_variation

        # ----------------- Indicateurs logiques -------------------
        near_border = float(abs(adapted_state[0]) > 1.6)
        end_node_y = position_y3 if self.n == 3 else position_y2 if self.n == 2 else position_y1
        end_node_upright = float(end_node_y > self.reward_manager.max_height * self.reward_manager.threshold_ratio)
        is_node_on_right_of_cart = float(position_x3 > adapted_state[0])
        normalized_steps = self.num_steps / 500

        # ----------------- Feature Engineering -------------------
        q1, q2, q3 = adapted_state[1:4]
        u1, u2, u3 = adapted_state[4:7]

        # Combinaisons d'angles
        if self.n > 1:
            sin_diff_12 = np.sin(q1 - q2)
        else:
            sin_diff_12 = 0
        
        if self.n > 2:
            cos_sum_23 = np.cos(q2 + q3)
        else:
            cos_sum_23 = 0

        # Interaction angle × vitesse
        if self.n > 1:
            v1_angle1 = u1 * q1
        else:
            v1_angle1 = 0

        if self.n > 2:
            v2_angle2 = u2 * q2
        else:
            v2_angle2 = 0

        # Approximation énergie
        KE = 0.5 * (u1 ** 2 + u2 ** 2 + u3 ** 2)
        PE = -GRAVITY * end_node_y

        # Distances inter-masses
        if self.n > 1:
            d12 = np.linalg.norm([position_x2 - position_x1, position_y2 - position_y1])
        else:
            d12 = 0

        if self.n > 2:
            d23 = np.linalg.norm([position_x3 - position_x2, position_y3 - position_y2])
        else:
            d23 = 0
        # ----------- Construction finale de l'état ---------------
        state_with_positions = np.hstack((
            adapted_state,
            position_x1, position_y1, position_x2, position_y2, position_x3, position_y3,
            x_penalty, upright_reward, non_alignement_penalty, stability_penalty, mse_penalty,
            consecutive_upright_steps, have_been_upright_once, came_back_down, steps_double_down,
            near_border, end_node_y, end_node_upright, time_over_threshold, smoothed_variation, 
            is_node_on_right_of_cart, normalized_steps,
            sin_diff_12, cos_sum_23, v1_angle1, v2_angle2,
            KE, PE, d12, d23
        ))

        return state_with_positions


    def step(self, action=0.0, manual_mode=False):
        """
        Effectue un pas de simulation avec l'action donnée (force appliquée).
        
        Args:
            action (float): Force appliquée au chariot
            
        Returns:
            np.array: Le nouvel état après le pas de simulation
        """
        if self.current_state is None:
            self.reset()
        
        if (self.current_state[0] > 1.65 and action > 0) :
            action = -0.1
        elif (self.current_state[0] < -1.65 and action < 0):
            action = 0.1
            
        # Appliquer le lissage à la force
        force_smoothing = 0.1
        if manual_mode:
            self.applied_force += force_smoothing * (action - self.applied_force)
        else:
            self.applied_force = action
        
        # Calcul du nouvel état
        state_derivative = self.rhs(self.current_state, self.current_time, self.parameter_vals, lambda state: self.applied_force)
        next_state = self.current_state + state_derivative * self.dt
        
        # Vérifier les limites du chariot
        num_joints = len(self.positions)
        cart_position = next_state[0]
        if cart_position - self.cart_width/(2*self.scale) < self.xmin:
            next_state[0] = self.xmin + self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        elif cart_position + self.cart_width/(2*self.scale) > self.xmax:
            next_state[0] = self.xmax - self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        
        # Appliquer un amortissement supplémentaire direct aux vitesses
        # Coefficient d'amortissement: plus élevé pour une dissipation d'énergie plus rapide
        damping_factor = 0.99
        
        # Appliquer l'amortissement aux vitesses angulaires (u1, u2, u3)
        for i in range(num_joints, 2*num_joints):
            next_state[i] *= damping_factor
        
        # Mise à jour de l'état et du temps
        self.current_state = next_state
        self.current_time += self.dt

        terminated = False # abs(self.current_state[0]) > 1.6
        
        return self.get_state(action), terminated
        
    def render(self, episode = 0, epsilon = 0, current_step = 0):
        """
        Affiche l'état actuel du pendule.
        """
        self._init_pygame()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if self.pygame_initialized:
                    pygame.quit()
                    self.pygame_initialized = False
                return False
        
        # Dessiner le fond
        self.screen.fill(self.BACKGROUND_COLOR)
        
        # Dessiner la grille
        for position_x in np.arange(self.xmin, self.xmax + 0.5, 0.5):
            grid_x = self._convert_to_screen_coords(position_x, 0)[0]
            pygame.draw.line(self.screen, self.GRID_COLOR, (grid_x, 0), (grid_x, self.height), 1)
        
        for position_y in np.arange(-1, 1.1, 0.5):
            grid_y = self._convert_to_screen_coords(0, position_y)[1]
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, grid_y), (self.width, grid_y), 1)
        
        # Dessiner la piste
        track_x, track_y = self._convert_to_screen_coords(self.xmin, self.cart_height/(2*self.scale) - 0.05)
        track_width = int((self.xmax - self.xmin) * self.scale)
        track_height = int(0.05 * self.scale)
        pygame.draw.rect(self.screen, self.TRACK_COLOR, (track_x, track_y, track_width, track_height))

        # dessiner un barre horizontale rouge qui indique le upright threshold
        upright_threshold = self.reward_manager.max_height * self.reward_manager.threshold_ratio
        upright_threshold_x, upright_threshold_y = self._convert_to_screen_coords(self.xmin, upright_threshold)
        pygame.draw.rect(self.screen, (255, 0, 0), (upright_threshold_x, upright_threshold_y, self.width, 1))
        
        # Dessiner le repère central
        center_x = self._convert_to_screen_coords(0, self.cart_height/(2*self.scale) - 0.025)[0]
        center_y = self._convert_to_screen_coords(0, self.cart_height/(2*self.scale) - 0.025)[1]
        pygame.draw.line(self.screen, (100, 100, 110), (center_x, center_y - 10), (center_x, center_y + 10), 2)
        
        # Dessiner le chariot
        cart_position = self.current_state[0]
        cart_screen_x, cart_screen_y = self._convert_to_screen_coords(cart_position - self.cart_width/(2*self.scale), self.cart_height/(2*self.scale))
        pygame.draw.rect(self.screen, self.CART_COLOR, (cart_screen_x, cart_screen_y, self.cart_width, self.cart_height))
        
        # Dessiner le surlignage du chariot
        highlight_x = cart_screen_x + 4
        highlight_y = cart_screen_y + 4
        highlight_width = self.cart_width - 8
        highlight_height = self.cart_height // 3
        pygame.draw.rect(self.screen, (80, 80, 90), (highlight_x, highlight_y, highlight_width, highlight_height))
        
        # Dessiner le pendule
        num_joints = len(self.positions)
        pendulum_x_positions = hstack((self.current_state[0], zeros(num_joints - 1)))
        pendulum_y_positions = zeros(num_joints)
        
        for joint in range(1, num_joints):
            pendulum_x_positions[joint] = pendulum_x_positions[joint - 1] + self.arm_length * cos(self.current_state[joint])
            pendulum_y_positions[joint] = pendulum_y_positions[joint - 1] + self.arm_length * sin(self.current_state[joint])
                
        for i in range(num_joints - 1):
            start_x, start_y = self._convert_to_screen_coords(pendulum_x_positions[i], pendulum_y_positions[i])
            end_x, end_y = self._convert_to_screen_coords(pendulum_x_positions[i+1], pendulum_y_positions[i+1])
            if end_y > self.height//2:
                color_index = 0
            else :
                color_index = 1
            pygame.draw.line(self.screen, self.PENDULUM_COLORS[color_index], (start_x, start_y), (end_x, end_y), 4)
            pygame.draw.circle(self.screen, (90, 90, 100), (end_x, end_y), 8)
            pygame.draw.circle(self.screen, (30, 30, 40), (end_x, end_y), 8, 1)
        
        # Afficher les infos de base
        time_text = self.font.render(f'time = {self.current_time:.2f}', True, self.TEXT_COLOR)
        # Convertir la force en nombre à virgule flottante si c'est un tableau numpy
        force_value = float(self.applied_force) if isinstance(self.applied_force, np.ndarray) else self.applied_force
        force_text = self.font.render(f'force = {force_value:.2f}', True, self.TEXT_COLOR)
        info_text = self.font.render('Utilisez les flèches gauche/droite pour appliquer une force', True, self.TEXT_COLOR)
        
        self.screen.blit(time_text, (20, 20))
        self.screen.blit(force_text, (20, 45))
        self.screen.blit(info_text, (20, self.height - 30))
        
        # Afficher les composants de récompense si le reward_manager est disponible
        if self.reward_manager is not None:
            # Calculer l'état une seule fois et l'utiliser pour tout le rendu
            base_state_result = self._calculate_base_state()
            
            if base_state_result is not None:
                adapted_state, position_x1, position_y1, position_x2, position_y2, position_x3, position_y3 = base_state_result
                
                # Créer un état temporaire pour le reward manager
                temp_state = np.hstack((
                    adapted_state,
                    position_x1, position_y1, position_x2, position_y2, position_x3, position_y3
                ))
                
                # Récupérer les composants de récompense
                _, reward_components, _ = self.reward_manager.calculate_reward(temp_state, False, current_step, self.applied_force)
                
                # Dessiner un conteneur pour les récompenses
                reward_panel_width = 300
                reward_panel_height = 240
                reward_panel_x = self.width - reward_panel_width - 10
                reward_panel_y = 10
                
                # Fond du conteneur avec bordure arrondie
                # Créer une surface avec canal alpha
                panel_surface = pygame.Surface((reward_panel_width, reward_panel_height), pygame.SRCALPHA)
                # Dessiner le fond semi-transparent
                pygame.draw.rect(panel_surface, (230, 230, 235, 180), 
                                pygame.Rect(0, 0, reward_panel_width, reward_panel_height),
                                border_radius=10)
                # Dessiner la bordure semi-transparente
                pygame.draw.rect(panel_surface, (200, 200, 205, 200), 
                                pygame.Rect(0, 0, reward_panel_width, reward_panel_height),
                                border_radius=10, width=2)
                # Appliquer la surface sur l'écran
                self.screen.blit(panel_surface, (reward_panel_x, reward_panel_y))
                
                # Titre du conteneur
                title_font = pygame.font.Font(None, 28)
                title_text = title_font.render('Composants de Récompense', True, (60, 60, 70))
                self.screen.blit(title_text, (reward_panel_x + 10, reward_panel_y + 10))
                
                # Séparateur
                pygame.draw.line(
                    self.screen, 
                    (200, 200, 205), 
                    (reward_panel_x + 10, reward_panel_y + 40), 
                    (reward_panel_x + reward_panel_width - 10, reward_panel_y + 40), 
                    2
                )
                
                # Configuration des barres de récompense
                bar_y = reward_panel_y + 50
                bar_width = reward_panel_width - 40
                bar_height = 16
                bar_spacing = 28
                max_bar_value = 3.0  # Échelle pour la visualisation
                
                # Définir les composants de récompense avec leurs couleurs spécifiques
                reward_components_display = [
                    {"name": "Base", "value": reward_components.get('reward', 0), "color": (100, 100, 200)},
                    {"name": "Upright", "value": reward_components.get('upright_reward', 0), "color": (80, 180, 80)},
                    {"name": "Position", "value": reward_components.get('x_penalty', 0), "color": (200, 80, 80)},
                    {"name": "Alignment", "value": reward_components.get('non_alignement_penalty', 0), "color": (180, 130, 80)},
                    {"name": "MSE", "value": reward_components.get('mse_penalty', 0), "color": (180, 130, 80)},
                    {"name": "Hera", "value": reward_components.get('heraticness_penalty', 0), "color": (180, 130, 80)}
                ]
                
                # Ajouter stability_penalty s'il existe
                if 'stability_penalty' in reward_components:
                    reward_components_display.append({"name": "Stability", "value": reward_components.get('stability_penalty', 0), "color": (150, 80, 150)})
                
                # Ajouter x_dot_penalty s'il existe
                if 'x_dot_penalty' in reward_components:
                    reward_components_display.append({"name": "Velocity", "value": reward_components.get('x_dot_penalty', 0), "color": (180, 80, 180)})
                
                # Dessiner les barres de récompense
                for comp in reward_components_display:
                    # Nom du composant
                    name_text = self.font.render(comp["name"], True, (60, 60, 70))
                    self.screen.blit(name_text, (reward_panel_x + 15, bar_y))
                    
                    # Calculer le point central pour la barre (point zéro)
                    center_x = reward_panel_x + 100 + (bar_width - 130) / 2
                    usable_width = bar_width - 130
                    
                    # Barre de fond
                    pygame.draw.rect(
                        self.screen,
                        (220, 220, 225),
                        pygame.Rect(center_x - usable_width/2, bar_y + 5, usable_width, bar_height),
                        border_radius=4
                    )
                    
                    # Ligne centrale pour marquer le zéro
                    pygame.draw.line(
                        self.screen, 
                        (150, 150, 155), 
                        (center_x, bar_y + 3), 
                        (center_x, bar_y + bar_height + 7), 
                        2
                    )
                    
                    # Barre de valeur
                    value = comp["value"]
                    
                    # Vérifier si value est un tableau numpy et le convertir en float si nécessaire
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            value = float(value)
                        else:
                            value = float(value.mean())  # Si c'est un tableau à plusieurs éléments, prendre la moyenne
                    
                    if value != 0:
                        if value > 0:
                            bar_length = min(value / max_bar_value * (usable_width/2), usable_width/2)
                            bar_x = center_x
                            bar_color = comp["color"]
                        else:
                            bar_length = min(abs(value) / max_bar_value * (usable_width/2), usable_width/2)
                            bar_x = center_x - bar_length
                            bar_color = (200, 90, 90)  # Rouge pour les valeurs négatives
                        
                        # S'assurer que la longueur de la barre est toujours valide et d'au moins 1 pixel
                        bar_length = max(1, int(bar_length))
                        
                        pygame.draw.rect(
                            self.screen,
                            bar_color,
                            pygame.Rect(int(bar_x), int(bar_y + 5), bar_length, bar_height),
                            border_radius=4
                        )
                    
                    # Afficher la valeur à droite
                    value_text = self.font.render(f"{value:.2f}", True, (60, 60, 70))
                    self.screen.blit(value_text, (reward_panel_x + bar_width - 30, bar_y + 5))
                    
                    bar_y += bar_spacing
                
                # Afficher informations sur l'épisode
                episode_text = self.font.render(f'Episode: {episode}', True, self.TEXT_COLOR)
                epsilon_text = self.font.render(f'Epsilon: {epsilon:.4f}', True, self.TEXT_COLOR)
                self.screen.blit(episode_text, (20, 70))
                self.screen.blit(epsilon_text, (20, 95))
        
        pygame.display.flip()
        self.clock.tick(60)
        
        return True

    def animate_pendulum_pygame(self, max_steps, manual_mode, title):
        """
        Anime le pendule en utilisant les méthodes step et render.
        
        Args:
            max_steps (int): Nombre maximal de pas de simulation avant réinitialisation
            states (np.array, optional): États préalablement calculés. Si None, ils seront générés.
            length (float, optional): Longueur des bras du pendule. Si None, utilise self.arm_length.
            title (str, optional): Titre de la fenêtre.
        """
        self._init_pygame()
        pygame.display.set_caption(title)
        
        # Créer un RewardManager si aucun n'est disponible
        if self.reward_manager is None:
            try:
                from reward import RewardManager
                self.reward_manager = RewardManager()
            except ImportError:
                print("ATTENTION: Impossible d'importer RewardManager. Les récompenses ne seront pas affichées.")
        
        if self.current_state is None:
            self.reset()
        
        force_increment = 0.5
        target_force = 0.0
        force_smoothing = 0.1
        running = True
        episode = 0
        epsilon = 1.0
        
        while running:
            if self.num_steps >= max_steps:
                self.reset()
                self.num_steps = 0
                episode += 1
                epsilon *= 0.995  # Simuler une décroissance d'epsilon
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        target_force = -force_increment
                    elif event.key == pygame.K_RIGHT:
                        target_force = force_increment
                    elif event.key == pygame.K_SPACE:
                        target_force = 0.0
                        self.reset()
                    elif event.key == pygame.K_s:
                        state = self.get_state()
                        print(f'State: {state}, length: {len(state)}')
                        print('------- Details: -------')
                        print(f'x: {state[0]}')
                        print(f'q1: {state[1]}')
                        print(f'q2: {state[2]}')
                        print(f'q3: {state[3]}')
                        print(f'u1: {state[4]}')
                        print(f'u2: {state[5]}')
                        print(f'u3: {state[6]}')
                        print(f'f: {state[7]}')
                        print(f'[x1, y1]: [{state[8]:.2f}, {state[9]:.2f}]')
                        if self.n > 1:
                            print(f'[x2, y2]: [{state[10]:.2f}, {state[11]:.2f}]')
                        if self.n > 2:
                            print(f'[x3, y3]: [{state[12]:.2f}, {state[13]:.2f}]')
                        print('------- Fin des details -------')
                        
                        # Afficher également les composants de récompense si disponibles
                        if self.reward_manager is not None:
                            _, reward_components, _ = self.reward_manager.calculate_reward(state, False, self.num_steps)
                            print('------- Composants de récompense -------')
                            for component, value in reward_components.items():
                                print(f'{component}: {value:.4f}')
                            print('------- Fin des composants de récompense -------')
                        
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        target_force = 0.0
            
            # Mise à jour de la force et de l'état
            next_state, terminated = self.step(target_force, manual_mode)
            
            # Rendu avec informations sur l'épisode et epsilon
            if not self.render(episode=episode, epsilon=epsilon, current_step=self.num_steps):
                break

            if terminated:
                self.reset()

            self.num_steps += 1
        
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

# Exemple d'utilisation
if __name__ == "__main__":
    env = TriplePendulumEnv()
    
    # Utilisation avec les nouvelles méthodes
    env.reset()
    env.animate_pendulum_pygame(max_steps=10_000, manual_mode = False, title='Simulation Triple Pendule')