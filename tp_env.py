import numpy as np
import pygame
from numpy.linalg import solve
from numpy import pi, cos, sin, hstack, zeros, ones
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
import sys

GRAVITY = 9.81


class TriplePendulumEnv:
    def __init__(self, reward_manager=None, render_mode=None, num_nodes=1, arm_length=1./3, bob_mass=0.01/3, friction_coefficient=0.1):
        self.reward_manager = reward_manager
        self.render_mode = render_mode
        self.n = num_nodes
        self.arm_length = arm_length
        self.bob_mass = bob_mass
        self.friction_coefficient = friction_coefficient

        # Paramètre de simulation pas-à-pas
        self.dt = 0.01  # Durée d'un pas de simulation
        self.current_state = None
        self.current_time = 0.0
        self.applied_force = 0.0

        # -----------------------------
        # Modèle symbolique
        # -----------------------------
        self.positions = dynamicsymbols(f'q:{num_nodes + 1}')  # Coordonnées généralisées
        self.velocities = dynamicsymbols(f'u:{num_nodes + 1}')  # Vitesses généralisées
        self.force = dynamicsymbols('f')                    # Force appliquée au chariot

        self.masses = symbols(f'm:{num_nodes + 1}')          # Masses
        self.lengths = symbols(f'l:{num_nodes}')              # Longueurs
        self.gravity, self.time = symbols('g t')              # Gravité et temps

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
        self.PENDULUM_COLORS = [(220, 60, 60), (60, 180, 60), (60, 60, 220)]
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
        friction_cart = -self.friction_coefficient * self.velocities[0] * inertial_frame.x
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
            friction = -self.friction_coefficient * self.velocities[i + 1] * inertial_frame.z
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
        angles_initiaux = -pi / 2
        vitesses_initiales = 1e-3
        state = hstack((
            position_initiale_chariot,
            angles_initiaux * ones(len(self.positions) - 1),
            vitesses_initiales * ones(len(self.velocities))
        ))
        self.current_state = state.copy()  # On stocke l'état courant
        self.current_time = 0.0            # Réinitialisation du temps courant
        self.dt = 0.01
        self.num_steps = 0
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
    
    def get_state(self):
        """
        Renvoie l'état courant du système.
        self.current_state: [x, q1, ..., qi, u1, u2, ..., ui, f] avec self.n = i
        x: position du chariot
        q1, q2, q3: angles des pendules
        u1, u2, u3: vitesses angulaires des pendules
        f: force appliquée au chariot    
            adapted_state: [x, q1, q2, q3, u1, u2, u3, f, x1, y1, x2, y2, x3, y3]
            x1, y1, x2, y2, x3, y3: positions des masses des pendules
            np.array: L'état actuel du système avec les positions x,y de chaque noeud
        """
        if self.current_state is None:
            return None
        
        non_adapted_state = self.current_state
        
        if self.n == 1:
            adapted_state = [non_adapted_state[0], non_adapted_state[1], 0, 0, non_adapted_state[2], 0, 0, non_adapted_state[3]]
        elif self.n == 2:
            adapted_state = [non_adapted_state[0], non_adapted_state[1], non_adapted_state[2], 0, non_adapted_state[3], non_adapted_state[4], 0, non_adapted_state[6]]
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
        
        # Ajouter les positions x et y à l'état
        state_with_positions = np.hstack((
            adapted_state,
            position_x1, position_y1, position_x2, position_y2, position_x3, position_y3
        ))
        
        return state_with_positions


    def step(self, action=0.0):
        """
        Effectue un pas de simulation avec l'action donnée (force appliquée).
        
        Args:
            action (float): Force appliquée au chariot
            
        Returns:
            np.array: Le nouvel état après le pas de simulation
        """
        if self.current_state is None:
            self.reset()
            
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
        
        return self.current_state.copy()
        
    def render(self):
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
        
        current_angle = self.current_state[1]
        color_index = min(2, int(abs(current_angle) / (pi/2)))
        
        for i in range(num_joints - 1):
            start_x, start_y = self._convert_to_screen_coords(pendulum_x_positions[i], pendulum_y_positions[i])
            end_x, end_y = self._convert_to_screen_coords(pendulum_x_positions[i+1], pendulum_y_positions[i+1])
            pygame.draw.line(self.screen, self.PENDULUM_COLORS[color_index], (start_x, start_y), (end_x, end_y), 4)
            pygame.draw.circle(self.screen, (90, 90, 100), (end_x, end_y), 8)
            pygame.draw.circle(self.screen, (30, 30, 40), (end_x, end_y), 8, 1)
        
        # Afficher les infos
        time_text = self.font.render(f'time = {self.current_time:.2f}', True, self.TEXT_COLOR)
        force_text = self.font.render(f'force = {self.applied_force:.2f}', True, self.TEXT_COLOR)
        info_text = self.font.render('Utilisez les flèches gauche/droite pour appliquer une force', True, self.TEXT_COLOR)
        
        self.screen.blit(time_text, (20, 20))
        self.screen.blit(force_text, (20, 45))
        self.screen.blit(info_text, (20, self.height - 30))
        
        pygame.display.flip()
        self.clock.tick(60)
        
        return True

    def animate_pendulum_pygame(self, max_steps, states=None, length=None, title='Pendulum'):
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
        
        if self.current_state is None:
            self.reset()
        
        force_increment = 0.3
        target_force = 0.0
        force_smoothing = 0.1
        running = True
        
        while running:
            if self.num_steps >= max_steps:
                self.reset()
                self.num_steps = 0
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
                        
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        target_force = 0.0
            
            # Mise à jour de la force et de l'état
            self.applied_force += force_smoothing * (target_force - self.applied_force)
            self.step(self.applied_force)
            
            # Rendu
            if not self.render():
                break

            self.num_steps += 1
        
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

# Exemple d'utilisation
if __name__ == "__main__":
    env = TriplePendulumEnv()
    
    # Utilisation avec les nouvelles méthodes
    env.reset()
    env.animate_pendulum_pygame(max_steps=10_000, title='Simulation Triple Pendule')