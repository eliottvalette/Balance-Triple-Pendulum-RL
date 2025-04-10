import numpy as np
import pygame
from numpy.linalg import solve
from numpy import pi, cos, sin, hstack, zeros, ones
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
import sys

class TriplePendulumEnv:
    def __init__(self, reward_manager=None, render_mode=None, num_nodes=3, arm_length=1./3, bob_mass=0.01/3, friction_coefficient=0.1):
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
        self.q = dynamicsymbols(f'q:{num_nodes + 1}')  # Coordonnées généralisées
        self.u = dynamicsymbols(f'u:{num_nodes + 1}')  # Vitesses généralisées
        self.f = dynamicsymbols('f')                    # Force appliquée au chariot

        self.m = symbols(f'm:{num_nodes + 1}')          # Masses
        self.l = symbols(f'l:{num_nodes}')              # Longueurs
        self.g, self.t = symbols('g t')                 # Gravité et temps

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
        I = ReferenceFrame('I')
        O = Point('O')
        O.set_vel(I, 0)

        P0 = Point('P0')
        P0.set_pos(O, self.q[0] * I.x)
        P0.set_vel(I, self.u[0] * I.x)
        Pa0 = Particle('Pa0', P0, self.m[0])

        frames = [I]
        points = [P0]
        particles = [Pa0]

        force_cart = self.f * I.x
        weight_cart = -self.m[0] * self.g * I.y
        friction_cart = -self.friction_coefficient * self.u[0] * I.x
        forces = [(P0, force_cart + weight_cart + friction_cart)]
        kindiffs = [self.q[0].diff(self.t) - self.u[0]]

        for i in range(self.n):
            Bi = I.orientnew(f'B{i}', 'Axis', [self.q[i + 1], I.z])
            Bi.set_ang_vel(I, self.u[i + 1] * I.z)
            frames.append(Bi)

            Pi = points[-1].locatenew(f'P{i + 1}', self.l[i] * Bi.x)
            Pi.v2pt_theory(points[-1], I, Bi)
            points.append(Pi)

            Pai = Particle(f'Pa{i + 1}', Pi, self.m[i + 1])
            particles.append(Pai)

            weight = -self.m[i + 1] * self.g * I.y
            friction = -self.friction_coefficient * self.u[i + 1] * I.z
            forces.append((Pi, weight + friction))
            kindiffs.append(self.q[i + 1].diff(self.t) - self.u[i + 1])

        self.kane = KanesMethod(I, q_ind=self.q, u_ind=self.u, kd_eqs=kindiffs)
        self.fr = self.kane._form_fr(forces)
        self.frstar = self.kane._form_frstar(particles)

    def _setup_numeric_evaluation(self):
        parameters = [self.g, self.m[0]]
        self.parameter_vals = [9.81, self.bob_mass]

        for i in range(self.n):
            parameters += [self.l[i], self.m[i + 1]]
            self.parameter_vals += [self.arm_length, self.bob_mass]

        dynamic = self.q + self.u
        dynamic.append(self.f)
        dummy_symbols = [Dummy() for _ in dynamic]
        dummy_dict = dict(zip(dynamic, dummy_symbols))
        kindiff_dict = self.kane.kindiffdict()

        M = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
        F = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)

        self.M_func = lambdify(dummy_symbols + parameters, M)
        self.F_func = lambdify(dummy_symbols + parameters, F)

    def rhs(self, x, t, args, controller=None):
        u_input = controller(x) if controller else 0.0
        arguments = hstack((x, u_input, args))
        dx = np.array(solve(self.M_func(*arguments), self.F_func(*arguments))).T[0]
        return dx

    def reset(self):
        # Initialisation de l'état
        position_initiale_chariot = 0.0
        angles_initiaux = -pi / 2
        vitesses_initiales = 1e-3
        state = hstack((
            position_initiale_chariot,
            angles_initiaux * ones(len(self.q) - 1),
            vitesses_initiales * ones(len(self.u))
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
        self.current_state: [x, q1, q2, q3, u1, u2, u3, f, x1, y1, x2, y2, x3, y3]
        x: position du chariot
        q1, q2, q3: angles des pendules
        u1, u2, u3: vitesses angulaires des pendules
        f: force appliquée au chariot
        x1, y1, x2, y2, x3, y3: positions des masses des pendules
        Returns:
            np.array: L'état actuel du système avec les positions x,y de chaque noeud
        """
        if self.current_state is None:
            return None
        
        # Calculer les positions x et y de tous les noeuds
        cart_x = self.current_state[0]
        
        # Point d'attache sur le chariot
        attach_x = cart_x
        attach_y = 0
        
        # Position de la première masse (après le premier bras)
        x1 = attach_x + self.arm_length * np.cos(self.current_state[1])
        y1 = attach_y + self.arm_length * np.sin(self.current_state[1])
        
        # Position de la deuxième masse (après le deuxième bras)
        x2 = x1 + self.arm_length * np.cos(self.current_state[2])
        y2 = y1 + self.arm_length * np.sin(self.current_state[2])
        
        # Position de la troisième masse (après le troisième bras)
        x3 = x2 + self.arm_length * np.cos(self.current_state[3])
        y3 = y2 + self.arm_length * np.sin(self.current_state[3])
        
        # Ajouter les positions x et y à l'état
        state_with_positions = np.hstack((
            self.current_state,
            x1, y1, x2, y2, x3, y3
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
        dx = self.rhs(self.current_state, self.current_time, self.parameter_vals, lambda x: self.applied_force)
        next_state = self.current_state + dx * self.dt
        
        # Vérifier les limites du chariot
        num_joints = len(self.q)
        cart_position = next_state[0]
        if cart_position - self.cart_width/(2*self.scale) < self.xmin:
            next_state[0] = self.xmin + self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        elif cart_position + self.cart_width/(2*self.scale) > self.xmax:
            next_state[0] = self.xmax - self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        
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
        for x_val in np.arange(self.xmin, self.xmax + 0.5, 0.5):
            x_screen = self._convert_to_screen_coords(x_val, 0)[0]
            pygame.draw.line(self.screen, self.GRID_COLOR, (x_screen, 0), (x_screen, self.height), 1)
        
        for y_val in np.arange(-1, 1.1, 0.5):
            y_screen = self._convert_to_screen_coords(0, y_val)[1]
            pygame.draw.line(self.screen, self.GRID_COLOR, (0, y_screen), (self.width, y_screen), 1)
        
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
        cart_x = self.current_state[0]
        cart_screen_x, cart_screen_y = self._convert_to_screen_coords(cart_x - self.cart_width/(2*self.scale), self.cart_height/(2*self.scale))
        pygame.draw.rect(self.screen, self.CART_COLOR, (cart_screen_x, cart_screen_y, self.cart_width, self.cart_height))
        
        # Dessiner le surlignage du chariot
        highlight_x = cart_screen_x + 4
        highlight_y = cart_screen_y + 4
        highlight_width = self.cart_width - 8
        highlight_height = self.cart_height // 3
        pygame.draw.rect(self.screen, (80, 80, 90), (highlight_x, highlight_y, highlight_width, highlight_height))
        
        # Dessiner le pendule
        num_joints = len(self.q)
        x_positions = hstack((self.current_state[0], zeros(num_joints - 1)))
        y_positions = zeros(num_joints)
        
        for joint in range(1, num_joints):
            x_positions[joint] = x_positions[joint - 1] + self.arm_length * cos(self.current_state[joint])
            y_positions[joint] = y_positions[joint - 1] + self.arm_length * sin(self.current_state[joint])
        
        current_angle = self.current_state[1]
        color_index = min(2, int(abs(current_angle) / (pi/2)))
        
        for i in range(num_joints - 1):
            start_x, start_y = self._convert_to_screen_coords(x_positions[i], y_positions[i])
            end_x, end_y = self._convert_to_screen_coords(x_positions[i+1], y_positions[i+1])
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
            steps (int): Nombre de pas de simulation
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
                        print(f'State: {state}')
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
                        print(f'[x2, y2]: [{state[10]:.2f}, {state[11]:.2f}]')
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