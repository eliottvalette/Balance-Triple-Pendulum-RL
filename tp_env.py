# tp_env.py
import numpy as np
import random as rd
import pygame
from numpy.linalg import solve
from numpy import pi, cos, sin, hstack, zeros
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
from config import config, validate_config
from reward import RewardManager

DT = 0.01
FIXED_NUM_NODES = 2

class TriplePendulumEnv:
    def __init__(
        self,
        reward_manager=None,
        render_mode=None,
        num_nodes=FIXED_NUM_NODES,
        arm_length=1.0 / 3.0,
        bob_mass=0.01 / 3.0,
        friction_coefficient=None,
        max_steps=None,
        env_config=None,
    ):
        self.config = config if env_config is None else env_config
        validate_config(self.config)
        if num_nodes != FIXED_NUM_NODES:
            raise ValueError(f"TriplePendulumEnv is fixed to {FIXED_NUM_NODES} nodes, got {num_nodes}")
        if self.config["num_nodes"] != FIXED_NUM_NODES:
            raise ValueError(
                f"config['num_nodes'] must be {FIXED_NUM_NODES}, got {self.config['num_nodes']}"
            )
        self.reward_manager = reward_manager
        self.render_mode = render_mode
        self.n = FIXED_NUM_NODES
        self.arm_length = arm_length
        self.bob_mass = bob_mass
        self.friction_coefficient = (
            float(self.config["friction_coefficient"])
            if friction_coefficient is None
            else float(friction_coefficient)
        )
        self.cart_friction = 0.1
        resolved_max_steps = self.config["max_steps"] if max_steps is None else max_steps
        if (
            not isinstance(resolved_max_steps, int)
            or isinstance(resolved_max_steps, bool)
            or resolved_max_steps <= 0
        ):
            raise ValueError(f"max_steps must be a positive integer, got {resolved_max_steps!r}")
        self.max_steps = resolved_max_steps
        # Paramètre de simulation pas-à-pas
        self.dt = DT  # Durée d'un pas de simulation
        self.current_state = None
        self.current_time = 0.0
        self.applied_force = 0.0
        self.previous_action = 0.0
        self.current_phase = 1
        self.initial_phase = 1
        self.switch_step = int(self.max_steps * 0.5)
        self.has_switched = False
        self.capture_started = False
        self.hold_streak = 0
        self.success_achieved = False
        if self.reward_manager is not None:
            self.reward_manager.verify_cart_termination_is_suboptimal(
                max_steps=self.max_steps,
                gamma=float(self.config["gamma"]),
            )

        # -----------------------------
        # Modèle symbolique
        # -----------------------------
        self.positions = dynamicsymbols(f'q:{FIXED_NUM_NODES + 1}')   # Coordonnées généralisées
        self.velocities = dynamicsymbols(f'u:{FIXED_NUM_NODES + 1}')  # Vitesses généralisées
        self.force = dynamicsymbols('f')                        # Force appliquée au chariot

        self.masses = symbols(f'm:{FIXED_NUM_NODES + 1}')             # Masses
        self.lengths = symbols(f'l:{FIXED_NUM_NODES}')                # Longueurs
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
        self.parameter_vals = [float(self.config["gravity"]), self.bob_mass]

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

        # Vérifier le nombre d'arguments attendus
        self.M_func = lambdify(dummy_symbols + parameters, mass_matrix)
        self.F_func = lambdify(dummy_symbols + parameters, forcing_vector)

        # Stocker le nombre d'arguments attendus
        self.num_args = len(dummy_symbols) + len(parameters)

    def rhs(self, state, time, args, controller=None):
        if controller is None:
            raise ValueError("controller is required")
        control_input = controller(state)
        arguments = hstack((state, control_input, args))
        state_derivative = np.array(solve(self.M_func(*arguments), self.F_func(*arguments))).T[0]
        return state_derivative

    def reset(self, phase=None, episode_mode=None):
        # Initialisation de l'état
        position_initiale_chariot = 0.0
        if episode_mode not in (None, "down_to_up", "capture_vertical", "up_to_fold", "fold_to_up"):
            raise ValueError(f"unknown episode_mode: {episode_mode}")
        if phase is not None and phase not in (-1, 1):
            raise ValueError(f"phase must be -1 or 1, got {phase}")
        if episode_mode is None and phase is None:
            raise ValueError("reset() requires an explicit episode_mode or phase")
        use_down_to_up = episode_mode == "down_to_up"
        use_capture_vertical = episode_mode == "capture_vertical"
        self.initial_pose_mode = "down" if use_down_to_up else "capture" if use_capture_vertical else "target"
        if use_down_to_up or use_capture_vertical or episode_mode == "up_to_fold":
            self.initial_phase = 1
        elif episode_mode == "fold_to_up":
            self.initial_phase = -1
        elif phase in (-1, 1):
            self.initial_phase = phase
        else:
            self.initial_phase = phase
        self.current_phase = self.initial_phase
        if use_down_to_up or use_capture_vertical:
            self.switch_step = self.max_steps + 1
        elif episode_mode in ("up_to_fold", "fold_to_up"):
            low = self.config['transition_switch_step_min']
            high = self.config['transition_switch_step_max']
            self.switch_step = rd.randint(low, high)
        else:
            low = int(self.max_steps * 0.40)
            high = int(self.max_steps * 0.60)
            self.switch_step = rd.randint(low, high)
        self.has_switched = False

        angle_noise = (
            self.config['down_angle_noise']
            if use_down_to_up
            else self.config['capture_angle_noise']
            if use_capture_vertical
            else self.config['initial_angle_noise']
        )
        if use_down_to_up:
            base_angles = [-pi / 2] * (len(self.positions) - 1)
        elif self.current_phase == 1:
            base_angles = [pi / 2] * (len(self.positions) - 1)
        else:
            base_angles = [pi / 2, -pi / 2] + [-pi / 2] * (len(self.positions) - 3)

        angles_initiaux = [
            base_angle + rd.uniform(-angle_noise, angle_noise)
            for base_angle in base_angles
        ]
        if use_capture_vertical:
            vitesses_initiales = np.array([
                rd.uniform(-self.config['capture_cart_velocity_noise'], self.config['capture_cart_velocity_noise']),
                *[
                    rd.uniform(
                        -self.config['capture_angular_velocity_noise'],
                        self.config['capture_angular_velocity_noise'],
                    )
                    for _ in range(len(self.velocities) - 1)
                ],
            ])
        else:
            velocity_noise = self.config['initial_velocity_noise']
            vitesses_initiales = np.array([
                rd.uniform(-velocity_noise, velocity_noise)
                for _ in range(len(self.velocities))
            ])
        state = hstack((
            position_initiale_chariot,
            angles_initiaux,
            vitesses_initiales
        ))
        self.current_state = state.copy()  # On stocke l'état courant
        self.current_time = 0.0            # Réinitialisation du temps courant
        self.dt = DT
        self.num_steps = 0
        self.applied_force = 0.0
        self.previous_action = 0.0
        self.capture_started = not use_down_to_up
        self.hold_streak = 0
        self.success_achieved = False
        return self.get_state(action=0.0, phase=self.current_phase)

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
            raise RuntimeError("state requested before reset()")

        adapted_state = np.array(self.current_state, dtype=float)
        
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

        return adapted_state, position_x1, position_y1, position_x2, position_y2

    def _target_points_for_phase(self, cart_x, phase):
        if phase == 1:
            return np.array([
                cart_x, self.arm_length,
                cart_x, 2 * self.arm_length,
            ], dtype=float)
        return np.array([
            cart_x, self.arm_length,
            cart_x, 0.0,
        ], dtype=float)
    
    def get_state(self, action, phase = None):
        """
        Renvoie une observation pure: physique normalisée, contexte de phase,
        cinématique cartésienne et erreurs vers la cible active.
        """
        if self.current_state is None:
            raise RuntimeError("state requested before reset()")

        base_result = self._calculate_base_state()

        adapted_state, position_x1, position_y1, position_x2, position_y2 = base_result
        phase = self.current_phase if phase is None else phase
        if phase not in (-1, 1):
            raise ValueError(f"phase must be -1 or 1, got {phase}")
        if action is None:
            raise ValueError("action cannot be None")
        action = float(action)
        if not np.isfinite(action):
            raise ValueError(f"action must be finite, got {action!r}")

        x = adapted_state[0]
        q1, q2 = adapted_state[1:3]
        x_dot, q1_dot, q2_dot = adapted_state[3:6]
        arm = self.arm_length

        vx1 = x_dot - arm * np.sin(q1) * q1_dot
        vy1 = arm * np.cos(q1) * q1_dot
        vx2 = vx1 - arm * np.sin(q2) * q2_dot
        vy2 = vy1 + arm * np.cos(q2) * q2_dot

        end_node_x = position_x2
        end_node_y = position_y2
        end_node_vx = vx2
        end_node_vy = vy2

        normalized_steps = self.num_steps / max(1, self.max_steps)
        time_to_switch = (self.switch_step - self.num_steps) / max(1, self.max_steps)
        has_switched = float(self.has_switched)
        phase_1 = float(phase == 1)
        initial_pose_down = float(self.initial_pose_mode == "down")
        capture_started = float(self.capture_started)
        hold_progress = self.hold_streak / max(1, self.max_steps)

        max_height = self.arm_length * self.n
        phase1_end_y_error = max_height - end_node_y
        phase1_end_x_error = end_node_x - x
        phase1_alignment_error = 1.0 - np.cos(q1 - q2)

        phase2_y1_error = self.arm_length - position_y1
        phase2_end_y_error = end_node_y
        phase2_end_x_error = end_node_x - x
        phase2_fold_error = 1.0 + np.cos(q1 - q2)

        if phase == 1:
            active_height_error = phase1_end_y_error
            active_x_error = phase1_end_x_error
            active_shape_error = phase1_alignment_error
        else:
            active_height_error = phase2_end_y_error
            active_x_error = phase2_end_x_error
            active_shape_error = phase2_fold_error

        target_velocity_norm = np.sqrt(end_node_vx**2 + end_node_vy**2)
        points = np.array([
            position_x1, position_y1,
            position_x2, position_y2,
        ], dtype=float)
        active_target_points = self._target_points_for_phase(x, phase)
        next_phase = -phase
        next_target_points = self._target_points_for_phase(x, next_phase)
        active_target_delta = active_target_points - points
        next_target_delta = next_target_points - points

        return np.hstack((
            x, x_dot,
            np.sin(q1), np.cos(q1), q1_dot,
            np.sin(q2), np.cos(q2), q2_dot,
            position_x1, position_y1, position_x2, position_y2,
            vx1, vy1, vx2, vy2,
            self.applied_force, self.previous_action, action,
            phase_1, capture_started, initial_pose_down, normalized_steps, time_to_switch, has_switched,
            active_height_error, active_x_error, active_shape_error,
            phase1_end_y_error, phase1_end_x_error, phase1_alignment_error,
            phase2_y1_error, phase2_end_y_error, phase2_end_x_error, phase2_fold_error,
            active_target_points,
            active_target_delta,
            next_target_points,
            next_target_delta,
            target_velocity_norm, hold_progress
        ))

    def get_physical_state(self):
        base_result = self._calculate_base_state()
        adapted_state, position_x1, position_y1, position_x2, position_y2 = base_result
        return np.hstack((
            adapted_state,
            self.applied_force,
            position_x1,
            position_y1,
            position_x2,
            position_y2,
        ))


    def step(self, action=0.0, phase=None):
        """
        Effectue un pas de simulation avec l'action donnée (force appliquée).
        
        Args:
            action (float): Force appliquée au chariot
            manual_mode (bool): Mode manuel ou automatique
            phase (int): Phase actuelle (1 ou -1)
            
        Returns:
            np.array: Le nouvel état après le pas de simulation
        """
        if self.current_state is None:
            raise RuntimeError("step() called before reset()")

        if phase is not None:
            if phase not in (-1, 1):
                raise ValueError(f"phase must be -1 or 1, got {phase}")
            self.current_phase = phase

        switched = False
        if not self.has_switched and self.num_steps >= self.switch_step:
            self.current_phase *= -1
            self.has_switched = True
            switched = True

        action = float(action)
        max_action = float(self.config['max_action'])
        if not np.isfinite(action):
            raise ValueError(f"action must be finite, got {action!r}")
        if not -max_action <= action <= max_action:
            raise ValueError(f"action must be in [{-max_action}, {max_action}], got {action}")
        if self.reward_manager is None:
            raise RuntimeError("step() requires a reward_manager")

        previous_physical_state = self.get_physical_state()
            
        # Appliquer le lissage à la force
        force_smoothing = 0.1
        self.applied_force += force_smoothing * (action - self.applied_force)
        
        # Calcul du nouvel état
        state_derivative = self.rhs(self.current_state, self.current_time, self.parameter_vals, lambda state: self.applied_force)
        if not np.all(np.isfinite(state_derivative)):
            raise FloatingPointError(
                f"non-finite derivative at step={self.num_steps}, "
                f"state={self.current_state}, action={action}, derivative={state_derivative}"
            )
        next_state = self.current_state + state_derivative * self.dt
        if not np.all(np.isfinite(next_state)):
            raise FloatingPointError(
                f"non-finite next_state at step={self.num_steps}, "
                f"state={self.current_state}, action={action}, derivative={state_derivative}"
            )
        
        # Vérifier les limites du chariot
        num_joints = len(self.positions)
        cart_position = next_state[0]
        hit_cart_limit = False
        if cart_position - self.cart_width/(2*self.scale) < self.xmin:
            hit_cart_limit = True
            next_state[0] = self.xmin + self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        elif cart_position + self.cart_width/(2*self.scale) > self.xmax:
            hit_cart_limit = True
            next_state[0] = self.xmax - self.cart_width/(2*self.scale)
            next_state[num_joints] = 0  # Vitesse du chariot à zéro
        
        # Appliquer un amortissement supplémentaire direct aux vitesses
        # Coefficient d'amortissement: plus élevé pour une dissipation d'énergie plus rapide
        damping_factor = 0.99
        
        # Appliquer l'amortissement aux vitesses angulaires (u1, u2)
        for i in range(num_joints, 2*num_joints):
            next_state[i] *= damping_factor
        
        # Mise à jour de l'état et du temps
        self.current_state = next_state
        self.current_time += self.dt

        self.num_steps += 1
        physical_state = self.get_physical_state()
        result = self.reward_manager.evaluate_transition(
            previous_physical_state,
            physical_state,
            action=action,
            phase=self.current_phase,
            capture_started=self.capture_started,
            hold_streak=self.hold_streak,
        )
        self.capture_started = result.capture_started
        self.hold_streak = result.hold_streak
        reward = result.reward
        reward_components = dict(result.components)

        terminated = bool(hit_cart_limit)
        truncated = bool(self.num_steps >= self.max_steps and not terminated)
        if hit_cart_limit:
            reward = float(self.config['cart_failure_penalty'])
            reward_components['reward'] = reward
            reward_components['cart_failure_penalty'] = reward
        else:
            reward_components['cart_failure_penalty'] = 0.0

        self.previous_action = action
        info = {
            "phase": self.current_phase,
            "initial_phase": self.initial_phase,
            "initial_pose_mode": self.initial_pose_mode,
            "switch_step": self.switch_step,
            "switched": switched,
            "capture_started": self.capture_started,
            "success_achieved": self.success_achieved,
            "entered_success": result.success,
            "reward_components": reward_components,
            "target_score": reward_components["target_score"],
            "in_target": reward_components["in_target"],
            "termination_reason": (
                "cart_limit"
                if hit_cart_limit
                else "max_steps"
                if truncated
                else None
            ),
        }
        return (
            self.get_state(action=action, phase=self.current_phase),
            reward,
            terminated,
            truncated,
            info,
        )
        
    def render(self, action, episode = 0, epsilon = 0, current_step = 0, phase = None):
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
        upright_threshold = self.reward_manager.max_height - self.reward_manager.phase_1_height_tolerance
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
        phase_text = self.font.render(f'Phase: {phase} (Touches 1 et 2 pour changer)', True, self.TEXT_COLOR)
        
        self.screen.blit(time_text, (20, 20))
        self.screen.blit(force_text, (20, 45))
        self.screen.blit(info_text, (20, self.height - 30))
        self.screen.blit(phase_text, (20, self.height - 55))
        
        # Afficher les composants de récompense si le reward_manager est disponible
        if self.reward_manager is not None:
            # Calculer l'état une seule fois et l'utiliser pour tout le rendu
            base_state_result = self._calculate_base_state()
            
            if base_state_result is not None:
                adapted_state, position_x1, position_y1, position_x2, position_y2 = base_state_result
                
                # Créer un état temporaire pour le reward manager
                temp_state = np.hstack((
                    adapted_state,
                    self.applied_force,
                    position_x1,
                    position_y1,
                    position_x2,
                    position_y2,
                ))
                
                # Récupérer les composants de récompense
                preview = self.reward_manager.evaluate_transition(
                    temp_state,
                    temp_state,
                    action=float(action),
                    phase=phase,
                    capture_started=self.capture_started,
                    hold_streak=self.hold_streak,
                )
                reward_components = preview.components
                
                # Dessiner un conteneur pour les récompenses
                reward_panel_width = 300
                reward_panel_height = 270
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
                    {"name": "Reward", "value": reward_components['reward'], "color": (100, 100, 200)},
                    {"name": "Target", "value": reward_components['target_score'], "color": (100, 200, 100)},
                    {"name": "Energy", "value": reward_components['energy_score'], "color": (80, 150, 210)},
                    {"name": "Height", "value": reward_components['height_score'], "color": (80, 180, 120)},
                    {"name": "Progress", "value": reward_components['potential_progress'], "color": (180, 130, 80)},
                    {"name": "Capture", "value": reward_components['capture_quality'], "color": (100, 180, 100)},
                    {"name": "Hold", "value": reward_components['hold_progress'], "color": (130, 100, 200)},
                    {"name": "In Target", "value": reward_components['in_target'], "color": (100, 200, 100)}
                ]
             
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
        self.clock.tick(120)
        
        return True

    def animate_pendulum_pygame(self, max_steps, manual_mode, title):
        """
        Anime le pendule en utilisant les méthodes step et render.
        
        Args:
            max_steps (int): Nombre maximal de pas de simulation avant réinitialisation
            manual_mode (bool): Mode manuel ou automatique
            title (str): Titre de la fenêtre.
        """
        self._init_pygame()
        pygame.display.set_caption(title)
        
        if self.reward_manager is None:
            raise RuntimeError("animate() requires a reward_manager")
        
        if self.current_state is None:
            self.reset(phase=1)  # Phase initiale par défaut
        
        force_increment = 0.5
        target_force = 0.0
        running = True
        episode = 0
        epsilon = 1.0
        current_phase = 1  # Phase initiale
        
        while running:
            if self.num_steps >= max_steps:
                self.reset(phase=current_phase)  # Réinitialiser avec la phase courante
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
                        self.reset(phase=current_phase)
                    elif event.key == pygame.K_p:  # Changer vers la phase 1
                        current_phase = 1 if current_phase == -1 else -1
                        self.reset(phase=current_phase)
                    elif event.key == pygame.K_s:
                        state = self.get_state(action=target_force, phase=current_phase)
                        physical_state = self.get_physical_state()
                        print(f'State: {state}, length: {len(state)}')
                        print('------- Details: -------')
                        print(f'x: {physical_state[0]}')
                        print(f'q1: {physical_state[1]}')
                        print(f'q2: {physical_state[2]}')
                        print(f'x_dot: {physical_state[3]}')
                        print(f'u1: {physical_state[4]}')
                        print(f'u2: {physical_state[5]}')
                        print(f'f: {physical_state[6]}')
                        print(f'[x1, y1]: [{physical_state[7]:.2f}, {physical_state[8]:.2f}]')
                        print(f'[x2, y2]: [{physical_state[9]:.2f}, {physical_state[10]:.2f}]')
                        print('------- Fin des details -------')
                        
                        # Afficher également les composants de récompense si disponibles
                        if self.reward_manager is not None:
                            preview = self.reward_manager.evaluate_transition(
                                physical_state,
                                physical_state,
                                action=target_force,
                                phase=current_phase,
                                capture_started=self.capture_started,
                                hold_streak=self.hold_streak,
                            )
                            reward_components = preview.components
                            print('------- Composants de récompense -------')
                            for component, value in reward_components.items():
                                print(f'{component}: {value:.4f}')
                            print('------- Fin des composants de récompense -------')
                        
                elif event.type == pygame.KEYUP:
                    if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        target_force = 0.0
            
            # Mise à jour de la force et de l'état
            _next_state, _reward, terminated, truncated, _info = self.step(
                target_force,
                phase=current_phase,
            )
            
            # Rendu avec informations sur l'épisode et epsilon
            if not self.render(action=target_force, episode=episode, epsilon=epsilon, current_step=self.num_steps, phase=current_phase):
                break

            if terminated or truncated:
                self.reset(phase=current_phase)

        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False

# Exemple d'utilisation
if __name__ == "__main__":
    env = TriplePendulumEnv(reward_manager=RewardManager(config), env_config=config)
    
    # Utilisation avec les nouvelles méthodes
    env.reset(phase = 1)
    env.animate_pendulum_pygame(max_steps=10_000, manual_mode = True, title='Simulation Triple Pendule')
