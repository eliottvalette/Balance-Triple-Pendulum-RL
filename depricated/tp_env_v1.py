# tp_env_old.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.integrate import odeint
from numpy.linalg import solve, matrix_rank
from numpy import pi, cos, sin, hstack, ones
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols
import control

class TriplePendulumEnv:
    def __init__(self, render_mode=None, num_nodes=3, arm_length=1./3, bob_mass=0.01/3, friction_coefficient=0.1, dt=0.01):
        self.render_mode = render_mode
        self.n = num_nodes
        self.arm_length = arm_length
        self.bob_mass = bob_mass
        self.friction_coefficient = friction_coefficient
        self.dt = dt  # Pas de temps pour l'intégration
        self.force_mag = 10.0  # Échelle de force pour le contrôle (utilisée dans l'action)

        # Variables symboliques (généralisées) pour la dynamique (même physique que tp_env.py)
        self.q = dynamicsymbols(f'q:{num_nodes + 1}')  # q0 pour le chariot, puis les angles
        self.u = dynamicsymbols(f'u:{num_nodes + 1}')  # vitesses associées
        self.f = dynamicsymbols('f')  # force appliquée au chariot

        self.m = symbols(f'm:{num_nodes + 1}')   # masses (chariot + bobs)
        self.l = symbols(f'l:{num_nodes}')         # longueurs des bras
        self.g, self.t = symbols('g t')            # gravité et temps

        # Mise en place du modèle symbolique et de l'évaluation numérique de la dynamique
        self._setup_symbolic_model()
        self._setup_numeric_evaluation()

        # Valeurs numériques des paramètres physiques : [g, m0, l0, m1, l1, ...]
        self.parameter_vals = [9.81, self.bob_mass]
        for i in range(self.n):
            self.parameter_vals += [self.arm_length, self.bob_mass]

        # Initialisation de l'état
        self.state, _ = self.reset()
    
    def _setup_symbolic_model(self):
        # Cadre inerte et point d'origine
        I = ReferenceFrame('I')
        O = Point('O')
        O.set_vel(I, 0)

        # Position et dynamique du chariot
        P0 = Point('P0')
        P0.set_pos(O, self.q[0] * I.x)
        P0.set_vel(I, self.u[0] * I.x)
        Pa0 = Particle('Pa0', P0, self.m[0])

        self.frames = [I]
        self.points = [P0]
        self.particles = [Pa0]

        # Forces sur le chariot
        force_cart = self.f * I.x
        weight_cart = -self.m[0] * self.g * I.y
        friction_cart = -self.friction_coefficient * self.u[0] * I.x
        self.forces = [(P0, force_cart + weight_cart + friction_cart)]
        kindiffs = [self.q[0].diff(self.t) - self.u[0]]

        # Boucle sur chaque pendule
        for i in range(self.n):
            Bi = I.orientnew(f'B{i}', 'Axis', [self.q[i+1], I.z])
            Bi.set_ang_vel(I, self.u[i+1] * I.z)
            self.frames.append(Bi)

            Pi = self.points[-1].locatenew(f'P{i+1}', self.l[i] * Bi.x)
            Pi.v2pt_theory(self.points[-1], I, Bi)
            self.points.append(Pi)

            Pai = Particle(f'Pa{i+1}', Pi, self.m[i+1])
            self.particles.append(Pai)

            weight = -self.m[i+1] * self.g * I.y
            friction = -self.friction_coefficient * self.u[i+1] * I.z
            self.forces.append((Pi, weight + friction))
            kindiffs.append(self.q[i+1].diff(self.t) - self.u[i+1])
        
        self.kane = KanesMethod(I, q_ind=self.q, u_ind=self.u, kd_eqs=kindiffs)
        self.fr = self.kane._form_fr(self.forces)
        self.frstar = self.kane._form_frstar(self.particles)

    def _setup_numeric_evaluation(self):
        # Constitution du vecteur dynamique : état = q + u puis la force f
        dynamic = list(self.q) + list(self.u) + [self.f]
        dummy_symbols = [Dummy() for _ in dynamic]
        dummy_dict = dict(zip(dynamic, dummy_symbols))
        kindiff_dict = self.kane.kindiffdict()

        # Matrice de masse et forcing
        M = self.kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
        F = self.kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)

        self.M_func = lambdify(dummy_symbols + [self.g] + list(self.m) + list(self.l), M)
        self.F_func = lambdify(dummy_symbols + [self.g] + list(self.m) + list(self.l), F)

    def dynamics(self, x, t, applied_force):
        """
        Calcule la dérivée d'état dx/dt pour l'état x et la force appliquée (applied_force).
        """
        args = np.concatenate((x, [applied_force], self.parameter_vals))
        M_val = self.M_func(*args)
        F_val = self.F_func(*args)
        # Résolution de M * dx = F
        dx = np.linalg.solve(M_val, F_val).flatten()
        return dx

    def step(self, action):
        """
        Effectue un pas de simulation avec l'action donnée.
        L'action est un scalaire dans [-1, 1] (multiplié par force_mag).
        Renvoie (état suivant, reward, done, info)
        """
        force = action * self.force_mag
        t_span = [0, self.dt]
        next_state = odeint(lambda x, t: self.dynamics(x, t, force), self.state, t_span)[-1]
        self.state = next_state

        # Exemple de fonction de reward : on cherche à maintenir les angles proches de -pi/2
        angles = self.state[1:self.n+1]
        reward = -np.sum((angles - (-pi/2))**2)

        # Condition terminale : dépassement de la position du chariot ou angles trop grands
        done = (np.abs(self.state[0]) > 2.4) or (np.any(np.abs(angles) > pi))
        info = {}
        return self.state, reward, done, info

    def reset(self):
        """
        Réinitialise l'environnement à son état de départ.
        Renvoie (état initial, info)
        """
        position_initiale_chariot = 0.0
        angles_initiaux = -pi/2
        vitesses_initiales = 1e-3
        state = hstack((
            position_initiale_chariot,
            angles_initiaux * ones(len(self.q) - 1),
            vitesses_initiales * ones(len(self.u))
        ))
        self.state = state
        return state, {}

    def get_rich_state(self, state=None):
        """
        Renvoie un état « enrichi » pour l'entraînement (ici, simplement le vecteur d'état).
        """
        if state is None:
            state = self.state
        return state

    def _render_init(self):
        """
        Initialisation du rendu (pour affichage en mode "human").
        """
        if self.render_mode == "human":
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim([-3, 3])
            self.ax.set_ylim([-2, 2])
            plt.ion()
            plt.show()

    def render(self, episode=None, epsilon=None):
        """
        Rendu visuel de l'état courant.
        Les paramètres episode et epsilon sont affichés dans le titre.
        """
        if self.render_mode != "human":
            return
        self.ax.clear()
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-2, 2])
        cart_width = 0.4
        cart_height = 0.2
        # Dessin du chariot
        cart = Rectangle((self.state[0]-cart_width/2, -cart_height/2), cart_width, cart_height, color='black')
        self.ax.add_patch(cart)
        # Calcul des positions des pendules
        x_positions = [self.state[0]]
        y_positions = [0]
        angle = self.state[1]
        for i in range(1, self.n):
            x_new = x_positions[-1] + self.arm_length * cos(angle)
            y_new = y_positions[-1] + self.arm_length * sin(angle)
            x_positions.append(x_new)
            y_positions.append(y_new)
            # Pour un pendule multiple, on cumule les angles (supposés relatifs)
            angle += self.state[1+i]
        self.ax.plot(x_positions, y_positions, marker='o', color='blue')
        title = f"Episode: {episode}" if episode is not None else "Triple Pendulum"
        if epsilon is not None:
            title += f"   Epsilon: {epsilon}"
        self.ax.set_title(title)
        plt.pause(0.001)

    def seed(self, seed_value):
        np.random.seed(seed_value)

# Exemple d'utilisation isolée (non RL)
if __name__ == "__main__":
    env = TriplePendulumEnv(render_mode="human")
    env._render_init()
    state, _ = env.reset()
    for _ in range(200):
        # Exemple d'action aléatoire dans [-1, 1]
        action = np.random.uniform(-1, 1)
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            state, _ = env.reset()