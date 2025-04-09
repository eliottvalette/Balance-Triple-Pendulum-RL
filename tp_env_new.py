import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from scipy.integrate import odeint
from numpy.linalg import solve, matrix_rank
from numpy import pi, cos, sin, hstack, zeros, linspace, ones, array, matrix, around, dot
from sympy import symbols, Dummy, lambdify
from sympy.physics.mechanics import ReferenceFrame, Point, Particle, KanesMethod, dynamicsymbols

import control

# -----------------------------
# Parameters
# -----------------------------
n = 3  # Triple pendulum
arm_length = 1. / n
bob_mass = 0.01 / n
friction_coefficient = 0.1  # Coefficient de frottement visqueux

# -----------------------------
# Symbolic Model
# -----------------------------
q = dynamicsymbols(f'q:{n + 1}')  # Generalized coordinates
u = dynamicsymbols(f'u:{n + 1}')  # Generalized speeds
f = dynamicsymbols('f')           # Force applied to the cart

m = symbols(f'm:{n + 1}')         # Masses
l = symbols(f'l:{n}')             # Lengths
g, t = symbols('g t')             # Gravity and time

I = ReferenceFrame('I')
O = Point('O')
O.set_vel(I, 0)

P0 = Point('P0')
P0.set_pos(O, q[0] * I.x)
P0.set_vel(I, u[0] * I.x)
Pa0 = Particle('Pa0', P0, m[0])

frames = [I]
points = [P0]
particles = [Pa0]
# Définir les forces comme des tuples (Point, Vector)
force_cart = f * I.x
weight_cart = -m[0] * g * I.y
friction_cart = -friction_coefficient * u[0] * I.x  # Frottement sur le chariot
forces = [(P0, force_cart + weight_cart + friction_cart)]
kindiffs = [q[0].diff(t) - u[0]]

for i in range(n):
    Bi = I.orientnew(f'B{i}', 'Axis', [q[i + 1], I.z])
    Bi.set_ang_vel(I, u[i + 1] * I.z)
    frames.append(Bi)

    Pi = points[-1].locatenew(f'P{i + 1}', l[i] * Bi.x)
    Pi.v2pt_theory(points[-1], I, Bi)
    points.append(Pi)

    Pai = Particle(f'Pa{i + 1}', Pi, m[i + 1])
    particles.append(Pai)

    # Définir la force de poids et le frottement comme des tuples (Point, Vector)
    weight = -m[i + 1] * g * I.y
    friction = -friction_coefficient * u[i + 1] * I.z  # Frottement sur le pendule
    forces.append((Pi, weight + friction))
    kindiffs.append(q[i + 1].diff(t) - u[i + 1])

kane = KanesMethod(I, q_ind=q, u_ind=u, kd_eqs=kindiffs)
# Passer les forces et les particules séparément
fr = kane._form_fr(forces)
frstar = kane._form_frstar(particles)

# -----------------------------
# Numeric evaluation
# -----------------------------
parameters = [g, m[0]]
parameter_vals = [9.81, bob_mass]

for i in range(n):
    parameters += [l[i], m[i + 1]]
    parameter_vals += [arm_length, bob_mass]

dynamic = q + u
dynamic.append(f)
dummy_symbols = [Dummy() for _ in dynamic]
dummy_dict = dict(zip(dynamic, dummy_symbols))
kindiff_dict = kane.kindiffdict()

M = kane.mass_matrix_full.subs(kindiff_dict).subs(dummy_dict)
F = kane.forcing_full.subs(kindiff_dict).subs(dummy_dict)

M_func = lambdify(dummy_symbols + parameters, M)
F_func = lambdify(dummy_symbols + parameters, F)

def rhs(x, t, args, controller=None):
    u_input = controller(x) if controller else 0.0
    arguments = hstack((x, u_input, args))
    dx = array(solve(M_func(*arguments), F_func(*arguments))).T[0]
    return dx

# -----------------------------
# Initial condition & simulation
# -----------------------------
# x0 contient les conditions initiales dans l'ordre suivant :
# [position_chariot, angle1, angle2, angle3, vitesse_chariot, vitesse_angle1, vitesse_angle2, vitesse_angle3]
position_initiale_chariot = 0.0  # Position initiale du chariot
angles_initiaux = -pi / 2  # Angles initiaux des pendules (en radians)
vitesses_initiales = 1e-3  # Vitesses initiales (petites perturbations)

x0 = hstack((
    position_initiale_chariot,  # Position du chariot
    angles_initiaux * ones(len(q) - 1),  # Angles des pendules
    vitesses_initiales * ones(len(u))  # Vitesses initiales
))

t_span = linspace(0, 10, 1000)  # Temps de simulation : de 0 à 10 secondes
y = odeint(rhs, x0, t_span, args=(parameter_vals,))

# -----------------------------
# Animation
# -----------------------------
def animate_pendulum(t, states, length, title='Pendulum'):
    """
    Fonction d'animation du pendule avec contrôle manuel.
    Design inspiré de tp_env.py avec pygame.
    """
    num_joints = states.shape[1] // 2
    fig = plt.figure(facecolor='#F0F0F5')  # Couleur de fond similaire à pygame
    cart_width, cart_height = 0.4, 0.2
    
    # Définition des limites de la fenêtre
    window_width = 4.0
    xmin = -window_width/2
    xmax = window_width/2
    
    # Configuration des axes avec un style similaire à pygame
    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')
    ax.set_title(title, color='#3C3C46', fontsize=12, pad=20)
    
    # Suppression du cadre et des graduations
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Couleurs inspirées de tp_env.py
    BACKGROUND_COLOR = '#F0F0F5'
    CART_COLOR = '#32323C'
    TRACK_COLOR = '#B4B4BE'
    PENDULUM_COLORS = ['#DC3C3C', '#3CB43C', '#3C3CDC']  # Rouge, Vert, Bleu
    
    # Fond de la figure
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Grille similaire à pygame
    for x in np.arange(xmin, xmax + 0.5, 0.5):
        ax.axvline(x=x, color='#D2D2D7', linestyle='-', alpha=0.3)
    for y in np.arange(-1, 1.1, 0.5):
        ax.axhline(y=y, color='#D2D2D7', linestyle='-', alpha=0.3)
    
    # Piste du chariot
    track = Rectangle([xmin, -cart_height/2 - 0.05], 
                     window_width, 0.05, 
                     facecolor=TRACK_COLOR, edgecolor='none')
    ax.add_patch(track)
    
    # Marqueur central
    ax.plot([0], [-cart_height/2 - 0.025], '|', 
            color='#64646E', markersize=10, markeredgewidth=2)
    
    # Affichage du temps et de la force
    time_display = ax.text(0.04, 0.9, '', transform=ax.transAxes,
                          color='#3C3C46', fontsize=10)
    
    # Chariot avec effet 3D
    cart = Rectangle([states[0, 0] - cart_width/2, -cart_height/2],
                    cart_width, cart_height,
                    facecolor=CART_COLOR, edgecolor='none')
    ax.add_patch(cart)
    
    # Effet 3D sur le chariot
    highlight = Rectangle([states[0, 0] - cart_width/2 + 0.02, -cart_height/2 + 0.02],
                         cart_width - 0.04, cart_height/3,
                         facecolor='#50505A', edgecolor='none')
    ax.add_patch(highlight)
    
    # Ligne du pendule avec marqueurs pour les joints
    pendulum_line, = ax.plot([], [], 
                            color=PENDULUM_COLORS[0], 
                            linewidth=4, 
                            marker='o',
                            markersize=8,
                            markeredgecolor='#1E1E28',
                            markerfacecolor='#5A5A64')
    
    # Variables de contrôle
    applied_force = 0.0
    force_increment = 0.3
    target_force = 0.0
    force_smoothing = 0.1  # Coefficient de lissage

    def handle_key_press(event):
        nonlocal target_force
        if event.key == 'left':
            target_force = -force_increment
        elif event.key == 'right':
            target_force = force_increment
        elif event.key == ' ':
            target_force = 0.0

    def handle_key_release(event):
        nonlocal target_force
        if event.key in ['left', 'right']:
            target_force = 0.0

    fig.canvas.mpl_connect('key_press_event', handle_key_press)
    fig.canvas.mpl_connect('key_release_event', handle_key_release)

    def initialize_animation():
        time_display.set_text('')
        cart.set_xy((0.0, -cart_height/2))
        highlight.set_xy((0.0 - cart_width/2 + 0.02, -cart_height/2 + 0.02))
        pendulum_line.set_data([], [])
        return time_display, cart, highlight, pendulum_line

    def update_animation(frame):
        nonlocal states, applied_force
        if frame < len(t) - 1:
            # Lissage de la force appliquée
            applied_force += force_smoothing * (target_force - applied_force)
            
            current_state = states[frame]
            time_step = t[frame+1] - t[frame]
            
            next_state = current_state + rhs(current_state, t[frame], parameter_vals, lambda x: applied_force) * time_step
            
            cart_position = next_state[0]
            if cart_position - cart_width/2 < xmin:
                next_state[0] = xmin + cart_width/2
                next_state[num_joints] = 0
            elif cart_position + cart_width/2 > xmax:
                next_state[0] = xmax - cart_width/2
                next_state[num_joints] = 0
                
            states[frame+1] = next_state

        time_display.set_text(f'time = {t[frame]:.2f}\nforce = {applied_force:.2f}')
        
        # Mise à jour du chariot
        cart.set_xy((states[frame, 0] - cart_width/2, -cart_height/2))
        highlight.set_xy((states[frame, 0] - cart_width/2 + 0.02, -cart_height/2 + 0.02))
        
        # Calcul des positions des joints du pendule
        x_positions = hstack((states[frame, 0], zeros(num_joints - 1)))
        y_positions = zeros(num_joints)
        for joint in range(1, num_joints):
            x_positions[joint] = x_positions[joint - 1] + length * cos(states[frame, joint])
            y_positions[joint] = y_positions[joint - 1] + length * sin(states[frame, joint])
        
        # Mise à jour de la couleur du pendule en fonction de l'angle
        current_angle = states[frame, 1]
        color_index = min(2, int(abs(current_angle) / (pi/2)))
        pendulum_line.set_color(PENDULUM_COLORS[color_index])
        pendulum_line.set_data(x_positions, y_positions)
        
        return time_display, cart, highlight, pendulum_line

    anim = animation.FuncAnimation(fig, update_animation, frames=len(t),
                                 init_func=initialize_animation, interval=20, blit=True)
    plt.show()

# Instructions pour l'utilisateur
print("Contrôles :")
print("- Flèche gauche : déplacer le chariot vers la gauche")
print("- Flèche droite : déplacer le chariot vers la droite")
print("- Barre d'espace : arrêter le chariot")

animate_pendulum(t_span, y, arm_length, title='Contrôle manuel du pendule')

# -----------------------------
# Controller (LQR)
# -----------------------------
equilibrium_point = hstack((0, pi / 2 * ones(len(q) - 1), zeros(len(u))))
equilibrium_dict = dict(zip(q + u, equilibrium_point))
parameter_dict = dict(zip(parameters, parameter_vals))

f_A_lin, f_B_lin, _ = kane.linearize()
f_A_lin = f_A_lin.subs(parameter_dict).subs(equilibrium_dict)
f_B_lin = f_B_lin.subs(parameter_dict).subs(equilibrium_dict)
m_mat = kane.mass_matrix_full.subs(parameter_dict).subs(equilibrium_dict)

A = matrix(m_mat.inv() * f_A_lin)
B = matrix(m_mat.inv() * f_B_lin)

assert matrix_rank(control.ctrb(A, B)) == A.shape[0]

K, _, _ = control.lqr(A, B, ones(A.shape), 1)

def lqr_controller(x):
    return float(dot(K, equilibrium_point - x))

# Simulate with controller
y_closed = odeint(rhs, x0, t_span, args=(parameter_vals, lqr_controller))

animate_pendulum(t_span, y_closed, arm_length, title='Closed-loop control')
