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
forces = [(P0, force_cart + weight_cart)]
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

    # Définir la force de poids comme un tuple (Point, Vector)
    weight = -m[i + 1] * g * I.y
    forces.append((Pi, weight))
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
x0 = hstack((0, pi / 2 * ones(len(q) - 1), 1e-3 * ones(len(u))))
t_span = linspace(0, 10, 1000)
y = odeint(rhs, x0, t_span, args=(parameter_vals,))

# -----------------------------
# Animation
# -----------------------------
def animate_pendulum(t, states, length, title='Pendulum'):
    numpoints = states.shape[1] // 2
    fig = plt.figure()
    cart_width, cart_height = 0.4, 0.2
    xmin = around(states[:, 0].min() - cart_width, 1)
    xmax = around(states[:, 0].max() + cart_width, 1)

    ax = plt.axes(xlim=(xmin, xmax), ylim=(-1.1, 1.1), aspect='equal')
    ax.set_title(title)

    time_text = ax.text(0.04, 0.9, '', transform=ax.transAxes)
    rect = Rectangle([states[0, 0] - cart_width / 2.0, -cart_height / 2],
                     cart_width, cart_height, fill=True, color='red', ec='black')
    ax.add_patch(rect)

    line, = ax.plot([], [], lw=2, marker='o', markersize=6)

    def init():
        time_text.set_text('')
        rect.set_xy((0.0, 0.0))
        line.set_data([], [])
        return time_text, rect, line

    def animate(i):
        time_text.set_text(f'time = {t[i]:.2f}')
        rect.set_xy((states[i, 0] - cart_width / 2.0, -cart_height / 2))
        x = hstack((states[i, 0], zeros(numpoints - 1)))
        y = zeros(numpoints)
        for j in range(1, numpoints):
            x[j] = x[j - 1] + length * cos(states[i, j])
            y[j] = y[j - 1] + length * sin(states[i, j])
        line.set_data(x, y)
        return time_text, rect, line

    anim = animation.FuncAnimation(fig, animate, frames=len(t),
                                   init_func=init, interval=20, blit=True)
    plt.show()

animate_pendulum(t_span, y, arm_length, title='Open-loop dynamics')

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
