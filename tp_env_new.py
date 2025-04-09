# Importation des bibliothèques nécessaires
import numpy as np
import pygame
from scipy.linalg import solve_continuous_are

# Initialisation de Pygame
pygame.init()

# Paramètres de la fenêtre
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Matrices du système (issues du papier)
A = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -12.4928, -2.0824, -2.2956, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, -67.1071, -65.2564, -71.9704, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, -144.5482, -394.2536, -272.1049, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, -300.4564, -512.8310, -258.9198, 0, 0, 0, 0]
])
B = np.array([[0], [3.651], [0], [10.012], [0], [3.716], [0], [7.720]])

# Poids du LQR (issus du papier)
Q = np.diag([700, 0, 3000, 0, 3000, 0, 3000, 0])
R = np.array([[1]])

# Calcul du gain LQR
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# État initial
state = np.zeros((8, 1))
state[2, 0] = 0.05  # Petit angle initial pour theta1
state[4, 0] = 0.05  # theta2
state[6, 0] = 0.05  # theta3

# Fonction de mise à jour de l'état (Euler explicite pour la simplicité)
def update(state, force, dt):
    u = -K @ state + force
    state_dot = A @ state + B * u
    return state + state_dot * dt

# Fonction d'affichage
def draw(state):
    screen.fill((255, 255, 255))
    
    # Transformation d'échelle pour affichage
    cart_x = WIDTH // 2 + int(state[0, 0] * 200)
    cart_y = HEIGHT // 2

    # Dessin du chariot
    pygame.draw.rect(screen, (0, 0, 0), (cart_x - 50, cart_y - 20, 100, 40))
    
    # Paramètres des pendules
    l1, l2, l3 = 150, 100, 75  # Longueurs graphiques des pendules
    
    # Calcul des positions
    theta1 = state[2, 0]
    theta2 = state[4, 0]
    theta3 = state[6, 0]
    
    p1 = (cart_x + l1 * np.sin(theta1), cart_y - l1 * np.cos(theta1))
    p2 = (p1[0] + l2 * np.sin(theta2), p1[1] - l2 * np.cos(theta2))
    p3 = (p2[0] + l3 * np.sin(theta3), p2[1] - l3 * np.cos(theta3))
    
    # Dessin des pendules
    pygame.draw.line(screen, (255, 0, 0), (cart_x, cart_y), p1, 5)
    pygame.draw.line(screen, (0, 255, 0), p1, p2, 5)
    pygame.draw.line(screen, (0, 0, 255), p2, p3, 5)
    
    pygame.draw.circle(screen, (0, 0, 0), (int(p1[0]), int(p1[1])), 10)
    pygame.draw.circle(screen, (0, 0, 0), (int(p2[0]), int(p2[1])), 10)
    pygame.draw.circle(screen, (0, 0, 0), (int(p3[0]), int(p3[1])), 10)
    
    pygame.display.flip()

# Boucle principale
running = True
while running:
    dt = clock.tick(60) / 1000.0  # 60 FPS
    
    force = 0.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Contrôle clavier
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        force = -50.0
    if keys[pygame.K_RIGHT]:
        force = 50.0
    
    # Mise à jour de l'état
    state = update(state, force, dt)
    
    # Affichage
    draw(state)

pygame.quit()
