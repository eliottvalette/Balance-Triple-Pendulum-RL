import pygame
import sys
import numpy as np
from tp_env import TriplePendulumEnv
from config import config

# Créer l'environnement avec affichage
env = TriplePendulumEnv(render_mode="human")

# Réinitialiser l'environnement et obtenir l'observation initiale
obs, info = env.reset()

# Initialiser l'action avec une force nulle
action = np.array([0.0], dtype=np.float32)

# Initialiser l'horloge pour contrôler le framerate
clock = pygame.time.Clock()

while True:
    # Récupérer l'état des touches enfoncées
    keys = pygame.key.get_pressed()
    action = action * 0.9
    
    # Appliquer la force en fonction des touches
    if keys[pygame.K_RIGHT]:
        action = np.array([10.0], dtype=np.float32)
    elif keys[pygame.K_LEFT]:
        action = np.array([-10.0], dtype=np.float32)
    
    # Gestion des événements
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                obs, info = env.reset()
            elif event.key == pygame.K_UP:
                print("Augmentation de la force (non implémentée dans ce demo)")
            elif event.key == pygame.K_DOWN:
                print("Diminution de la force (non implémentée dans ce demo)")
            elif event.key == pygame.K_s:
                print("État actuel :", env.state)
    
    obs, done = env.step(action)
    env.render()
    
    if done:
        obs, info = env.reset()
    
    clock.tick(30)

env.close()
