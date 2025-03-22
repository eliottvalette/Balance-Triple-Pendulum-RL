import pygame
import sys
import numpy as np
from env import TriplePendulumEnv

# Create the environment with rendering
env = TriplePendulumEnv(render_mode="human", test_mode=True)

# Reset the environment and get initial observation
obs, info = env.reset()

# Run a small loop to see it in action
while True:
    # Get the current state of all keyboard buttons
    keys = pygame.key.get_pressed()
    
    # Initialize action with zero force
    action = np.array([0.0], dtype=np.float32)
    
    # Check for held keys
    if keys[pygame.K_RIGHT]:
        action = np.array([-5.0], dtype=np.float32)
    elif keys[pygame.K_LEFT]:
        action = np.array([5.0], dtype=np.float32)
    
    # Check for other events (like quitting or resetting)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                obs, info = env.reset()
    
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()
