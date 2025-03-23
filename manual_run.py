import pygame
import sys
import numpy as np
from env import TriplePendulumEnv
from reward import RewardManager

# Create the environment with rendering
env = TriplePendulumEnv(render_mode="human")
reward_manager = RewardManager()

# Reset the environment and get initial observation
obs, info = env.reset()

# Track last direction for more responsive controls
last_direction = 0
# Force magnitude for manual control
force_magnitude = 5.0
# Whether to apply stronger braking when direction changes
strong_braking = True

# Font for instructions
pygame.font.init()
font = pygame.font.Font(None, 24)

# Initialize action with zero force
action = np.array([0.0], dtype=np.float32)

# Run a small loop to see it in action
while True:
    # Get the current state of all keyboard buttons
    keys = pygame.key.get_pressed()
    
    # Get current state for velocity information
    action = action * 0.9
    
    # Check for held keys and apply appropriate force
    if keys[pygame.K_RIGHT]:
        action = np.array([force_magnitude], dtype=np.float32)
        last_direction = -1

    elif keys[pygame.K_LEFT]:
        action = np.array([-force_magnitude], dtype=np.float32)
        last_direction = 1

    # Apply braking when B is held down
    elif keys[pygame.K_b]:
        action = env.apply_brake()
        env.state[1] = 0.0  # Directly zero out velocity
    # Apply full stop when S is held down
    elif keys[pygame.K_s]:
        action = np.array([0.0], dtype=np.float32)
        env.state[1] = 0.0  # Directly zero out velocity
    else:
        # When no key is pressed, reset direction
        last_direction = 0
    
    # Check for other events (like quitting or resetting)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                obs, info = env.reset()
            # Adjust force magnitude with up/down arrows
            elif event.key == pygame.K_UP:
                force_magnitude = min(force_magnitude + 1.0, 20.0)
                print(f"Force magnitude: {force_magnitude}")
            elif event.key == pygame.K_DOWN:
                force_magnitude = max(force_magnitude - 1.0, 1.0)
                print(f"Force magnitude: {force_magnitude}")
            # Toggle momentum stopping when T is pressed
            elif event.key == pygame.K_t:
                strong_braking = not strong_braking
                print(f"Auto-braking on direction change: {'ON' if strong_braking else 'OFF'}")
    
    obs, terminated = env.step(action)
    
    # Calculate and display reward
    reward, upright_reward, x_penalty, non_alignement_penalty = reward_manager.calculate_reward(obs, terminated)
    reward_components = reward_manager.get_reward_components(obs)
    
    # Update the environment's display with reward information
    env.current_reward = reward
    env.reward_components = reward_components
    
    # Render the environment
    env.render()
    
    if terminated:
        obs, info = env.reset()

env.close()
