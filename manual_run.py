import pygame
import sys
import numpy as np
from tp_env_v1 import TriplePendulumEnv
from reward import RewardManager
from config import config

# Create the environment with rendering
reward_manager = RewardManager()
env = TriplePendulumEnv(render_mode="human", reward_manager=reward_manager, num_nodes=config['num_nodes'])

# Reset the environment and get initial observation
obs = env.reset()

# Track last direction for more responsive controls
last_direction = 0
# Force magnitude for manual control
force_magnitude = 0.1
# Whether to apply stronger braking when direction changes
strong_braking = True

# Font for instructions
pygame.font.init()
font = pygame.font.Font(None, 24)

# Initialize action with zero force
action = np.array([0.0], dtype=np.float32)

# Initialize clock for controlling frame rate
clock = pygame.time.Clock()

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
        env.state_for_simu[1] = 0.0  # Directly zero out velocity
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
                reward_manager.reset()  # Reset reward manager state
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
            elif event.key == pygame.K_s: # print state
                print(env.state_for_simu)  # Print the observation instead of internal state
                print(env.get_rich_state(env.state_for_simu))
    
    obs, terminated = env.step(action)
    rich_obs = env.get_rich_state(obs)
    
    # Calculate and display reward - only calculate once
    reward_components = reward_manager.get_reward_components(rich_obs, 0)
    reward = reward_components['reward']
    
    # Update the environment's display with reward information
    env.current_reward = reward
    env.reward_components = reward_components
    
    # Render the environment
    env.render()
    
    # Control frame rate
    clock.tick(30)
    
    if terminated:
        obs, info = env.reset()
        reward_manager.reset()  # Reset reward manager state

env.close()
