import gym
from triple_pendulum_env import TriplePendulumEnv

# Create the environment with rendering
env = TriplePendulumEnv(render_mode="human")

# Reset the environment and get initial observation
obs, info = env.reset()

# Run a small loop to see it in action
for _ in range(200):
    action = env.action_space.sample()  # random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs, info = env.reset()

env.close()
