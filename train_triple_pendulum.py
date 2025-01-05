import gym
from stable_baselines3 import PPO
from triple_pendulum_env import TriplePendulumEnv

def main():
    # 1. Create the environment
    env = TriplePendulumEnv(render_mode='human')
    
    # 2. Create the PPO model
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. Train the model for a given number of timesteps
    model.learn(total_timesteps=400_000)

    # 4. Save the trained model
    model.save("ppo_triple_pendulum")

    # 5. (Optional) Test the trained model
    #    If you want to see the environment running, do something like:
    test_env = TriplePendulumEnv(render_mode="human")
    obs, info = test_env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        if done:
            obs, info = test_env.reset()
    test_env.close()

if __name__ == "__main__":
    main()
