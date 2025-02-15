import gymnasium as gym
import blocksworld_env
from stable_baselines3 import DQN

env = gym.make("blocksworld_env/BlocksWorld-v0", render_mode="human")
observation, info = env.reset() 

model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

for _ in range(10000):
    #action = env.action_space.sample()  # agent policy that uses the observation and info
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()