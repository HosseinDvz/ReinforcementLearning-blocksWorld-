import gymnasium
import blocksworld_env
env = gymnasium.make("blocksworld_env/BlocksWorld-v0", render_mode="human")
observation, info = env.reset()

# do a random action 1000 times
for _ in range(1000):
    action = env.action_space.sample()  # get a random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()