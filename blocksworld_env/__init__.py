from gymnasium.envs.registration import register

register(
    id="blocksworld_env/BlocksWorld-v0",
    entry_point="blocksworld_env.envs:BlocksWorldEnv",
)
