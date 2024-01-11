from lop.envs.slippery_ant import SlipperyAntEnv, SlipperyAntEnv3


from gym.envs.registration import (
    register,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()


register(
    id="SlipperyAnt-v2",
    entry_point="lop.envs.slippery_ant:SlipperyAntEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id="SlipperyAnt-v3",
    entry_point="lop.envs.slippery_ant:SlipperyAntEnv3",
    max_episode_steps=1000,
    reward_threshold=6000.0,
)