"""Several game instances in parallel (one subprocess and one game per environment).

This is the gymnasium AsyncVectorEnv. For Stable-Baselines3 use brotato_rl.sb3.make_sb3_vec_env
(SB3 needs its own VecEnv classes, plus a crash-tolerant wrapper for long training runs).
"""
import functools

import gymnasium as gym


def _make_env(index, kwargs):
    from .env import BrotatoEnv
    return BrotatoEnv(**kwargs)


def make_vec_env(num_envs: int, **env_kwargs) -> gym.vector.AsyncVectorEnv:
    """AsyncVectorEnv with num_envs BrotatoEnv copies. Each one launches its own game.

    Windows needs the spawn start method; run this from under `if __name__ == "__main__":`.
    """
    fns = [functools.partial(_make_env, i, dict(env_kwargs)) for i in range(num_envs)]
    return gym.vector.AsyncVectorEnv(fns, context="spawn", shared_memory=False)
