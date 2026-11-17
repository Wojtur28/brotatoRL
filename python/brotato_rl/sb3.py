"""Stable-Baselines3 integration: crash-tolerant wrapper and a SubprocVecEnv factory.

    from brotato_rl.sb3 import make_sb3_vec_env
    if __name__ == "__main__":                       # required on Windows (spawn)
        venv = make_sb3_vec_env(6, env_kwargs=dict(...), monitor_dir="runs/x/monitor")

`brotato_rl.vector.make_vec_env` builds a gymnasium AsyncVectorEnv, which SB3 does not accept;
this module builds SB3's own SubprocVecEnv instead.

Why the wrapper: a long training run on several games will see instances crash or hang.
BrotatoEnv.step already turns a lost game into truncated=True, and the next reset starts a new
game, but BrotatoEnv.reset gives up after its second attempt and an exception there kills the
SubprocVecEnv worker and with it model.learn(). RobustBrotato retries resets with a backoff,
turns any unexpected step exception into a truncated episode, and makes sure every final info
has the keys Monitor logs (Monitor raises KeyError otherwise).
"""
import functools
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym

from .env import BrotatoEnv, RunConfigError

MONITOR_KEYWORDS = ("wave", "outcome", "kills", "materials_collected", "damage_taken", "crashes")


class RobustBrotato(gym.Wrapper):
    """Keeps a BrotatoEnv usable for days of training.

    reset: retries up to `reset_retries` times (backoff `retry_delay` * attempt seconds) on any
           error except RunConfigError (a bad character/weapon/danger is a config mistake and is
           raised at once). After the last attempt the error is raised.
    step:  an unexpected exception drops the game instance and ends the episode as truncated
           (info["error"]); the next reset starts a fresh game.
    info:  final infos always carry MONITOR_KEYWORDS; info["outcome"] is "died", "won",
           "truncated" (step limit) or "error" (lost game). info["crashes"] counts, for this env
           since it was created, lost games plus failed reset attempts of this wrapper (each of
           which already includes BrotatoEnv's own one retry).
    """

    def __init__(self, env: BrotatoEnv, reset_retries=5, retry_delay=5.0, log=sys.stderr):
        super().__init__(env)
        self.reset_retries = reset_retries
        self.retry_delay = retry_delay
        self.log = log
        self.crashes = 0
        self._last = {}

    def _say(self, text):
        if self.log is not None:
            print(f"[RobustBrotato {time.strftime('%H:%M:%S')}] {text}", file=self.log, flush=True)

    def reset(self, **kwargs):
        for attempt in range(1, self.reset_retries + 1):
            try:
                obs, info = self.env.reset(**kwargs)
                self._last = info
                return obs, info
            except RunConfigError:
                raise
            except Exception as exc:  # noqa: BLE001 - any failure to start a game is retried
                self.crashes += 1
                self._drop()
                if attempt == self.reset_retries:
                    self._say(f"reset failed {attempt} times, giving up: {type(exc).__name__}: {exc}")
                    raise
                self._say(f"reset failed (attempt {attempt}/{self.reset_retries}): {type(exc).__name__}: "
                          f"{str(exc).splitlines()[0] if str(exc) else ''}; retrying")
                time.sleep(self.retry_delay * attempt)

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
        except RuntimeError:
            raise  # step() after the episode ended: a caller bug, not a game crash
        except Exception as exc:  # noqa: BLE001
            self._say(f"step failed, ending the episode: {type(exc).__name__}: {exc}\n"
                      + traceback.format_exc(limit=3))
            self._drop()
            obs = self.env.unwrapped._obs
            reward, terminated, truncated = 0.0, False, True
            info = dict(self._last, error=f"{type(exc).__name__}: {exc}", decisions=[])
            self.crashes += 1
        else:
            if "error" in info:
                self.crashes += 1
                self._say(f"game instance lost: {str(info['error']).splitlines()[0]}")
        if terminated or truncated:
            info.setdefault("outcome", "error" if "error" in info else "truncated")
            for key in MONITOR_KEYWORDS:
                info.setdefault(key, self._last.get(key))
            info["crashes"] = self.crashes
        self._last = info
        return obs, reward, terminated, truncated, info

    def _drop(self):
        env = self.env.unwrapped
        try:
            env._drop_instance()
        except Exception:  # noqa: BLE001 - the instance may already be gone
            env.instance = None
        env._needs_reset = True


def make_brotato_env(rank=0, env_kwargs=None, monitor_dir=None, wrapper_kwargs=None):
    """One monitored, crash-tolerant BrotatoEnv (module-level, so it pickles for spawn)."""
    from stable_baselines3.common.monitor import Monitor

    env = RobustBrotato(BrotatoEnv(**(env_kwargs or {})), **(wrapper_kwargs or {}))
    filename = None
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
        filename = str(Path(monitor_dir) / f"env{rank}")
    return Monitor(env, filename=filename, info_keywords=MONITOR_KEYWORDS)


def make_sb3_vec_env(n_envs, env_kwargs=None, seed=None, monitor_dir=None, wrapper_kwargs=None,
                     subprocess=True):
    """SubprocVecEnv (spawn) of n_envs games, or DummyVecEnv with subprocess=False (debugging).

    env_kwargs go to BrotatoEnv and must be picklable (dataclass configs and meta policies are).
    Monitor CSVs land in monitor_dir/env<rank>.monitor.csv. Call under `if __name__ == "__main__":`.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [functools.partial(make_brotato_env, i, dict(env_kwargs or {}), monitor_dir, dict(wrapper_kwargs or {}))
           for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn") if subprocess else DummyVecEnv(fns)
    if seed is not None:
        venv.seed(seed)
    return venv
