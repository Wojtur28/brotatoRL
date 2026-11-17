"""End-to-end checks of the environment (about two minutes).

    python scripts/smoke_test.py

Exit code 0 when every check passes.
"""
import hashlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from brotato_rl import BrotatoEnv, RunConfigError, paths
from brotato_rl.bots import HeuristicBot

REAL_SAVES = Path(os.environ.get("APPDATA", "")) / "Brotato"
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'OK' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def snapshot(folder: Path) -> dict:
    if not folder.is_dir():
        return {}
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def main():
    warnings.filterwarnings("ignore", message="Steam is running")
    saves_before = snapshot(REAL_SAVES)
    env = BrotatoEnv(include_raw=True)
    try:
        obs, info = env.reset(seed=7)
        user_dir = Path(env.hello["user_dir"])
        check("instance uses its own lab user folder", paths.INSTANCES_DIR.resolve() in user_dir.resolve().parents,
              str(user_dir))
        check("first observation fits the observation space", env.observation_space.contains(obs))

        # Lockstep: every step advances exactly frame_skip physics frames, and the action
        # moves the character for exactly those frames.
        frames, xs, rewards = [env.raw_state["frame"]], [env.raw_state["player"]["pos"][0]], []
        for _ in range(10):
            obs, reward, *_ = env.step(1)  # east
            frames.append(env.raw_state["frame"])
            xs.append(env.raw_state["player"]["pos"][0])
            rewards.append(reward)
        frame_deltas = set(np.diff(frames).tolist())
        expected_dx = env.raw_state["player"]["move_speed"] * env.frame_skip / 60
        dx = np.diff(xs)
        check("each step advances exactly frame_skip frames", frame_deltas == {env.frame_skip}, f"deltas {frame_deltas}")
        check("movement per step equals speed * frame_skip / 60", np.allclose(dx, expected_dx, atol=0.5),
              f"dx {dx[:3].round(2).tolist()}..., expected {expected_dx:.2f}")
        check("rewards are finite", all(np.isfinite(rewards)))

        obs, info = env.reset(seed=8)
        check("reset in the middle of a run starts a new run", info["wave"] == 1 and info["frame"] <= env.frame_skip,
              f"wave {info['wave']}, frame {info['frame']}")

        try:
            env.reset(options={"character": "character_does_not_exist"})
            check("unknown character is rejected", False)
        except RunConfigError as exc:
            check("unknown character is rejected", True, str(exc))
        obs, info = env.reset(seed=9)
        check("instance still usable after a rejected reset", info["wave"] == 1)

        # Wave transition with decisions (invulnerable so the bot surely reaches wave 2).
        obs, info = env.reset(seed=10, options={"invulnerable": True})
        bot, decisions = HeuristicBot(), []
        for _ in range(2000):
            obs, reward, terminated, truncated, info = env.step(bot.act(obs))
            decisions += info["decisions"]
            if info["wave_changed"] or terminated or truncated:
                break
        kinds = sorted({d["kind"] for d in decisions})
        check("wave 1 -> 2 transition, shop and level-up handled", info["wave"] == 2 and "shop" in kinds, f"decisions {kinds}")

        # Death: standing still at wave 5 is fatal quickly.
        obs, info = env.reset(seed=11, options={"start_wave": 5})
        terminated = False
        for steps in range(1, 5000):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
        check("death ends the episode (terminated, outcome=died)", terminated and info.get("outcome") == "died",
              f"after {steps} steps, reward parts {info.get('reward_parts')}")
        try:
            env.step(0)
            check("step after the end requires reset", False)
        except RuntimeError:
            check("step after the end requires reset", True)
    finally:
        env.close()

    checker_env = BrotatoEnv()
    try:
        check_env(checker_env, skip_render_check=True)
        check("gymnasium check_env", True)
    except Exception as exc:  # noqa: BLE001 - report any checker failure
        check("gymnasium check_env", False, f"{type(exc).__name__}: {exc}")
    finally:
        checker_env.close()

    check("real saves in %APPDATA%\\Brotato untouched", snapshot(REAL_SAVES) == saves_before,
          f"{len(saves_before)} files compared")
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
