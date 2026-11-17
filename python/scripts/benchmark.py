"""Throughput of N game instances in parallel, driven by the heuristic bot.

    python scripts/benchmark.py --envs 4 --steps 4000
"""
import argparse
import time
import warnings

import numpy as np

from brotato_rl.bots import HeuristicBot
from brotato_rl.vector import make_vec_env


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000, help="vector steps (each one steps every env)")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--danger", type=int, default=0)
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Steam is running")

    start = time.perf_counter()
    envs = make_vec_env(args.envs, frame_skip=args.frame_skip, danger=args.danger)
    obs, _ = envs.reset(seed=0)
    boot = time.perf_counter() - start
    bots = [HeuristicBot() for _ in range(args.envs)]
    episodes, waves, errors = 0, [], 0
    start = time.perf_counter()
    try:
        for _ in range(args.steps):
            actions = np.array([bot.act({k: v[i] for k, v in obs.items()}) for i, bot in enumerate(bots)])
            obs, _, terminated, truncated, info = envs.step(actions)
            for i in np.flatnonzero(terminated | truncated):
                episodes += 1
                waves.append(int(info["wave"][i]))
                errors += int("error" in info and bool(info["_error"][i]))
    finally:
        elapsed = time.perf_counter() - start
        envs.close()
    total = args.steps * args.envs
    game_seconds = total * args.frame_skip / 60
    print(f"{args.envs} envs: startup {boot:.1f}s, {total} steps in {elapsed:.1f}s = {total / elapsed:.0f} steps/s "
          f"({total / elapsed / args.envs:.0f} per env), {game_seconds / elapsed:.0f}x real time overall")
    print(f"finished episodes: {episodes}, waves reached: {waves}, crashed/hung instances: {errors}")


if __name__ == "__main__":
    main()
