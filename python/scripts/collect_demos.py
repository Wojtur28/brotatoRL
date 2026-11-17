"""Record the HeuristicBot playing, as demonstrations for the behaviour cloning warm start.

    python scripts/collect_demos.py --name bot_ppo3 --config python/configs/ppo3.json --episodes 12

The observations are encoded with the spec of the training config (same EncoderConfig, reward
and meta policy), so the network can be trained on them directly. One file per episode:
demos/<name>/ep_<seed>.npz with obs_<key> (float16), actions (int8), rewards, terminated;
demos/<name>/meta.json describes the set. Seeds start at 40000 (not used by evaluations).
"""
import argparse
import json
import multiprocessing as mp
import multiprocessing.util
import time
import warnings
from pathlib import Path

import numpy as np

from brotato_rl import BrotatoEnv, paths, spec as specs
from brotato_rl.bots import HeuristicBot

DEMOS_DIR = paths.ROOT / "demos"
_worker = {}


def _init(spec, out_dir, bot_params):
    warnings.filterwarnings("ignore", message="Steam is running")
    env = BrotatoEnv(**specs.env_kwargs(spec))
    _worker.update(env=env, bot=HeuristicBot(**bot_params), out=Path(out_dir))
    mp.util.Finalize(None, env.close, exitpriority=10)


def _episode(seed):
    env, bot, out = _worker["env"], _worker["bot"], _worker["out"]
    obs, info = env.reset(seed=seed)
    keys = list(obs)
    buf = {k: [] for k in keys}
    actions, rewards, terminated = [], [], False
    while True:
        action = bot.act(obs)
        for k in keys:
            buf[k].append(obs[k].astype(np.float16))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    np.savez_compressed(out / f"ep_{seed}.npz", actions=np.asarray(actions, dtype=np.int8),
                        rewards=np.asarray(rewards, dtype=np.float32), terminated=np.bool_(terminated),
                        **{f"obs_{k}": np.stack(v) for k, v in buf.items()})
    return {"seed": seed, "steps": len(actions), "wave": info.get("wave"), "outcome": info.get("outcome"),
            "return": float(np.sum(rewards))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="folder under demos/")
    parser.add_argument("--config", required=True, help="training config whose spec encodes the observations")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=2, help="games in parallel (~0.5 GB RAM each)")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    spec = specs.normalize(cfg["spec"])
    out = DEMOS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(args.seed, args.seed + args.episodes) if not (out / f"ep_{s}.npz").is_file()]
    print(f"{len(seeds)} episodes to record into {out} on {args.workers} games")
    start = time.time()
    results = []
    if seeds:
        pool = mp.get_context("spawn").Pool(min(args.workers, len(seeds)), initializer=_init, initargs=(spec, str(out), {}))
        try:
            for r in pool.imap_unordered(_episode, seeds):
                results.append(r)
                print(f"seed {r['seed']}: {r['outcome']} at wave {r['wave']}, {r['steps']} steps", flush=True)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    meta_file = out / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {"episodes": []}
    meta.update(spec=spec, bot="HeuristicBot()", config=str(args.config))
    meta["episodes"] = sorted(meta["episodes"] + results, key=lambda r: r["seed"])
    meta["total_steps"] = int(sum(r["steps"] for r in meta["episodes"]))
    meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    waves = [r["wave"] or 0 for r in meta["episodes"]]
    print(f"{len(waves)} episodes, {meta['total_steps']} steps, mean wave {np.mean(waves):.2f} "
          f"({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
