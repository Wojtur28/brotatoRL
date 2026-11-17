"""Offline test: how well can a policy network imitate HeuristicBot? No game needed.

    python scripts/bc_arch_test.py --config python/configs/ppo4.json --epochs 30
    python scripts/bc_arch_test.py --config python/configs/ppo4.json --archs mlp256 set64 --epochs 10

For each architecture a fresh policy network (the same SB3 policy class PPO uses) learns the bot's
actions from the demonstrations of train.bc.demos (cross-entropy, policy head only). Whole episodes
are held out: neighbouring frames are nearly identical, so holding out random frames would mostly
test memory. Reported per epoch on the held-out episodes: exact action, the neighbouring direction
(actions 1-8 go round the circle) and "wrong way" (3 or more of the 8 directions away).
Results: results/<date>_bc_arch.json.
"""
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch as th

from brotato_rl import BrotatoEnv, paths, spec as specs
from brotato_rl.bc import load_demos
from brotato_rl.extractors import policy_kwargs_for

ARCHS = {
    "mlp256": {"net_arch": [256, 256], "extractor": None},              # ppo3 / ppo4
    "mlp512": {"net_arch": [512, 512, 256], "extractor": None},         # more capacity, same input
    "set64": {"net_arch": [256, 256], "extractor": {"type": "entity_set", "hidden": 64, "out": 64}},
    "set128": {"net_arch": [256, 256], "extractor": {"type": "entity_set", "hidden": 128, "out": 128}},
}


def scores(pred, target):
    both = (pred > 0) & (target > 0)
    d = np.abs(pred - target) % 8
    steps = np.minimum(d, 8 - d)
    return {"exact": float(np.mean(pred == target)), "neighbour": float(np.mean(both & (steps == 1))),
            "wrong_way": float(np.mean(both & (steps >= 3)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training config: its spec gives the spaces, train.bc.demos the data")
    parser.add_argument("--demos", help="override train.bc.demos")
    parser.add_argument("--archs", nargs="+", default=["mlp256", "mlp512", "set64"], choices=ARCHS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--holdout-episodes", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, help="train on only the first N episodes (learning curve)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    th.set_num_threads(args.threads)
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy

    cfg = json.loads(open(args.config, encoding="utf-8").read())
    env = BrotatoEnv(**specs.env_kwargs(specs.normalize(cfg["spec"])))  # spaces only, no game is started
    demos = args.demos or cfg["train"]["bc"]["demos"]
    obs, actions, _, _, n_eps = load_demos(demos)
    lengths = [len(a) for a in actions]
    acts = np.concatenate(actions)
    split = sum(lengths[:-args.holdout_episodes])
    train_end = sum(lengths[:args.train_episodes]) if args.train_episodes else split
    train_idx, hold_idx = np.arange(min(train_end, split)), np.arange(split, len(acts))
    print(f"{demos}: {n_eps} episodes, {len(acts)} steps; held out the last {args.holdout_episodes} episodes "
          f"({len(hold_idx)} steps)")

    def batch(idx):
        return {k: th.as_tensor(v[idx].astype(np.float32)) for k, v in obs.items()}

    results = {"config": args.config, "demos": str(demos), "train_steps": int(len(train_idx)),
               "holdout_steps": int(len(hold_idx)), "epochs": args.epochs, "archs": {}}
    for name in args.archs:
        arch = ARCHS[name]
        th.manual_seed(0)
        policy = MultiInputActorCriticPolicy(
            env.observation_space, env.action_space, lr_schedule=lambda _: args.learning_rate,
            net_arch={"pi": arch["net_arch"], "vf": [64]}, activation_fn=th.nn.Tanh,
            **policy_kwargs_for(arch["extractor"]))
        params = [p for n, p in policy.named_parameters() if not n.startswith(("value_net", "mlp_extractor.value_net"))]
        optimizer = th.optim.Adam(params, lr=args.learning_rate)
        n_params = sum(p.numel() for p in params)
        rng = np.random.default_rng(0)
        history = []
        print(f"== {name}: {n_params:,} parameters (feature extractor + policy head)")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            policy.set_training_mode(True)
            order = rng.permutation(train_idx)
            losses = []
            for i in range(0, len(order), args.batch_size):
                idx = np.sort(order[i:i + args.batch_size])
                logits = policy.get_distribution(batch(idx)).distribution.logits
                loss = th.nn.functional.cross_entropy(logits, th.as_tensor(acts[idx], dtype=th.long))
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(params, 0.5)
                optimizer.step()
                losses.append(float(loss))
            policy.set_training_mode(False)
            with th.no_grad():
                pred = np.concatenate([policy.get_distribution(batch(hold_idx[i:i + 4096])).distribution.logits
                                       .argmax(dim=1).numpy() for i in range(0, len(hold_idx), 4096)])
            s = scores(pred, acts[hold_idx])
            history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                            **{k: round(v, 4) for k, v in s.items()}, "seconds": round(time.time() - start, 1)})
            if epoch in (1, 2, 5) or epoch % 5 == 0:
                print(f"  epoch {epoch:3d}: loss {history[-1]['train_loss']:.3f}, held out: exact {s['exact']:.1%}, "
                      f"neighbour {s['neighbour']:.1%}, wrong way {s['wrong_way']:.1%} ({history[-1]['seconds']}s)",
                      flush=True)
        best = max(history, key=lambda h: h["exact"])
        results["archs"][name] = {"params": n_params, "arch": arch, "best_epoch": best, "last": history[-1],
                                  "history": history}
    out = paths.ROOT / "results" / f"{datetime.now():%Y-%m-%d_%H%M}_bc_arch.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("\n" + "\n".join(f"{n}: best exact {r['best_epoch']['exact']:.1%} (epoch {r['best_epoch']['epoch']}), "
                           f"last: exact {r['last']['exact']:.1%}, wrong way {r['last']['wrong_way']:.1%}"
                           for n, r in results["archs"].items()))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
