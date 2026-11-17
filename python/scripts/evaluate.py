"""Measure a bot or a trained model over seeded runs; save every episode to results/<date>_<name>.json.

    python scripts/evaluate.py --name heur_rule --episodes 32 --seed 10000             # HeuristicBot, RuleMetaPolicy
    python scripts/evaluate.py --name heur_random --episodes 32 --seed 10000 --meta random
    python scripts/evaluate.py --name escape0 --bot-param escape_weight=0
    python scripts/evaluate.py --name ppo1 --model runs/ppo1/best/model.zip --episodes 32 --seed 10000

Environment: --spec file (see brotato_rl/spec.py), else the defaults (Well Rounded, Danger 0,
RuleMetaPolicy). With --model, spec.json next to the model (or one folder up) is used, so the
model plays in the environment it was trained in. --character / --danger / --meta ... override
single fields. Evaluation seeds should not overlap training seeds; 10000+ is the convention.

Episode i uses seed --seed + i, so two configurations with the same --seed and --episodes
play the same starting runs (paired comparison; the game is not fully deterministic later on).
Metrics (brotato_rl/evaluation.py): outcome, wave, return, position metrics (corner_frac,
wall_frac, freedom, death_place) and, for won runs, win_by = "timer" (survived until the wave-20
timer ran out) or "boss_kill" (the wave-20 bosses died and the game ended the wave early).
--seeds 10003,10017 replays chosen seeds instead of a range.
"""
import argparse
import json
import multiprocessing as mp
import multiprocessing.util
import time
import warnings
from datetime import datetime
from pathlib import Path

from brotato_rl import BrotatoEnv, paths, spec as specs
from brotato_rl.bots import HeuristicBot, RandomBot
from brotato_rl.evaluation import run_episode, summarize

BOTS = {"heuristic": HeuristicBot, "random": RandomBot, "still": None}
RESULTS_DIR = paths.ROOT / "results"
_worker = {}


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_model(path, algo):
    import importlib
    for module in ("stable_baselines3", "sb3_contrib"):
        try:
            cls = getattr(importlib.import_module(module), algo, None)
        except ImportError:
            continue
        if cls is not None:
            return cls.load(str(path), device="cpu")
    raise ImportError(f"no {algo} in stable_baselines3 / sb3_contrib")


def check_spaces(model, env):
    if model.observation_space != env.observation_space:
        mk = set(getattr(model.observation_space, "spaces", {}) or {})
        ek = set(getattr(env.observation_space, "spaces", {}) or {})
        detail = f"keys only in model: {sorted(mk - ek)}, only in env: {sorted(ek - mk)}" if mk != ek else \
            "same keys, different shapes (EncoderConfig differs from training)"
        raise ValueError(f"model observation space does not match the environment ({detail}); "
                         "evaluate with the spec.json the model was trained with")
    if model.action_space != env.action_space:
        raise ValueError(f"model action space {model.action_space} != env {env.action_space} (action_type)")


def _init(cfg):
    warnings.filterwarnings("ignore", message="Steam is running")
    env = BrotatoEnv(**specs.env_kwargs(cfg["spec"]))
    if cfg["model"]:
        import torch
        torch.set_num_threads(1)  # one game + one network per worker; workers run in parallel
        model = load_model(cfg["model"], cfg["algo"])
        check_spaces(model, env)
        _worker["act"] = lambda obs: model.predict(obs, deterministic=cfg["deterministic"])[0]
    else:
        bot_cls = BOTS[cfg["bot"]]
        bot = bot_cls(**cfg["bot_params"]) if bot_cls else None
        _worker["act"] = bot.act if bot else (lambda obs: 0)
    _worker["env"], _worker["cfg"] = env, cfg
    # Runs when the pool worker exits normally (pool.close + join): quits the game and
    # removes its instance folder.
    mp.util.Finalize(None, env.close, exitpriority=10)


def _episode(seed):
    return run_episode(_worker["env"], _worker["act"], seed, _worker["cfg"]["margin"])


def build_spec(args):
    if args.spec:
        source = Path(args.spec)
    elif args.model:
        source = specs.find_for_model(args.model)
        if source is None:
            print("warning: no spec.json next to the model, using the default spec (may not match training)")
    else:
        source = None
    spec = specs.load(source) if source else specs.normalize()
    for key in ("character", "weapon", "danger", "zone", "start_wave", "frame_skip"):
        value = getattr(args, key)
        if value is not None:
            spec[key] = value
    if args.meta:
        spec["meta"] = {"type": args.meta}
    for item in args.meta_param:
        key, _, value = item.partition("=")
        spec["meta"][key] = parse_value(value)
    for item in args.game_option:
        key, _, value = item.partition("=")
        spec["game_options"][key] = parse_value(value)
    return specs.normalize(spec), (str(source) if source else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="label used in the results file name")
    who = parser.add_argument_group("who plays")
    who.add_argument("--bot", choices=BOTS, default="heuristic")
    who.add_argument("--bot-param", action="append", default=[], metavar="KEY=VALUE",
                     help="bot constructor argument, e.g. escape_weight=0 (repeatable)")
    who.add_argument("--model", help="trained SB3 model (.zip); replaces --bot")
    who.add_argument("--algo", default="PPO", help="SB3 class of the model (PPO, A2C, DQN, ...)")
    who.add_argument("--stochastic", action="store_true", help="sample model actions instead of the argmax")
    env = parser.add_argument_group("environment (override fields of the spec)")
    env.add_argument("--spec", help="run spec JSON (brotato_rl/spec.py)")
    env.add_argument("--meta", choices=("rule", "random", "first", "learned"),
                     help="meta policy type (default: from spec = rule); learned needs --meta-param path=SHOP.pt")
    env.add_argument("--meta-param", action="append", default=[], metavar="KEY=VALUE",
                     help="MetaConfig field, JSON value, e.g. max_rerolls=2 (repeatable)")
    env.add_argument("--game-option", action="append", default=[], metavar="KEY=VALUE",
                     help='mod game option, JSON value, e.g. wave_duration=20 (repeatable)')
    env.add_argument("--character")
    env.add_argument("--weapon")
    env.add_argument("--danger", type=int)
    env.add_argument("--zone", type=int)
    env.add_argument("--start-wave", type=int)
    env.add_argument("--frame-skip", type=int)
    run = parser.add_argument_group("run")
    run.add_argument("--episodes", type=int, default=16)
    run.add_argument("--seed", type=int, default=10000, help="seed of episode 0 (10000+ = evaluation seeds)")
    run.add_argument("--seeds", help="comma-separated list of seeds to play instead of --seed/--episodes")
    run.add_argument("--workers", type=int, default=4, help="parallel game instances")
    run.add_argument("--margin", type=float, default=200.0, help="px from a wall that counts as near it")
    args = parser.parse_args()

    bot_params = {}
    for item in args.bot_param:
        key, _, value = item.partition("=")
        bot_params[key] = float(value)
    spec, spec_source = build_spec(args)
    model = str(Path(args.model).resolve()) if args.model else None
    if model and not Path(model).exists() and not Path(model + ".zip").exists():
        parser.error(f"model not found: {model}")
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.seed, args.seed + args.episodes))
    cfg = {"bot": None if model else args.bot, "bot_params": {} if model else bot_params, "model": model,
           "algo": args.algo, "deterministic": not args.stochastic, "spec": spec, "spec_source": spec_source,
           "margin": args.margin, "meta_policy": spec["meta"]["type"], "seeds": seeds}
    who_plays = f"model {model}" if model else f"bot {args.bot} {bot_params or ''}"
    print(f"{who_plays} | meta {spec['meta']['type']} | {spec['character']}, danger {spec['danger']} | "
          f"{len(seeds)} seeds from {seeds[0]} on {min(args.workers, len(seeds))} workers")

    start = time.perf_counter()
    pool = mp.get_context("spawn").Pool(min(args.workers, len(seeds)), initializer=_init, initargs=(cfg,))
    episodes = []
    try:
        for result in pool.imap_unordered(_episode, seeds):
            episodes.append(result)
            print(f"seed {result['seed']:5d}: {result['outcome']:>6} at wave {result['wave']:2d}"
                  f"{' (' + result['win_by'] + ')' if result['win_by'] else ''} | "
                  f"corner {result['corner_frac']:.0%} | wall {result['wall_frac']:.0%} | "
                  f"freedom {result['freedom']} | death {result['death_place']} | {result['steps']} steps")
        pool.close()  # let workers exit normally so their finalizers close the games
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    episodes.sort(key=lambda e: e["seed"])
    summary = summarize(episodes)
    summary["wall_seconds"] = round(time.perf_counter() - start, 1)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y-%m-%d_%H%M}_{args.name}.json"
    out.write_text(json.dumps({"name": args.name, "created": datetime.now().isoformat(timespec="seconds"),
                               "config": cfg, "summary": summary, "episodes": episodes}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
