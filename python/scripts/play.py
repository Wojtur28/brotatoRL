"""Play Brotato with a baseline bot or a trained model, optionally in a game window.

    python scripts/play.py --bot heuristic --episodes 5                      # headless, fast
    python scripts/play.py --bot heuristic --watch                           # game window, real time
    python scripts/play.py --model models/ppo1_v1/model.zip --watch          # trained model in a window
    python scripts/play.py --model runs/ppo1/best/model.zip --watch --speed 2 --sound --seed 10027
    python scripts/play.py --run ppo1 --at 2.5M --watch                      # earlier iteration (checkpoint)
    python scripts/play.py --run ppo1 --at 1M --episodes 1 --seed 30000 --replay-dir showcase/replays
    python scripts/play.py --run ppo3 --watch --resolution 960x540 --position 0,0   # one of several windows

--run/--at pick a model of a training run: best (default), final, bc or a step count (the
nearest checkpoint). --replay-dir saves a replay of every episode (open it in viewer/replay.html).
With --model the environment comes from spec.json next to the model (or one folder up), so the
model sees exactly the observations it was trained on, with the same meta policy. --character /
--danger override those fields (the model was trained on one setup; others are an experiment).
The window keeps all visual effects (good for recording); add --lite to switch them off.
"""
import argparse
import time
from pathlib import Path

from brotato_rl import BrotatoEnv, runs, spec as specs
from brotato_rl.bots import HeuristicBot, RandomBot
from brotato_rl.replay import ReplayRecorder

BOTS = {"heuristic": HeuristicBot, "random": RandomBot, "still": None}


def run_episode(env, act, seed, recorder=None):
    obs, info = env.reset(seed=seed)
    if recorder is not None:
        recorder.add(env.raw_state, None, info)
    total, steps, start = 0.0, 0, time.perf_counter()
    decisions = {}
    while True:
        action = act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.add(env.raw_state, action, info)
        total += reward
        steps += 1
        for decision in info["decisions"]:
            decisions[decision["kind"]] = decisions.get(decision["kind"], 0) + 1
        if terminated or truncated:
            break
    if recorder is not None:
        recorder.finish(info, outcome=info.get("outcome") or "truncated", seed=seed, episode_return=round(total, 3))
    return {
        "seed": seed,
        "outcome": info.get("outcome") or ("error: " + info["error"] if "error" in info else "truncated"),
        "wave": info["wave"],
        "return": total,
        "steps": steps,
        "kills": info["kills"],
        "materials": info["materials_collected"],
        "damage_taken": info["damage_taken"],
        "decisions": decisions,
        "steps_per_s": steps / (time.perf_counter() - start),
    }


def model_player(path, algo, deterministic, env):
    import importlib

    import torch
    torch.set_num_threads(2)
    cls = None
    for module in ("stable_baselines3", "sb3_contrib"):
        try:
            cls = getattr(importlib.import_module(module), algo, None)
        except ImportError:
            continue
        if cls is not None:
            break
    if cls is None:
        raise SystemExit(f"no {algo} in stable_baselines3 / sb3_contrib")
    model = cls.load(str(path), device="cpu")
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise SystemExit("the model's observation/action space does not match the environment; "
                         "keep spec.json next to the model (or pass --spec)")
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bot", choices=BOTS, default="heuristic")
    parser.add_argument("--bot-param", action="append", default=[], metavar="KEY=VALUE",
                        help="HeuristicBot parameter, e.g. escape_weight=0 (the bot before the corner fix)")
    parser.add_argument("--model", help="trained SB3 model (.zip); replaces --bot")
    parser.add_argument("--run", help="training run under runs/ (instead of --model)")
    parser.add_argument("--at", default="best", help="with --run: best | final | bc | steps, e.g. 2.5M or 750k")
    parser.add_argument("--replay-dir", help="save a replay of every episode here (viewer/replay.html)")
    parser.add_argument("--algo", default="PPO")
    parser.add_argument("--stochastic", action="store_true", help="sample model actions instead of the argmax")
    parser.add_argument("--spec", help="run spec JSON (default: spec.json next to the model)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="seed of the first episode (then +1)")
    parser.add_argument("--character", help="default: from the spec (Well Rounded)")
    parser.add_argument("--weapon", help="starting weapon id (default: first choice)")
    parser.add_argument("--danger", type=int)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--watch", action="store_true", help="show the game window")
    parser.add_argument("--speed", type=float, default=1.0, help="with --watch: tempo factor (1 = real time, 2 = 2x)")
    parser.add_argument("--sound", action="store_true", help="game sound and music (for recordings)")
    parser.add_argument("--lite", action="store_true", help="no screenshake / effects / damage numbers")
    parser.add_argument("--resolution", default="1280x720", help="window size, e.g. 1920x1080")
    parser.add_argument("--position", help="with --watch: window top-left corner x,y (several windows side by side)")
    args = parser.parse_args()
    trained_steps = None
    if args.run:
        path, trained_steps = runs.resolve(args.run, args.at)
        args.model = str(path)
        print(f"{runs.label(args.run, trained_steps)}: {path}")

    spec_path = Path(args.spec) if args.spec else (specs.find_for_model(args.model) if args.model else None)
    if args.model and spec_path is None:
        print("warning: no spec.json next to the model, using the default spec (may not match training)")
    spec = specs.load(spec_path) if spec_path else specs.normalize()  # default meta policy: rule
    for key in ("character", "weapon", "danger", "frame_skip"):
        if getattr(args, key) is not None:
            spec[key] = getattr(args, key)
    width, height = (int(x) for x in args.resolution.lower().split("x"))
    position = tuple(int(v) for v in args.position.split(",")) if args.position else None
    env = BrotatoEnv(**specs.env_kwargs(spec), render_mode="human" if args.watch else None, speed=args.speed,
                     sound=args.sound, lite_visuals=args.lite if args.watch else None, resolution=(width, height),
                     window_position=position)
    if args.model:
        act, who = model_player(Path(args.model), args.algo, not args.stochastic, env), f"model {args.model}"
    else:
        bot_cls = BOTS[args.bot]
        params = {k: float(v) for k, v in (item.split("=", 1) for item in args.bot_param)}
        bot = bot_cls(**params) if bot_cls else None
        act, who = (bot.act if bot else (lambda obs: 0)), args.bot
    print(f"{who} | meta {spec['meta']['type']} | {spec['character']}, danger {spec['danger']}"
          + (f" | spec {spec_path}" if spec_path else ""))
    results = []
    try:
        for i in range(args.episodes):
            seed = args.seed + i
            recorder = None
            if args.replay_dir:
                title = runs.label(args.run, trained_steps) if args.run else (Path(args.model).stem if args.model else args.bot)
                recorder = ReplayRecorder({"title": title, "model": args.model, "steps_trained": trained_steps,
                                           "seed": seed, "character": spec["character"], "danger": spec["danger"]})
            result = run_episode(env, act, seed, recorder)
            if recorder is not None:
                slug = (runs.label(args.run, trained_steps) if args.run else title).replace(" @ ", "_").replace(" ", "_")
                saved = recorder.save(Path(args.replay_dir) / f"{slug}_seed{seed}.json.gz")
                print(f"replay saved: {saved}")
            results.append(result)
            print(f"episode {i + 1} (seed {result['seed']}): {result['outcome']:>6} at wave {result['wave']:2d} | "
                  f"return {result['return']:7.2f} | {result['steps']:6d} steps ({result['steps_per_s']:.0f}/s) | "
                  f"kills {result['kills']} | materials {result['materials']} | damage taken {result['damage_taken']}"
                  f" | decisions {result['decisions']}", flush=True)
    except KeyboardInterrupt:
        print("stopped")
    finally:
        env.close()
    if results:
        waves = [r["wave"] for r in results]
        print(f"\n{who}: mean wave {sum(waves) / len(waves):.1f}, best {max(waves)}, "
              f"wins {sum(r['outcome'] == 'won' for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
