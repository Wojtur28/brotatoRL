"""Record one run of a model (or bot) to an MP4, plus a replay file next to it.

    python scripts/record_video.py --run ppo1 --at 2.5M --seed 30000
    python scripts/record_video.py --run ppo1 --at best --speed 2 --max-minutes 5 --resolution 1920x1080
    python scripts/record_video.py --bot heuristic --seed 30000 --out showcase/videos/bot.mp4
    python scripts/record_video.py --bot heuristic --spec runs/ppo3/spec.json --bot-param escape_weight=0

The game window opens always on top at --position and must stay visible while recording
(ffmpeg captures that part of the screen). --speed 2 records a 2x time-lapse. Audio is not
recorded. The environment (observations, meta policy) comes from the model's spec.json.
Default output: showcase/videos/<run>_<steps>_seed<seed>.mp4 (+ .json.gz replay).
"""
import argparse
import time
import warnings
from pathlib import Path

from brotato_rl import BrotatoEnv, paths, runs, spec as specs
from brotato_rl.bots import HeuristicBot, RandomBot
from brotato_rl.policy import actor, check_spaces, load_sb3
from brotato_rl.replay import ReplayRecorder
from brotato_rl.video import WindowRecorder, wait_for_window

BOTS = {"heuristic": HeuristicBot, "random": RandomBot, "still": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--run", help="training run under runs/")
    who.add_argument("--model", help="model .zip (spec.json next to it or one folder up)")
    who.add_argument("--bot", choices=BOTS)
    parser.add_argument("--bot-param", action="append", default=[], metavar="KEY=VALUE",
                        help="HeuristicBot parameter, e.g. escape_weight=0 (the bot before the corner fix)")
    parser.add_argument("--spec", help="run spec JSON for a bot (default: the default spec; a model uses its spec.json)")
    parser.add_argument("--at", default="best", help="with --run: best | final | bc | steps (2.5M, 750k)")
    parser.add_argument("--stochastic", action="store_true", help="sample model actions instead of the argmax")
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--out", help="output .mp4 (default under showcase/videos/)")
    parser.add_argument("--label", help="caption burnt into the video (default: model and seed)")
    parser.add_argument("--speed", type=float, default=1.0, help="game tempo while recording (2 = 2x time-lapse)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--position", default="40,40", help="window top-left corner x,y")
    parser.add_argument("--max-minutes", type=float, default=0, help="stop the recording after this many minutes")
    parser.add_argument("--lite", action="store_true", help="no screenshake / effects / damage numbers")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Steam is running")

    steps = None
    if args.run:
        path, steps = runs.resolve(args.run, args.at)
        model_path, name = path, runs.label(args.run, steps)
    else:
        model_path = Path(args.model) if args.model else None
        name = model_path.stem if model_path else f"bot {args.bot}"
    spec_path = Path(args.spec) if args.spec else (specs.find_for_model(model_path) if model_path else None)
    spec = specs.load(spec_path) if spec_path else specs.normalize()
    width, height = (int(v) for v in args.resolution.lower().split("x"))
    x, y = (int(v) for v in args.position.split(","))
    env = BrotatoEnv(**specs.env_kwargs(spec), render_mode="human", speed=args.speed, resolution=(width, height),
                     window_position=(x, y), lite_visuals=args.lite, game_args=["--always-on-top"])
    if model_path:
        model = load_sb3(model_path, threads=2)
        check_spaces(model, env)
        act = actor(model, deterministic=not args.stochastic)
    else:
        params = {k: float(v) for k, v in (item.split("=", 1) for item in args.bot_param)}
        bot = BOTS[args.bot](**params) if BOTS[args.bot] else None
        act = bot.act if bot else (lambda obs: 0)

    slug = name.replace(" @ ", "_").replace(" ", "_")
    out = Path(args.out) if args.out else paths.ROOT / "showcase" / "videos" / f"{slug}_seed{args.seed}.mp4"
    label = args.label or f"{name} | seed {args.seed}" + (f" | {args.speed:g}x" if args.speed != 1 else "")
    replay = ReplayRecorder({"title": name, "model": str(model_path) if model_path else args.bot,
                             "steps_trained": steps, "seed": args.seed, "video": out.name})
    video = None
    info = {}
    try:
        obs, info = env.reset(seed=args.seed)  # the game now waits for the first action
        replay.add(env.raw_state, None, info)
        video = WindowRecorder(wait_for_window(env.instance.proc.pid), out, fps=args.fps, label=label)
        time.sleep(0.5)  # ffmpeg start-up
        start = time.time()
        while True:
            action = act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            replay.add(env.raw_state, action, info)
            if terminated or truncated:
                time.sleep(1.5)  # keep the end screen in the video
                break
            if args.max_minutes and time.time() - start > args.max_minutes * 60:
                info = dict(info, outcome="stopped")
                break
    except KeyboardInterrupt:
        info = dict(info, outcome="stopped")
    finally:
        if video is not None:
            print(f"video saved: {video.stop()}")
        env.close()
    replay.finish(info, outcome=info.get("outcome") or "truncated")
    print(f"replay saved: {replay.save(out.with_suffix('.json.gz'))}")
    print(f"{name}: {info.get('outcome')} at wave {info.get('wave')}")


if __name__ == "__main__":
    main()
