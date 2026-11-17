"""Train a PPO movement agent with Stable-Baselines3.

    python python\\scripts\\train.py --name ppo1                                  # python\\configs\\ppo_baseline.json
    python python\\scripts\\train.py --name smoke --set train.total_timesteps=20000 --set train.n_envs=2
    python python\\scripts\\train.py --name ppo1 --resume                          # continue from the last checkpoint
    tensorboard --logdir runs                                                     # http://localhost:6006

Everything goes to runs/<name>/:
  spec.json          environment of the model (evaluate.py / play.py / the panel read it)
  config.json        the full config used (spec + ppo + train)
  progress.csv, events.out.tfevents.*   SB3 logs: rollout/mean_wave, rollout/win_rate, eval/... too
  monitor/           one CSV per env: a row per episode (wave, outcome, kills, crashes ...)
  checkpoints/       model_<steps>_steps.zip + vecnormalize_<steps>_steps.pkl every train.checkpoint_every
  best/model.zip     best model by evaluation mean wave (best/eval.json: its evaluation)
  evals.jsonl        every evaluation (summary per line)
  final/model.zip    model at the end (or interrupted/ after Ctrl+C)
  progress.partN.csv the log of earlier sessions (--resume starts a new progress.csv)
  monitor.partN/     the episodes of earlier sessions (--resume starts a new monitor/)

Evaluation every train.eval_every steps: train.eval_episodes runs on fixed seeds (train.eval_seed+i)
on train.eval_envs extra games, deterministic actions (plus sampled actions with
train.eval_stochastic). best/ is chosen on these runs, so they are VALIDATION seeds: use 20000+
here and keep 10000-10031 (the HeuristicBot baseline seeds) for the final test with evaluate.py.
Otherwise the reported result is biased by the selection. The extra games exist only during the
evaluation (~0.5 GB RAM each).

ppo.lr_schedule: "linear" (old: linear to 0 at total_timesteps, depends on total_timesteps) or
{"type": "linear" | "cosine" | "constant", "start": 3e-4, "end": 6e-5, "horizon": 15000000}:
computed from the absolute step count, so --resume with a longer run continues smoothly.

train.bc: optional behaviour cloning warm start from demonstrations (scripts/collect_demos.py),
see brotato_rl/bc.py. Runs once, before PPO, on a fresh run only.

ppo.extractor: optional feature extractor, e.g. {"type": "entity_set", "hidden": 64, "out": 64}
(brotato_rl/extractors.py: one shared network per table row, pooled; default = flatten everything).

train.teacher: optional HeuristicBot as a teacher during PPO: after every update, cross-entropy
towards the bot's actions on the states of the last rollout, with a weight decaying to 0
(brotato_rl/teacher.py). Logs teacher/agreement etc. to progress.csv.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np

from brotato_rl import paths, spec as specs
from brotato_rl.evaluation import evaluate_model
from brotato_rl.extractors import policy_kwargs_for
from brotato_rl.runs import model_steps
from brotato_rl.sb3 import make_sb3_vec_env

DEFAULT_CONFIG = paths.ROOT / "python" / "configs" / "ppo_baseline.json"
RUNS_DIR = paths.ROOT / "runs"


# -- config ---------------------------------------------------------------------------------

def set_path(cfg, dotted, raw):
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path, overrides):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.pop("_comment", None)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        set_path(cfg, key, value)
    cfg["spec"] = specs.normalize(cfg.get("spec"))
    return cfg


def lr_at(schedule, timesteps):
    """Learning rate of a dict schedule at an absolute step count."""
    start, end = float(schedule["start"]), float(schedule.get("end", schedule["start"]))
    frac = min(1.0, max(0.0, timesteps / float(schedule.get("horizon", 1))))
    kind = schedule.get("type", "linear")
    if kind == "constant":
        return start
    if kind == "linear":
        return start + (end - start) * frac
    if kind == "cosine":
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"unknown lr_schedule type {kind!r}")


# -- callbacks ------------------------------------------------------------------------------

def make_callbacks(cfg, run_dir, spec, log):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

    t = cfg["train"]

    class WaveStats(BaseCallback):
        """Logs wave / outcome / reward parts of finished training episodes (last 50)."""

        def __init__(self):
            super().__init__()
            self.episodes = deque(maxlen=50)
            self.parts = None
            self.total_episodes = 0

        def _on_training_start(self):
            self.parts = [dict() for _ in range(self.training_env.num_envs)]
            self.crashes = [0] * self.training_env.num_envs

        def _on_step(self):
            for i, info in enumerate(self.locals["infos"]):
                for k, v in (info.get("reward_parts") or {}).items():
                    self.parts[i][k] = self.parts[i].get(k, 0.0) + v
                ep = info.get("episode")
                if ep is not None:
                    self.episodes.append({**ep, "parts": self.parts[i]})
                    self.parts[i] = {}
                    self.total_episodes += 1
                    self.crashes[i] = ep.get("crashes") or 0
            return True

        def _on_rollout_end(self):
            if not self.episodes:
                return
            eps = list(self.episodes)
            waves = [e["wave"] or 0 for e in eps]
            rec = self.logger.record
            rec("rollout/mean_wave", float(np.mean(waves)))
            rec("rollout/max_wave", int(max(waves)))
            rec("rollout/win_rate", float(np.mean([e["outcome"] == "won" for e in eps])))
            rec("rollout/error_rate", float(np.mean([e["outcome"] == "error" for e in eps])))
            rec("rollout/episodes_total", self.total_episodes)
            rec("rollout/crashes_total", int(sum(self.crashes)))
            for k in sorted({k for e in eps for k in e["parts"]}):
                rec(f"reward_parts/{k}", float(np.mean([e["parts"].get(k, 0.0) for e in eps])))

    class PeriodicEval(BaseCallback):
        """Evaluation on fixed seeds; keeps best/model.zip by mean wave (ties: wins, then return)."""

        def __init__(self):
            super().__init__()
            self.every = int(t["eval_every"])
            self.next_at = None
            self.best = None
            best_file = run_dir / "best" / "eval.json"
            if best_file.is_file():
                self.best = json.loads(best_file.read_text(encoding="utf-8"))["summary"]

        def _key(self, s):
            return (s["mean_wave"], s["wins"], s["mean_return"])

        def _on_training_start(self):
            self.next_at = (self.num_timesteps // self.every + 1) * self.every

        def _on_step(self):
            if self.every > 0 and self.num_timesteps >= self.next_at:
                self.next_at += self.every
                self.run_eval()
            return True

        def run_eval(self):
            seeds = list(range(int(t["eval_seed"]), int(t["eval_seed"]) + int(t["eval_episodes"])))
            start = time.time()
            log(f"evaluation at {self.num_timesteps} steps: {len(seeds)} episodes on {t['eval_envs']} games "
                f"(seeds {seeds[0]}-{seeds[-1]})")
            try:
                summary, episodes = evaluate_model(self.model, lambda: specs.env_kwargs(spec), seeds,
                                                   n_envs=int(t["eval_envs"]), deterministic=True)
            except Exception as exc:  # noqa: BLE001 - an evaluation failure must not stop training
                log(f"evaluation failed: {type(exc).__name__}: {exc}")
                return
            summary.update(timesteps=int(self.num_timesteps), seconds=round(time.time() - start, 1),
                           seeds=[seeds[0], seeds[-1]])
            if t.get("eval_stochastic"):
                try:
                    stoch, _ = evaluate_model(self.model, lambda: specs.env_kwargs(spec), seeds,
                                              n_envs=int(t["eval_envs"]), deterministic=False)
                    summary["stochastic"] = {k: stoch[k] for k in ("mean_wave", "sem_wave", "wins", "mean_return")}
                    self.logger.record("eval_stochastic/mean_wave", stoch["mean_wave"])
                    log(f"evaluation (sampled actions): mean wave {stoch['mean_wave']} +- {stoch['sem_wave']}")
                except Exception as exc:  # noqa: BLE001
                    log(f"stochastic evaluation failed: {type(exc).__name__}: {exc}")
            with open(run_dir / "evals.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")
            rec = self.logger.record
            for key in ("mean_wave", "sem_wave", "median_wave", "best_wave", "wins", "corner_frac", "wall_frac",
                        "freedom", "mean_return"):
                if summary.get(key) is not None:
                    rec(f"eval/{key}", summary[key])
            improved = self.best is None or self._key(summary) > self._key(self.best)
            log(f"evaluation: mean wave {summary['mean_wave']} +- {summary['sem_wave']}, wins {summary['wins']}, "
                f"corner {summary['corner_frac']:.1%}, {summary['seconds']}s"
                + (" -> new best" if improved else f" (best {self.best['mean_wave']})"))
            if improved:
                self.best = summary
                (run_dir / "best").mkdir(exist_ok=True)
                self.model.save(run_dir / "best" / "model.zip")
                (run_dir / "best" / "eval.json").write_text(
                    json.dumps({"summary": summary, "episodes": episodes}, indent=1), encoding="utf-8")

    class TimestepLr(BaseCallback):
        """Sets the learning rate from the absolute step count before every update, so the
        schedule does not depend on total_timesteps (and --resume does not make it jump)."""

        def __init__(self, schedule):
            super().__init__()
            self.schedule = schedule

        def _on_step(self):
            return True

        def _on_rollout_end(self):
            self.model.learning_rate = lr_at(self.schedule, self.num_timesteps)
            self.model._setup_lr_schedule()

    n_envs = int(t["n_envs"])
    checkpoint = CheckpointCallback(save_freq=max(1, int(t["checkpoint_every"]) // n_envs),
                                    save_path=str(run_dir / "checkpoints"), name_prefix="model",
                                    save_vecnormalize=True)
    callbacks = [WaveStats(), checkpoint]
    if isinstance(cfg["ppo"].get("lr_schedule"), dict):
        callbacks.append(TimestepLr(cfg["ppo"]["lr_schedule"]))
    if t.get("teacher"):
        from brotato_rl.teacher import TeacherCallback
        if spec.get("action_type", "discrete") != "discrete":
            raise SystemExit("train.teacher needs action_type discrete (the bot picks one of the 9 directions)")
        tc = t["teacher"]
        log(f"teacher: HeuristicBot, weight {tc.get('weight', 1.0)} -> 0 over {tc.get('decay_steps', 5_000_000)} steps, "
            f"lr {tc.get('learning_rate', 1e-4)}, {tc.get('epochs', 1)} pass(es) per rollout")
        callbacks.append(TeacherCallback(tc))
    evaluator = None
    if int(t["eval_every"]) > 0 and int(t["eval_episodes"]) > 0:
        evaluator = PeriodicEval()
        callbacks.append(evaluator)
    return CallbackList(callbacks), evaluator


# -- main -----------------------------------------------------------------------------------

def latest_checkpoint(run_dir):
    """The saved model with the most steps: a checkpoint, or interrupted/ / crashed/ when newer."""
    candidates = list((run_dir / "checkpoints").glob("model_*_steps.zip"))
    candidates += [p for p in (run_dir / "interrupted" / "model.zip", run_dir / "crashed" / "model.zip")
                   if p.is_file()]
    return max(candidates, key=model_steps) if candidates else None


def rotate_progress(run_dir):
    """SB3's CSV logger truncates progress.csv; keep the earlier session as progress.partN.csv."""
    current = run_dir / "progress.csv"
    if not current.is_file() or current.stat().st_size == 0:
        return None
    n = 1
    while (run_dir / f"progress.part{n}.csv").exists():
        n += 1
    target = run_dir / f"progress.part{n}.csv"
    current.rename(target)
    return target


def rotate_monitor(run_dir):
    """SB3's Monitor reopens monitor/env<N>.monitor.csv with "w"; keep the earlier session as monitor.partN/."""
    current = run_dir / "monitor"
    if not current.is_dir() or not any(current.iterdir()):
        return None
    n = 1
    while (run_dir / f"monitor.part{n}").exists():
        n += 1
    target = run_dir / f"monitor.part{n}"
    current.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="run folder name under runs/")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, dotted path + JSON value, e.g. train.n_envs=4")
    parser.add_argument("--resume", nargs="?", const="latest",
                        help="continue: 'latest' checkpoint of this run (default) or a model .zip path")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(errors="replace")
    warnings.filterwarnings("ignore", message="Steam is running")

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import VecNormalize

    run_dir = RUNS_DIR / args.name
    if args.resume:
        if not (run_dir / "config.json").is_file():
            raise SystemExit(f"--resume: {run_dir} has no config.json")
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for item in args.set:  # allowed: e.g. a longer train.total_timesteps
            key, _, value = item.partition("=")
            set_path(cfg, key, value)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"{run_dir} already exists; use another --name or --resume")
        cfg = load_config(args.config, args.set)
    spec, p, t = cfg["spec"], cfg["ppo"], cfg["train"]
    run_dir.mkdir(parents=True, exist_ok=True)
    specs.save(spec, run_dir / specs.SPEC_FILE)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    log_file = open(run_dir / "train.log", "a", encoding="utf-8")

    def log(text):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    torch.set_num_threads(int(t["torch_threads"]))
    n_envs = int(t["n_envs"])
    log(f"run {run_dir}: {n_envs} games, {int(t['total_timesteps'])} steps, meta {spec['meta']['type']}, "
        f"reward shaping {spec['reward']['freedom_weight']}/{spec['reward']['wall_weight']}, "
        f"encoder free_rays={spec['encoder']['free_rays']} sectors={spec['encoder']['sectors']}")
    if abs(spec["reward"]["shaping_gamma"] - float(p["gamma"])) > 1e-9:
        log(f"WARNING: reward.shaping_gamma {spec['reward']['shaping_gamma']} != ppo.gamma {p['gamma']}: "
            "the positional shaping is then not potential-based")

    if args.resume:
        rotated = rotate_monitor(run_dir)
        if rotated:
            log(f"earlier monitor/ kept as {rotated.name}")
    venv = make_sb3_vec_env(n_envs, env_kwargs=specs.env_kwargs(spec), seed=int(t["seed"]),
                            monitor_dir=run_dir / "monitor")
    resume_from = None
    if args.resume:
        resume_from = latest_checkpoint(run_dir) if args.resume == "latest" else Path(args.resume)
        if resume_from is None or not resume_from.exists():
            venv.close()
            raise SystemExit(f"--resume: no checkpoint found ({resume_from})")
    if t.get("norm_reward"):
        pkl = None
        if resume_from is not None:
            for candidate in (resume_from.with_name(resume_from.stem.replace("model_", "model_vecnormalize_", 1) + ".pkl"),
                              resume_from.with_name("vecnormalize.pkl")):  # checkpoints/ or final/ / interrupted/
                if candidate.is_file():
                    pkl = candidate
                    break
            if pkl is None:
                log(f"WARNING: no VecNormalize statistics next to {resume_from}; starting them fresh")
        venv = VecNormalize.load(str(pkl), venv) if pkl else \
            VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=float(p["gamma"]))
        venv.training = True

    if resume_from is not None:
        model = PPO.load(str(resume_from), env=venv, device="cpu")
        done = model.num_timesteps
        log(f"resumed from {resume_from} at {done} steps")
        rotated = rotate_progress(run_dir)
        if rotated:
            log(f"earlier progress.csv kept as {rotated.name}")
    else:
        lr = float(p["learning_rate"])
        if isinstance(p.get("lr_schedule"), dict):
            schedule = lr_at(p["lr_schedule"], 0)  # updated before every PPO update (TimestepLr)
        elif p.get("lr_schedule") == "linear":
            schedule = lambda progress: lr * progress  # noqa: E731 - old runs: linear to 0 at total_timesteps
        else:
            schedule = lr
        activation = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[p.get("activation", "tanh")]
        model = PPO(p["policy"], venv, n_steps=int(p["n_steps"]), batch_size=int(p["batch_size"]),
                    n_epochs=int(p["n_epochs"]), learning_rate=schedule, gamma=float(p["gamma"]),
                    gae_lambda=float(p["gae_lambda"]), clip_range=float(p["clip_range"]),
                    ent_coef=float(p["ent_coef"]), vf_coef=float(p["vf_coef"]), max_grad_norm=float(p["max_grad_norm"]),
                    target_kl=p.get("target_kl"), seed=int(t["seed"]), device="cpu", verbose=0,
                    policy_kwargs={"net_arch": p["net_arch"], "activation_fn": activation,
                                   **policy_kwargs_for(p.get("extractor"))})
        if p.get("extractor"):
            log(f"feature extractor: {p['extractor']}")
        done = 0
    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"]))
    callbacks, evaluator = make_callbacks(cfg, run_dir, spec, log)
    if resume_from is None and t.get("bc"):
        from brotato_rl.bc import pretrain
        bc_report = pretrain(model, venv if isinstance(venv, VecNormalize) else None, t["bc"], float(p["gamma"]), log)
        (run_dir / "bc.json").write_text(json.dumps(bc_report, indent=1), encoding="utf-8")
        (run_dir / "bc").mkdir(exist_ok=True)
        model.save(run_dir / "bc" / "model.zip")
        if evaluator is not None and t["bc"].get("evaluate", True):
            evaluator.init_callback(model)
            evaluator.run_eval()

    remaining = int(t["total_timesteps"]) - done
    status = "final"
    start = time.time()
    try:
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False,
                        tb_log_name="ppo", progress_bar=False)
    except KeyboardInterrupt:
        status = "interrupted"
        log("interrupted (Ctrl+C), saving")
    except Exception as exc:  # noqa: BLE001 - save what we have, then re-raise
        status = "crashed"
        log(f"training crashed: {type(exc).__name__}: {exc}; saving")
        raise
    finally:
        out = run_dir / status
        out.mkdir(exist_ok=True)
        model.save(out / "model.zip")
        if isinstance(venv, VecNormalize):
            venv.save(str(out / "vecnormalize.pkl"))
        elapsed = time.time() - start
        log(f"{status}: {model.num_timesteps} steps total, {model.num_timesteps - done} this session in "
            f"{elapsed / 60:.1f} min ({(model.num_timesteps - done) / max(elapsed, 1):.0f} steps/s); saved {out}")
        if evaluator is not None and evaluator.best is not None:
            log(f"best evaluation: mean wave {evaluator.best['mean_wave']} at {evaluator.best['timesteps']} steps "
                f"-> {run_dir / 'best' / 'model.zip'}")
        try:
            venv.close()
        finally:
            log_file.close()


if __name__ == "__main__":
    main()
