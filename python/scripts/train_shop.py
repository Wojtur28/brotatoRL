"""Shop agent: rule demonstrations, behaviour cloning, PPO and evaluation (brotato_rl/shop_train.py).

    python python\\scripts\\train_shop.py collect --name shop_rule300 --episodes 300 --explore 0.1
    python python\\scripts\\train_shop.py bc --name shop1 --demos demos\\shop_rule300.npz
    python python\\scripts\\train_shop.py ppo --name shop1 --init runs\\shop1\\bc\\shop.pt --hours 10
    python python\\scripts\\train_shop.py stop --name shop1                       # save and exit (seconds)
    python python\\scripts\\train_shop.py ppo --name shop1 --resume --hours 10      # continue where it stopped
    python python\\scripts\\train_shop.py eval --name shop1_best --model runs\\shop1\\best\\shop.pt --seed 10000 --episodes 96
    python python\\scripts\\train_shop.py eval --name rule_s96 --rule --seed 10000 --episodes 96

Movement is always the frozen pilot (--pilot, default models\\pilot_ppo5_10M\\pilot.npz, exported with
python -m brotato_rl.pilot): its spec (character, reward, rule meta policy) is the environment.
Seeds: 10000+ tests, 20000+ validation (PPO evaluations), 1000000+ PPO training, 2000000+ demos.

runs/<name>/ (bc and ppo):
  config.json, catalog.json   settings and the id vocabulary (fixed for the run)
  train.log                   everything printed
  bc/shop.pt, bc.json         behaviour cloning model and its accuracy
  progress.csv                one row per PPO update; episodes.csv: one row per training episode
  evals.jsonl                 every evaluation (greedy, eval_episodes validation seeds)
  best/shop.pt, best/eval.json; final/shop.pt
  checkpoints/latest.pt       after every PPO update; shop_<decisions>.pt every checkpoint_every updates
  interrupted/shop.pt         after `stop` (a STOP file in the run folder) or Ctrl+C
  --resume continues from the newest of these (network, optimizer, counters, seeds, best so far).
  A hard kill or a reboot loses at most the episodes since the last update (~4 minutes).
Evaluations and demos also go to results/<date>_<name>.json (eval) and demos/<name>.npz (collect).
"""
import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from brotato_rl import paths
from brotato_rl.shop import ShopConfig, Vocab
from brotato_rl.shop_train import (ActorPool, ShopTrainConfig, copy_net, fetch_catalog, ppo_update,
                                   serve_actions, summarize_episodes, write_json)

RUNS, RESULTS, DEMOS = paths.ROOT / "runs", paths.ROOT / "results", paths.ROOT / "demos"
STOP_FILE = "STOP"
EPISODE_FIELDS = ("seed", "mode", "wave", "outcome", "steps", "seconds", "decisions", "return", "error", "tag")


def make_logger(path=None):
    handle = open(path, "a", encoding="utf-8") if path else None

    def log(text):
        line = f"[{datetime.now():%H:%M:%S}] {text}"
        print(line, flush=True)
        if handle:
            handle.write(line + "\n")
            handle.flush()
    return log


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def episode_row(ep):
    return {k: ep.get(k) for k in EPISODE_FIELDS}


def run_jobs(pool, next_job, on_episode, net=None, greedy_net=lambda: None, abort=lambda: False):
    """Serves the actors until every actor got None from next_job, or until abort() is true (then
    returns True at once, with episodes still running). greedy_net(): the network for greedy
    ("eval") requests (None = net)."""
    while len(pool.stopped) < len(pool.conns):
        if abort():
            return True
        acts = []
        for rank, msg in pool.poll(timeout=1.0):
            if msg[0] == "act":
                acts.append((rank, msg[2], msg[3], msg[4]))
            elif msg[0] == "next":
                pool.send(rank, next_job())
            elif msg[0] == "episode":
                on_episode(msg[2])
        if acts:
            for rank, reply in serve_actions(net, acts, greedy_net()):
                pool.send(rank, reply)
    return False


def start_pool(args, cfg, catalog, log):
    log(f"starting {cfg['envs']} games (pilot {cfg['pilot']})")
    return ActorPool(cfg["envs"], cfg, catalog, log)


def load_catalog(folder: Path, log):
    """folder/catalog.json, read from a game (and saved there) the first time."""
    path = folder / "catalog.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    catalog = fetch_catalog()
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog), encoding="utf-8")
    log(f"catalog: {', '.join(f'{len(v)} {k}' for k, v in catalog.items())}")
    return catalog


# -- collect: rule demonstrations -------------------------------------------------------------

def cmd_collect(args, cfg):
    DEMOS.mkdir(exist_ok=True)
    log = make_logger()
    catalog = load_catalog(DEMOS / f"{args.name}_meta", log)
    pool = start_pool(args, cfg, catalog, log)
    episodes, assigned, errors = [], [0], [0]
    seeds = iter(range(args.seed, args.seed + 10 * args.episodes))
    start = time.time()

    def next_job():
        if assigned[0] - errors[0] >= args.episodes:
            return None
        assigned[0] += 1
        return {"seed": next(seeds), "mode": "rule", "explore": args.explore}

    def on_episode(ep):
        if ep["outcome"] == "error" or "obs" not in ep:
            errors[0] += 1
            log(f"seed {ep['seed']}: error {str(ep.get('error'))[:200]}")
            return
        episodes.append(ep)
        if len(episodes) % 10 == 0:
            rate = len(episodes) / (time.time() - start) * 3600
            log(f"{len(episodes)}/{args.episodes} episodes, {sum(e['decisions'] for e in episodes)} decisions, "
                f"mean wave {np.mean([e['wave'] for e in episodes]):.2f}, {rate:.0f} runs/h")

    try:
        run_jobs(pool, next_job, on_episode)
    finally:
        pool.close()
    keys = episodes[0]["obs"].keys()
    out = DEMOS / f"{args.name}.npz"
    np.savez_compressed(
        out, **{f"obs_{k}": np.concatenate([e["obs"][k] for e in episodes]) for k in keys},
        mask=np.concatenate([e["mask"] for e in episodes]), action=np.concatenate([e["action"] for e in episodes]),
        reward=np.concatenate([e["reward"] for e in episodes]),
        episode=np.concatenate([np.full(e["decisions"], i) for i, e in enumerate(episodes)]))
    summary = summarize_episodes(episodes)
    write_json(DEMOS / f"{args.name}_meta" / "demos.json",
               {"created": datetime.now().isoformat(timespec="seconds"), "explore": args.explore, "config": cfg,
                "summary": summary, "episodes": [episode_row(e) for e in episodes]})
    log(f"saved {out}: {summary} in {(time.time() - start) / 60:.1f} min")


# -- bc ---------------------------------------------------------------------------------------

def discounted_returns(reward, episode, gamma):
    ret = np.zeros_like(reward)
    running, last_ep = 0.0, None
    for t in range(len(reward) - 1, -1, -1):
        if episode[t] != last_ep:
            running, last_ep = 0.0, episode[t]
        running = reward[t] + gamma * running
        ret[t] = running
    return ret


def cmd_bc(args, cfg):
    import torch as th

    from brotato_rl.shop_net import ShopNet, masked_categorical, save_shop_model

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    demos = Path(args.demos)
    catalog = json.loads((demos.parent / f"{demos.stem}_meta" / "catalog.json").read_text(encoding="utf-8"))
    (run_dir / "catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    vocab = Vocab(catalog)
    data = np.load(demos)
    obs = {k[4:]: data[k] for k in data.files if k.startswith("obs_")}
    mask, action, episode = data["mask"], data["action"], data["episode"]
    ret = discounted_returns(data["reward"] * cfg["reward_scale"], episode, cfg["gamma"])
    n_ep = int(episode.max()) + 1
    rng = np.random.default_rng(0)
    val_eps = set(rng.choice(n_ep, max(1, n_ep // 10), replace=False).tolist())
    is_val = np.array([e in val_eps for e in episode])
    log(f"bc: {len(action)} decisions from {n_ep} episodes ({is_val.sum()} held out), vocab {vocab.size}")

    th.manual_seed(0)
    net = ShopNet(vocab.size, len(vocab.items), **cfg["net"])
    opt = th.optim.Adam(net.parameters(), lr=args.lr)
    t = {k: th.as_tensor(v) for k, v in obs.items()}
    t_mask, t_act, t_ret = th.as_tensor(mask), th.as_tensor(action), th.as_tensor(ret)
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    kind = obs["glob"][:, :3].argmax(axis=1)  # 0 shop, 1 level-up, 2 crate

    def evaluate(idx):
        with th.no_grad():
            logits, value = net({k: v[idx] for k, v in t.items()})
            pred = masked_categorical(logits, t_mask[idx]).probs.argmax(dim=1).numpy()
        hit = pred == action[idx]
        out = {"acc": round(float(hit.mean()), 4),
               "v_mse": round(float(((value.numpy() - ret[idx]) ** 2).mean()), 4)}
        for k, name in enumerate(("shop", "level_up", "item")):
            sel = kind[idx] == k
            if sel.any():
                out[f"acc_{name}"] = round(float(hit[sel].mean()), 4)
        return out

    best, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), 256):
            idx = th.as_tensor(order[start:start + 256])
            logits, value = net({k: v[idx] for k, v in t.items()})
            ce = -masked_categorical(logits, t_mask[idx]).log_prob(t_act[idx]).mean()
            v_loss = ((value - t_ret[idx]) ** 2).mean()
            loss = ce + args.value_coef * v_loss
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(ce.item())
        net.eval()
        val = evaluate(val_idx)
        if val["acc"] > best:
            best, best_state = val["acc"], {k: v.clone() for k, v in net.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            log(f"bc epoch {epoch}: train ce {np.mean(losses):.4f}, held-out {val}")
    net.load_state_dict(best_state)
    report = {"demos": str(demos), "decisions": int(len(action)), "episodes": n_ep, "epochs": args.epochs,
              "train": evaluate(train_idx), "held_out": evaluate(val_idx)}
    save_shop_model(run_dir / "bc" / "shop.pt", net, vocab, ShopConfig(**cfg["shop"]), {"bc": report})
    write_json(run_dir / "bc.json", report)
    log(f"bc saved {run_dir / 'bc' / 'shop.pt'}: held-out {report['held_out']}")


# -- ppo --------------------------------------------------------------------------------------

def latest_checkpoint(run_dir: Path):
    """The most recently written training state (checkpoints/, interrupted/, final/)."""
    found = list((run_dir / "checkpoints").glob("*.pt"))
    found += [p for p in (run_dir / "interrupted" / "shop.pt", run_dir / "final" / "shop.pt") if p.is_file()]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def cmd_ppo(args, cfg):
    import torch as th

    from brotato_rl.shop_net import ShopNet, load_shop_model, save_shop_model

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(run_dir / "train.log")
    stop_file = run_dir / STOP_FILE
    if stop_file.exists():
        stop_file.unlink()
        log(f"removed an old {STOP_FILE} file")
    c = ShopTrainConfig(**cfg)
    catalog = load_catalog(run_dir, log)
    vocab = Vocab(catalog)
    shop_config = ShopConfig(**c.shop)
    state = {"decisions": 0, "updates": 0, "episodes": 0, "train_seed": c.train_seed, "best": None, "game_steps": 0,
             "eval_pending": False}

    resume = latest_checkpoint(run_dir) if args.resume else None
    if args.resume and resume is None:
        raise SystemExit(f"--resume: no checkpoint in {run_dir / 'checkpoints'}")
    if resume:
        net, vocab, shop_config, data = load_shop_model(resume)
        state.update(data["train_state"])
        log(f"resumed from {resume} at {state['decisions']} decisions, {state['updates']} updates")
    elif c.init:
        net, init_vocab, _, _ = load_shop_model(c.init)
        if init_vocab.catalog != catalog:
            raise SystemExit("the --init model was trained with another catalog")
        log(f"starting from {c.init}")
    else:
        net = ShopNet(vocab.size, len(vocab.items), **c.net)
        log("starting from a fresh network")
    optimizer = th.optim.Adam(net.parameters(), lr=c.lr)
    if resume and "optimizer" in data:
        optimizer.load_state_dict(data["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = c.lr
    write_json(run_dir / "config.json", {**cfg, "resumed_from": str(resume) if resume else None})

    def save(path, extra=None, optimizer_state=True):
        committed = dict(state)  # episodes waiting in the buffer are not in the network yet
        committed["decisions"] -= sum(e["decisions"] for e in buffer)
        committed["episodes"] -= len(buffer)
        committed["game_steps"] -= sum(e.get("steps", 0) for e in buffer)
        extra = dict(extra or {})
        if optimizer_state:
            extra["optimizer"] = optimizer.state_dict()
        save_shop_model(path, net, vocab, shop_config, {"train_state": committed, **extra})

    progress_new = not (run_dir / "progress.csv").is_file()
    progress = open(run_dir / "progress.csv", "a", newline="", encoding="utf-8")
    progress_writer = None
    episodes_new = not (run_dir / "episodes.csv").is_file()
    episodes_file = open(run_dir / "episodes.csv", "a", newline="", encoding="utf-8")
    episodes_writer = csv.DictWriter(episodes_file, fieldnames=("time", "decisions_total") + EPISODE_FIELDS)
    if episodes_new:
        episodes_writer.writeheader()

    buffer, recent = [], []
    eval_queue, eval_results, eval_net = [], [], [None]
    session_start = time.time()
    session_decisions = [0]
    stopping = [False]

    def schedule_eval():
        if eval_queue or eval_results:
            return
        eval_net[0] = copy_net(net)
        eval_queue.extend(range(c.eval_seed, c.eval_seed + c.eval_episodes))
        eval_results.clear()
        eval_net[0].at = dict(state)
        state["eval_pending"] = True

    def finish_eval():
        summary = summarize_episodes(eval_results)
        at = eval_net[0].at
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "decisions": at["decisions"],
                 "updates": at["updates"], "summary": summary}
        with open(run_dir / "evals.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log(f"evaluation at {at['decisions']} decisions ({at['updates']} updates): mean wave {summary['mean_wave']} "
            f"+- {summary['sem_wave']}, wins {summary['wins']}/{summary['episodes']}, early deaths {summary['early_deaths']}")
        if state["best"] is None or summary["mean_wave"] > state["best"]:
            state["best"] = summary["mean_wave"]
            live = net.state_dict()
            net.load_state_dict(eval_net[0].state_dict())
            save(run_dir / "best" / "shop.pt", {"eval": entry}, optimizer_state=False)
            net.load_state_dict(live)
            write_json(run_dir / "best" / "eval.json", {**entry, "episodes": [episode_row(e) for e in eval_results]})
            log(f"new best: {summary['mean_wave']}")
        eval_results.clear()
        eval_net[0] = None
        state["eval_pending"] = False

    def next_job():
        if eval_queue:  # an evaluation that has started is finished even when stopping
            return {"seed": eval_queue.pop(0), "mode": "eval", "tag": "eval"}
        if stopping[0]:
            return None
        state["train_seed"] += 1
        return {"seed": state["train_seed"], "mode": "train"}

    def on_episode(ep):
        nonlocal progress_writer
        if ep.get("mode") == "eval":
            eval_results.append(ep)
            if len(eval_results) >= c.eval_episodes and not eval_queue:
                finish_eval()
            return
        episodes_writer.writerow({"time": round(time.time() - session_start, 1), "decisions_total": state["decisions"],
                                  **episode_row(ep)})
        episodes_file.flush()
        if ep["outcome"] == "error" or "obs" not in ep:
            log(f"training episode lost (seed {ep['seed']}): {str(ep.get('error'))[:300]}")
            return
        buffer.append(ep)
        recent.append(ep)
        state["episodes"] += 1
        state["decisions"] += ep["decisions"]
        state["game_steps"] += ep.get("steps", 0)
        session_decisions[0] += ep["decisions"]
        if sum(e["decisions"] for e in buffer) < c.rollout:
            return
        t0 = time.time()
        warmup = state["updates"] < c.value_warmup
        stats = ppo_update(net, optimizer, buffer, c, train_policy=not warmup)
        state["updates"] += 1
        hours = (time.time() - session_start) / 3600
        row = {"time": datetime.now().isoformat(timespec="seconds"), "decisions": state["decisions"],
               "updates": state["updates"], "episodes": state["episodes"], **summarize_episodes(recent),
               "decisions_per_h": round(session_decisions[0] / max(hours, 1e-6)), "update_s": round(time.time() - t0, 1),
               **stats}
        if progress_writer is None:
            progress_writer = csv.DictWriter(progress, fieldnames=list(row), extrasaction="ignore")
            if progress_new:
                progress_writer.writeheader()
        progress_writer.writerow(row)
        progress.flush()
        log(f"update {state['updates']}{' (critic only)' if warmup else ''}: {state['decisions']} decisions, "
            f"train wave {row['mean_wave']} "
            f"({row['wins']}/{row['episodes']} won), kl {stats['approx_kl']}, clip {stats['clip_frac']}, "
            f"entropy {stats['entropy']}, ev {stats['explained_var']}, {row['decisions_per_h']} decisions/h")
        buffer.clear()
        recent.clear()
        save(run_dir / "checkpoints" / "latest.pt")  # every update: a stop or crash loses at most one rollout
        if c.checkpoint_every and state["updates"] % c.checkpoint_every == 0:
            save(run_dir / "checkpoints" / f"shop_{state['decisions']}.pt")
        if c.eval_every and state["updates"] % c.eval_every == 0:
            schedule_eval()
        if not stopping[0] and (state["decisions"] >= c.total_decisions or (c.hours and hours >= c.hours)):
            stopping[0] = True
            log(f"stopping: {state['decisions']} decisions, {hours:.2f} h this session")

    log(f"ppo: {c.envs} games, rollout {c.rollout}, lr {c.lr}, gamma {c.gamma}, ent {c.ent_coef}, "
        f"until {c.total_decisions} decisions" + (f" or {c.hours} h" if c.hours else ""))
    if c.eval_every and (not resume or state["eval_pending"]):
        state["eval_pending"] = False
        schedule_eval()
    log(f"to stop and keep the state: python python\\scripts\\train_shop.py stop --name {args.name} "
        f"(or create {stop_file}); continue with: ppo --name {args.name} --resume")
    pool = start_pool(args, cfg, catalog, log)
    aborted = False
    try:
        aborted = run_jobs(pool, next_job, on_episode, net=net, greedy_net=lambda: eval_net[0],
                           abort=stop_file.exists)
    except KeyboardInterrupt:
        log("interrupted (Ctrl+C), saving")
        save(run_dir / "interrupted" / "shop.pt")
        pool.terminate()
        raise
    finally:
        progress.close()
        episodes_file.close()
    if aborted:
        save(run_dir / "interrupted" / "shop.pt")
        stop_file.unlink(missing_ok=True)
        log(f"stopped on request ({STOP_FILE}): saved interrupted/shop.pt at "
            f"{state['decisions'] - sum(e['decisions'] for e in buffer)} decisions "
            f"({len(buffer)} episodes not yet trained on dropped); continue with --resume")
        pool.terminate()
        return
    pool.close()
    save(run_dir / "final" / "shop.pt")
    save(run_dir / "checkpoints" / f"shop_{state['decisions']}.pt")
    log(f"done: {state['decisions']} decisions, {state['updates']} updates, best validation wave {state['best']}")


# -- eval -------------------------------------------------------------------------------------

def cmd_eval(args, cfg):
    log = make_logger()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(range(args.seed, args.seed + args.episodes))
    net = None
    if args.rule:
        catalog = load_catalog(paths.LAB, log)  # the rule needs no vocabulary, the encoder does
        mode = "rule"
    else:
        from brotato_rl.shop_net import load_shop_model
        net, vocab, shop_config, _ = load_shop_model(args.model)
        catalog = vocab.catalog
        cfg["shop"] = asdict(shop_config)
        mode = "eval"
    pool = start_pool(args, cfg, catalog, log)
    queue, episodes, start = list(seeds), [], time.time()

    def next_job():
        return {"seed": queue.pop(0), "mode": mode, "explore": 0.0} if queue else None

    def on_episode(ep):
        episodes.append(ep)
        log(f"seed {ep['seed']}: {ep['outcome']:>6} at wave {ep.get('wave')} | {ep.get('decisions')} decisions"
            + (f" | error {str(ep.get('error'))[:200]}" if ep["outcome"] == "error" else ""))

    try:
        run_jobs(pool, next_job, on_episode, net=net)
    finally:
        pool.close()
    episodes.sort(key=lambda e: e["seed"])
    summary = summarize_episodes(episodes)
    summary["wall_seconds"] = round(time.time() - start, 1)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M}_{args.name}.json"
    write_json(out, {"name": args.name, "created": datetime.now().isoformat(timespec="seconds"),
                     "config": {**cfg, "model": args.model, "rule": args.rule, "seeds": seeds},
                     "summary": summary, "episodes": [episode_row(e) for e in episodes]})
    log(f"{summary}\nsaved {out}")


def cmd_stop(args, cfg):
    run_dir = RUNS / args.name
    if not run_dir.is_dir():
        raise SystemExit(f"no run {run_dir}")
    (run_dir / STOP_FILE).write_text("stop\n", encoding="utf-8")
    print(f"created {run_dir / STOP_FILE}: the training saves its state and exits within seconds; "
          f"continue later with: train_shop.py ppo --name {args.name} --resume")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "bc", "ppo", "eval", "stop"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--envs", type=int, help="parallel games (default 8; 10 is the fastest on 6 cores/16 GB)")
    parser.add_argument("--pilot", help="numpy movement pilot (.npz)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="ShopTrainConfig field (JSON value), e.g. rollout=4096 or shop={\"weapon_type\":\"any\"}")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, help="first seed (collect: 2000000, eval: 10000)")
    parser.add_argument("--seeds", help="eval: comma-separated seeds instead of --seed/--episodes")
    parser.add_argument("--explore", type=float, default=0.0, help="collect: share of random actions")
    parser.add_argument("--demos", help="bc: demos .npz from collect")
    parser.add_argument("--epochs", type=int, default=40, help="bc epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="bc learning rate (ppo: --set lr=...)")
    parser.add_argument("--value-coef", type=float, default=0.5, help="bc: weight of the value regression")
    parser.add_argument("--init", help="ppo: start from this shop model (bc)")
    parser.add_argument("--hours", type=float, help="ppo: stop after this many hours")
    parser.add_argument("--resume", action="store_true", help="ppo: continue from the last checkpoint")
    parser.add_argument("--model", help="eval: shop model .pt")
    parser.add_argument("--rule", action="store_true", help="eval: the pilot spec's RuleMetaPolicy instead of a model")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Steam is running")

    cfg = asdict(ShopTrainConfig())
    if args.resume and (RUNS / args.name / "config.json").is_file():
        saved = json.loads((RUNS / args.name / "config.json").read_text(encoding="utf-8"))
        cfg.update({k: v for k, v in saved.items() if k in cfg})
    known = {f.name for f in fields(ShopTrainConfig)}
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in known:
            parser.error(f"unknown setting {key!r} (ShopTrainConfig: {', '.join(sorted(known))})")
        cfg[key] = parse_value(value)
    for key in ("envs", "pilot", "init", "hours"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg["pilot"] = str((paths.ROOT / cfg["pilot"]).resolve()) if not Path(cfg["pilot"]).is_absolute() else cfg["pilot"]
    if args.seed is None:
        args.seed = 2_000_000 if args.command == "collect" else 10000
    if args.command == "eval" and not (args.rule or args.model):
        parser.error("eval needs --model or --rule")
    {"collect": cmd_collect, "bc": cmd_bc, "ppo": cmd_ppo, "eval": cmd_eval, "stop": cmd_stop}[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
