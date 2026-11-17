"""Shop agent training machinery: game actors + one learner (asynchronous PPO with action masks).

Every actor process runs one game. Movement is played by a frozen numpy pilot (pilot.py); at every
between-wave decision the actor asks the learner for an action and waits for it. Games never wait
for each other: with SB3's VecEnv all games step together, so one game playing a 30-second wave
would hold up every other game's shop clicks (~4x slower here). torch lives only in the learner.

Episode = one run. Reward of decision t = the pilot spec's movement reward earned between decision t
and decision t+1 (wave cleared, damage, death, win, ...). Wave 1 comes before the first decision and
is not counted. Jobs (seed + mode) come from the learner:
  "train"  sample from the current policy, the episode goes into the next PPO update
  "eval"   greedy actions of a frozen copy of the policy (validation / test seeds)
  "rule"   the pilot spec's RuleMetaPolicy (optionally with random actions, `explore`), recorded
           with the rule's action as the label (behaviour cloning data, rule baseline)
"""
import copy
import json
import multiprocessing as mp
import multiprocessing.connection
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import spec as specs
from .meta import MetaPolicy
from .shop import N_WEAPONS, RECYCLE, ShopConfig, ShopEncoder, Vocab


@dataclass
class ShopTrainConfig:
    pilot: str = "models/pilot_ppo5_10M/pilot.npz"
    envs: int = 8
    shop: dict = field(default_factory=lambda: asdict(ShopConfig()))
    net: dict = field(default_factory=lambda: {"emb": 16, "hidden": 128})
    init: str = ""                    # shop model .pt to start from (behaviour cloning)
    total_decisions: int = 300_000
    hours: float = 0.0                # stop after this many hours (0 = no limit)
    rollout: int = 2048               # decisions per PPO update (whole episodes)
    epochs: int = 4
    minibatch: int = 256
    lr: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    value_warmup: int = 5             # first updates train only the critic (the policy starts from BC)
    train_seed: int = 1_000_000
    eval_every: int = 10              # PPO updates between evaluations (0 = never)
    eval_seed: int = 20000
    eval_episodes: int = 24
    checkpoint_every: int = 10


# -- actor (one game, no torch) ---------------------------------------------------------------

class _ActorMeta(MetaPolicy):
    def __init__(self, conn, rank, encoder: ShopEncoder, rule: MetaPolicy):
        self.conn, self.rank, self.encoder, self.rule = conn, rank, encoder, rule
        self.mode, self.explore = "train", 0.0
        self.rng = np.random.default_rng()
        self.records = []

    def seed(self, rng):
        self.rng = rng
        self.rule.seed(rng)

    def _decide(self, d, rule_method):
        obs, mask = self.encoder.encode(d), self.encoder.mask(d)
        if self.mode == "rule":
            reply = rule_method(d)
            label = self.encoder.action_of(d, reply)
            if label is not None and mask[label]:
                self.records.append((obs, mask, label, 0.0, 0.0))
            if self.explore and self.rng.random() < self.explore:
                # A random action for state variety, but never recycling a weapon: with up to six
                # recycle actions in every shop, uniform picks would mostly melt the build down.
                safe = mask.copy()
                safe[RECYCLE:RECYCLE + N_WEAPONS] = False
                reply = self.encoder.reply(d, int(self.rng.choice(np.flatnonzero(safe if safe.any() else mask))))
            return reply
        self.conn.send(("act", self.rank, obs, mask, self.mode == "eval"))
        action, logp, value = self.conn.recv()
        self.records.append((obs, mask, action, logp, value))
        return self.encoder.reply(d, action)

    def level_up(self, d):
        return self._decide(d, self.rule.level_up)

    def item(self, d):
        return self._decide(d, self.rule.item)

    def shop(self, d):
        return self._decide(d, self.rule.shop)


def _play(env, pilot, meta, seed):
    meta.records = []
    rewards, steps, start = [], 0, time.perf_counter()
    obs, info = env.reset(seed=seed)
    while True:
        before = len(meta.records)
        obs, reward, terminated, truncated, info = env.step(pilot.act(obs))
        steps += 1
        rewards += [0.0] * (len(meta.records) - len(rewards))
        if before:  # the reward of a step belongs to the last decision before it
            rewards[before - 1] += reward
        if terminated or truncated:
            break
    outcome = info.get("outcome") or ("error" if "error" in info else "truncated")
    ep = {"seed": seed, "mode": meta.mode, "wave": info.get("wave"), "outcome": outcome, "steps": steps,
          "seconds": round(time.perf_counter() - start, 1), "decisions": len(meta.records),
          "return": round(float(sum(rewards)), 3), "error": info.get("error")}
    if meta.records:
        obs_list, masks, actions, logps, values = zip(*meta.records)
        ep.update(obs={k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}, mask=np.stack(masks),
                  action=np.array(actions, np.int64), logp=np.array(logps, np.float32),
                  value=np.array(values, np.float32), reward=np.array(rewards, np.float32))
    return ep


def actor_main(rank, conn, cfg: dict, catalog: dict):
    import warnings
    warnings.filterwarnings("ignore", message="Steam is running")
    from .env import BrotatoEnv
    from .pilot import NumpyPilot

    pilot = NumpyPilot(cfg["pilot"])
    spec = specs.normalize(pilot.spec)
    encoder = ShopEncoder(Vocab(catalog), ShopConfig(**cfg["shop"]))
    meta = _ActorMeta(conn, rank, encoder, specs.build_meta(spec["meta"]))
    kwargs = specs.env_kwargs(spec)
    kwargs.update(meta_policy=meta, instance_name=f"shop{rank}-{int(time.time()) % 100000}")
    env = BrotatoEnv(**kwargs)
    try:
        while True:
            conn.send(("next", rank))
            job = conn.recv()
            if job is None:
                break
            meta.mode, meta.explore = job["mode"], float(job.get("explore", 0.0))
            try:
                ep = _play(env, pilot, meta, job["seed"])
            except Exception as exc:  # noqa: BLE001 - a lost game ends the episode, the actor goes on
                env._drop_instance()
                env._needs_reset = True
                ep = {"seed": job["seed"], "mode": meta.mode, "outcome": "error", "decisions": 0,
                      "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"}
            ep["tag"] = job.get("tag")
            conn.send(("episode", rank, ep))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        env.close()


def fetch_catalog() -> dict:
    """Starts one game to read the item catalog from its hello (mod 0.5.0+)."""
    from .instance import GameInstance
    game = GameInstance(name=f"catalog-{int(time.time()) % 100000}")
    try:
        hello = game.start()
    finally:
        game.close()
    if "catalog" not in hello:
        raise RuntimeError(f"the lab mod ({hello.get('mod_version')}) sends no catalog; "
                           "run python\\scripts\\setup_lab.py (mod 0.5.0+)")
    return hello["catalog"]


# -- learner side -----------------------------------------------------------------------------

class ActorPool:
    """Starts the actor processes and routes their messages; `jobs` hands out the next job."""

    def __init__(self, n, cfg: dict, catalog: dict, log):
        self.cfg, self.catalog, self.log = cfg, catalog, log
        self.ctx = mp.get_context("spawn")
        self.procs, self.conns = {}, {}
        self.stopped = set()  # told to exit (got None for "next")
        for rank in range(n):
            self._start(rank)

    def _start(self, rank):
        parent, child = self.ctx.Pipe()
        proc = self.ctx.Process(target=actor_main, args=(rank, child, self.cfg, self.catalog), daemon=True)
        proc.start()
        child.close()
        self.procs[rank], self.conns[rank] = proc, parent

    def poll(self, timeout=1.0):
        """[(rank, message)] of the actors that sent something; dead actors are restarted."""
        out = []
        conns = [c for r, c in self.conns.items() if r not in self.stopped]
        if not conns:
            time.sleep(timeout)
            return out
        for conn in mp.connection.wait(conns, timeout=timeout):
            rank = next(r for r, c in self.conns.items() if c is conn)
            try:
                out.append((rank, conn.recv()))
            except (EOFError, OSError):
                self.log(f"actor {rank} died (exit code {self.procs[rank].exitcode}); restarting it")
                conn.close()
                self._start(rank)
        return out

    def send(self, rank, msg):
        if msg is None:
            self.stopped.add(rank)
        try:
            self.conns[rank].send(msg)
        except (BrokenPipeError, OSError):
            pass

    def close(self, timeout=60):
        deadline = time.time() + timeout
        while len(self.stopped) < len(self.conns) and time.time() < deadline:
            for rank, msg in self.poll(timeout=1.0):
                if msg[0] == "next":
                    self.send(rank, None)
                elif msg[0] == "act":
                    self.send(rank, (int(np.flatnonzero(msg[3])[0]), 0.0, 0.0))
        for rank, proc in self.procs.items():
            proc.join(timeout=max(1.0, deadline - time.time()))
            if proc.is_alive():
                proc.kill()

    def terminate(self):
        """Ends every actor at once; their games die with them (Windows job object)."""
        for proc in self.procs.values():
            if proc.is_alive():
                proc.kill()
        for proc in self.procs.values():
            proc.join(timeout=10)


def summarize_episodes(episodes):
    waves = [int(e.get("wave") or 0) for e in episodes]
    if not waves:
        return {"episodes": 0}
    return {
        "episodes": len(waves),
        "mean_wave": round(float(np.mean(waves)), 2),
        "sem_wave": round(float(np.std(waves, ddof=1) / np.sqrt(len(waves))), 2) if len(waves) > 1 else None,
        "median_wave": float(np.median(waves)),
        "wins": sum(e["outcome"] == "won" for e in episodes),
        "errors": sum(e["outcome"] == "error" for e in episodes),
        "early_deaths": sum(e["outcome"] == "died" and w <= 8 for e, w in zip(episodes, waves)),
        "mean_return": round(float(np.mean([e.get("return", 0.0) for e in episodes])), 3),
        "mean_decisions": round(float(np.mean([e.get("decisions", 0) for e in episodes])), 1),
    }


def gae(reward, value, gamma, lam):
    """Advantages and returns of one finished episode (terminal at the end)."""
    adv = np.zeros_like(reward)
    last = 0.0
    for t in range(len(reward) - 1, -1, -1):
        next_value = value[t + 1] if t + 1 < len(reward) else 0.0
        delta = reward[t] + gamma * next_value - value[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + value


def concat_episodes(episodes):
    keys = episodes[0]["obs"].keys()
    return ({k: np.concatenate([e["obs"][k] for e in episodes]) for k in keys},
            {k: np.concatenate([e[k] for e in episodes]) for k in ("mask", "action", "logp", "value", "reward")})


def ppo_update(net, optimizer, episodes, c: ShopTrainConfig, train_policy=True):
    """One PPO update on finished episodes. train_policy=False: only the value head learns (the
    shared layers and the policy stay as they are), for the first updates after behaviour cloning."""
    import torch as th

    from .shop_net import masked_categorical

    obs, data = concat_episodes(episodes)
    advs, rets = [], []
    for e in episodes:
        a, r = gae(e["reward"] * c.reward_scale, e["value"], c.gamma, c.gae_lambda)
        advs.append(a)
        rets.append(r)
    adv, ret = np.concatenate(advs), np.concatenate(rets)
    n = len(adv)
    t_obs = {k: th.as_tensor(v) for k, v in obs.items()}
    t_mask = th.as_tensor(data["mask"])
    t_act, t_logp = th.as_tensor(data["action"]), th.as_tensor(data["logp"])
    t_adv, t_ret = th.as_tensor(adv), th.as_tensor(ret)
    t_value = th.as_tensor(data["value"])
    stats = {"pg_loss": [], "v_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    net.train()
    for _ in range(c.epochs):
        order = np.random.permutation(n)
        for start in range(0, n, c.minibatch):
            idx = th.as_tensor(order[start:start + c.minibatch])
            logits, value = net({k: v[idx] for k, v in t_obs.items()})
            dist = masked_categorical(logits, t_mask[idx])
            logp = dist.log_prob(t_act[idx])
            a = t_adv[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = th.exp(logp - t_logp[idx])
            pg = -th.min(ratio * a, th.clamp(ratio, 1 - c.clip, 1 + c.clip) * a).mean()
            v_clipped = t_value[idx] + th.clamp(value - t_value[idx], -c.clip * 10, c.clip * 10)
            v_loss = 0.5 * th.max((value - t_ret[idx]) ** 2, (v_clipped - t_ret[idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = pg + c.vf_coef * v_loss - c.ent_coef * entropy if train_policy else v_loss
            optimizer.zero_grad()
            loss.backward()
            if not train_policy:
                for name, param in net.named_parameters():
                    if not name.startswith("value_head."):
                        param.grad = None
            th.nn.utils.clip_grad_norm_(net.parameters(), c.max_grad_norm)
            optimizer.step()
            with th.no_grad():
                log_ratio = logp - t_logp[idx]
                stats["approx_kl"].append(float(((th.exp(log_ratio) - 1) - log_ratio).mean()))
                stats["clip_frac"].append(float(((ratio - 1).abs() > c.clip).float().mean()))
            stats["pg_loss"].append(pg.item())
            stats["v_loss"].append(v_loss.item())
            stats["entropy"].append(entropy.item())
    net.eval()
    out = {k: round(float(np.mean(v)), 5) for k, v in stats.items()}
    var = float(np.var(ret))
    out["explained_var"] = round(1 - float(np.var(ret - data["value"])) / var, 4) if var > 0 else None
    out["samples"] = n
    return out


def serve_actions(net, requests, greedy_net=None):
    """requests: [(rank, obs, mask, greedy)] -> [(rank, (action, logp, value))], one forward per net."""
    import torch as th

    from .shop_net import masked_categorical, obs_batch

    replies = []
    for use_greedy in (False, True):
        batch = [r for r in requests if r[3] == use_greedy]
        if not batch:
            continue
        model = greedy_net if (use_greedy and greedy_net is not None) else net
        with th.no_grad():
            logits, value = model(obs_batch([r[1] for r in batch]))
            dist = masked_categorical(logits, th.as_tensor(np.stack([r[2] for r in batch])))
            action = dist.probs.argmax(dim=1) if use_greedy else dist.sample()
            logp = dist.log_prob(action)
        for i, r in enumerate(batch):
            replies.append((r[0], (int(action[i]), float(logp[i]), float(value[i]))))
    return replies


def copy_net(net):
    clone = copy.deepcopy(net)
    clone.eval()
    return clone


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
