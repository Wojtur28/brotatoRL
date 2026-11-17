"""Behaviour cloning warm start: train the PPO network on demonstrations before PPO starts.

PPO from scratch spends millions of steps rediscovering what HeuristicBot already does. Here the
policy head learns to imitate the bot (cross-entropy on its actions) and the value head learns the
bot's discounted returns, in the same units PPO will use: VecNormalize's return statistics are
filled from the demonstrations first, and the value targets are divided by the same scale.

Config (train.bc in the training config):
    {"demos": "demos/bot_ppo3", "epochs": 4, "batch_size": 1024, "learning_rate": 3e-4,
     "ent_coef": 0.003, "vf_coef": 0.5, "holdout": 0.1, "evaluate": true}
A small entropy bonus keeps the cloned policy from becoming fully deterministic, so PPO can still
explore.
"""
import time
from pathlib import Path

import numpy as np

from . import paths


def load_demos(folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = paths.ROOT / folder
    files = sorted(folder.glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"no demonstrations (ep_*.npz) in {folder}; run scripts/collect_demos.py")
    obs, actions, rewards, ends = {}, [], [], []
    for f in files:
        with np.load(f) as d:
            for k in d.files:
                if k.startswith("obs_"):
                    obs.setdefault(k[4:], []).append(d[k])
            actions.append(d["actions"].astype(np.int64))
            rewards.append(d["rewards"].astype(np.float64))
            ends.append(bool(d["terminated"]))
    return {k: np.concatenate(v) for k, v in obs.items()}, actions, rewards, ends, len(files)


def discounted_returns(rewards, gamma, terminated):
    """Monte Carlo return of every step of one episode. Bot episodes end in death or victory
    (terminated); a truncated one would need a bootstrap value, which a demonstration lacks, so
    its last steps are slightly underestimated."""
    out = np.zeros(len(rewards))
    g = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        g = rewards[i] + gamma * g
        out[i] = g
    return out


def pretrain(model, vec_normalize, cfg: dict, gamma: float, log=print) -> dict:
    import torch as th
    import torch.nn.functional as F

    start = time.time()
    obs, actions, rewards, ends, n_eps = load_demos(cfg["demos"])
    returns = np.concatenate([discounted_returns(r, gamma, e) for r, e in zip(rewards, ends)])
    acts = np.concatenate(actions)
    scale = 1.0
    if vec_normalize is not None and vec_normalize.norm_reward:
        # The same running statistic VecNormalize keeps: forward-discounted reward sums.
        for r in rewards:
            vec_normalize.ret_rms.update(_forward_returns(r, gamma))
        scale = float(np.sqrt(vec_normalize.ret_rms.var + vec_normalize.epsilon))
    targets = (returns / scale).astype(np.float32)
    n = len(acts)
    expected = set(model.observation_space.spaces)
    if set(obs) != expected:
        raise ValueError(f"demonstration observations {sorted(obs)} do not match the model {sorted(expected)}; "
                         "record them with the same config")

    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    n_hold = int(n * float(cfg.get("holdout", 0.1)))
    hold, train = order[:n_hold], order[n_hold:]
    batch = int(cfg.get("batch_size", 1024))
    ent_coef, vf_coef = float(cfg.get("ent_coef", 0.003)), float(cfg.get("vf_coef", 0.5))
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=float(cfg.get("learning_rate", 3e-4)))

    def tensors(idx):
        o = {k: th.as_tensor(v[idx].astype(np.float32)) for k, v in obs.items()}
        return o, th.as_tensor(acts[idx]), th.as_tensor(targets[idx])

    def holdout_metrics():
        policy.set_training_mode(False)
        correct, vals, tgts = 0, [], []
        with th.no_grad():
            for i in range(0, len(hold), batch):
                o, a, t = tensors(hold[i:i + batch])
                dist = policy.get_distribution(o)
                correct += int((dist.distribution.probs.argmax(dim=1) == a).sum())
                vals.append(policy.predict_values(o).flatten().numpy())
                tgts.append(t.numpy())
        v, t = np.concatenate(vals), np.concatenate(tgts)
        ev = 1.0 - np.var(t - v) / max(np.var(t), 1e-8)
        return correct / max(1, len(hold)), float(ev)

    history = []
    for epoch in range(int(cfg.get("epochs", 4))):
        policy.set_training_mode(True)
        rng.shuffle(train)
        losses, entropies = [], []
        for i in range(0, len(train), batch):
            o, a, t = tensors(train[i:i + batch])
            values, log_prob, entropy = policy.evaluate_actions(o, a)
            loss = -log_prob.mean() - ent_coef * entropy.mean() + vf_coef * F.mse_loss(values.flatten(), t)
            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss))
            entropies.append(float(entropy.mean()))
        acc, ev = holdout_metrics()
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4),
                        "entropy": round(float(np.mean(entropies)), 4), "holdout_accuracy": round(acc, 4),
                        "holdout_value_explained_variance": round(ev, 4)})
        log(f"bc epoch {epoch + 1}: loss {history[-1]['loss']}, entropy {history[-1]['entropy']}, "
            f"bot-action accuracy {acc:.1%}, value explained variance {ev:.2f}")
    policy.set_training_mode(False)
    return {"demos": str(cfg["demos"]), "episodes": n_eps, "steps": n, "reward_scale": scale,
            "seconds": round(time.time() - start, 1), "history": history}


def _forward_returns(rewards, gamma):
    out = np.zeros(len(rewards))
    g = 0.0
    for i, r in enumerate(rewards):
        g = g * gamma + r
        out[i] = g
    return out
