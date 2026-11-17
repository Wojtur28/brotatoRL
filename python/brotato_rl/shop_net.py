"""The shop agent's network (torch) and a MetaPolicy that plays with it.

ShopNet scores every offer and every owned weapon with the same small network (an offer's logits do
not depend on its slot), so "buy the knife" is learned once, not four times. The context (wave,
materials, stats, owned items and weapons) feeds every score. Impossible actions are masked out.

    net = ShopNet(vocab.size, len(vocab.items))
    logits, value = net(batch_obs)                 # batch_obs: dict of tensors (see obs_batch)
    dist = masked_categorical(logits, mask)
"""
from pathlib import Path

import numpy as np
import torch as th
from torch import nn

from .meta import MetaPolicy
from .shop import (GLOB_DIM, N_ACTIONS, N_OFFERS, N_WEAPONS, OFFER_DIM, WEAPON_DIM, ShopConfig, ShopEncoder,
                   Vocab)


def _mlp(*sizes, act=nn.ReLU):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])


class ShopNet(nn.Module):
    def __init__(self, vocab_size: int, n_items: int, emb: int = 16, hidden: int = 128):
        super().__init__()
        self.kwargs = {"vocab_size": vocab_size, "n_items": n_items, "emb": emb, "hidden": hidden}
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.offer = _mlp(OFFER_DIM + emb, 64, 64)
        self.weapon = _mlp(WEAPON_DIM + emb, 32, 32)
        self.items = nn.Linear(n_items, 64)
        self.glob = _mlp(GLOB_DIM, 64)
        self.trunk = nn.Sequential(_mlp(64 + 2 * 64 + 2 * 32 + 64, hidden, hidden), nn.ReLU())
        self.offer_head = _mlp(64 + hidden, 64, 2)    # pick, lock
        self.weapon_head = _mlp(32 + hidden, 32, 2)   # combine, recycle
        self.misc_head = nn.Linear(hidden, 3)         # reroll, go, take
        self.value_head = _mlp(hidden, 64, 1)

    @staticmethod
    def _pool(h, present):
        mask = present.unsqueeze(-1)
        total = (h * mask).sum(dim=1)
        top = h.masked_fill(mask == 0, -1e4).max(dim=1).values
        return th.cat([total, th.where(present.sum(dim=1, keepdim=True) > 0, top, th.zeros_like(top))], dim=1)

    def forward(self, obs: dict):
        offers, weapons = obs["offers"], obs["weapons"]
        o = th.relu(self.offer(th.cat([offers, self.embed(obs["offer_id"].long())], dim=-1)))
        w = th.relu(self.weapon(th.cat([weapons, self.embed(obs["weapon_id"].long())], dim=-1)))
        ctx = th.cat([th.relu(self.glob(obs["glob"])), self._pool(o, offers[..., 0]), self._pool(w, weapons[..., 0]),
                      th.relu(self.items(obs["items"]))], dim=1)
        h = self.trunk(ctx)
        batch = h.shape[0]
        per_offer = self.offer_head(th.cat([o, h.unsqueeze(1).expand(-1, N_OFFERS, -1)], dim=-1))
        per_weapon = self.weapon_head(th.cat([w, h.unsqueeze(1).expand(-1, N_WEAPONS, -1)], dim=-1))
        misc = self.misc_head(h)
        logits = th.cat([per_offer[..., 0], misc, per_offer[..., 1], per_weapon[..., 0], per_weapon[..., 1]], dim=1)
        assert logits.shape == (batch, N_ACTIONS)
        return logits, self.value_head(h).squeeze(-1)


def masked_categorical(logits, mask):
    return th.distributions.Categorical(logits=logits.masked_fill(~mask, -1e9))


def obs_batch(observations, device="cpu") -> dict:
    keys = observations[0].keys()
    return {k: th.as_tensor(np.stack([o[k] for o in observations]), device=device) for k in keys}


def save_shop_model(path, net: ShopNet, vocab: Vocab, config: ShopConfig, extra: dict = None) -> Path:
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save({**(extra or {}), "state": net.state_dict(), "kwargs": net.kwargs, "catalog": vocab.catalog,
             "shop_config": asdict(config)}, path)
    return path


def load_shop_model(path):
    data = th.load(str(path), map_location="cpu", weights_only=False)
    net = ShopNet(**data["kwargs"])
    net.load_state_dict(data["state"])
    net.eval()
    return net, Vocab(data["catalog"]), ShopConfig(**data["shop_config"]), data


class LearnedMetaPolicy(MetaPolicy):
    """Plays the between-wave decisions with a trained ShopNet (greedy unless stochastic=True)."""

    def __init__(self, path, stochastic=False):
        self.path = str(path)
        self.stochastic = stochastic
        self._net = None
        self.rng = np.random.default_rng()

    def __getstate__(self):  # picklable for subprocess workers: the model is loaded where it is used
        return {"path": self.path, "stochastic": self.stochastic, "_net": None, "rng": self.rng}

    def _decide(self, d):
        if self._net is None:
            net, vocab, config, _ = load_shop_model(self.path)
            self._net, self._encoder = net, ShopEncoder(vocab, config)
        mask = self._encoder.mask(d)
        with th.no_grad():
            logits, _ = self._net(obs_batch([self._encoder.encode(d)]))
            dist = masked_categorical(logits, th.as_tensor(mask).unsqueeze(0))
            action = int(dist.sample()[0]) if self.stochastic else int(dist.probs[0].argmax())
        return self._encoder.reply(d, action)

    def level_up(self, d):
        return self._decide(d)

    def item(self, d):
        return self._decide(d)

    def shop(self, d):
        return self._decide(d)
