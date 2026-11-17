"""Between-wave decisions (shop, level-ups, crates) as fixed-size observations and one action space.

The shop agent sees one decision at a time (a "decision" message from the mod, 0.5.0+ for "build").
Observation (dict of float32 arrays, see ShopEncoder):
  glob     (GLOB_DIM,)            decision kind, wave, materials, level, stats, slots, prices
  offers   (4, OFFER_DIM)         shop items / level-up options / the crate item in slot 0
  offer_id (4,)                   vocabulary index of each offer (0 = empty, 1 = unknown id)
  weapons  (6, WEAPON_DIM)        owned weapons in slot order
  weapon_id (6,)
  items    (n_items,)             owned items as counts over the item vocabulary (/ 4)
Action (Discrete(N_ACTIONS)), always used with a mask of the actions that are possible now:
  0-3   PICK i      shop: buy offer i; level-up: choose option i
  4     REROLL      shop or level-up reroll
  5     GO          shop: start the next wave; crate: discard (recycle) the item
  6     TAKE        crate: take the item
  7-10  LOCK i      shop: lock offer i (unlock if it is locked)
  11-16 COMBINE j   shop: combine owned weapon j
  17-22 RECYCLE j   shop: recycle owned weapon j
Ban (Abyssal Terrors ban mode) is not used.
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .meta import STAT_KEYS, MetaPolicy

N_OFFERS, N_WEAPONS = 4, 6
PICK, REROLL, GO, TAKE, LOCK, COMBINE, RECYCLE = 0, 4, 5, 6, 7, 11, 17
N_ACTIONS = 23
KINDS = ("shop", "level_up", "item")
ACTION_NAMES = tuple([f"pick{i}" for i in range(4)] + ["reroll", "go", "take"] + [f"lock{i}" for i in range(4)]
                     + [f"combine{j}" for j in range(6)] + [f"recycle{j}" for j in range(6)])
STAT_SCALE = {"range": 100.0}  # other stats / 20
GLOB_DIM = 3 + 9 + len(STAT_KEYS)
OFFER_DIM = 1 + 3 + 4 + 6 + len(STAT_KEYS) + 1
WEAPON_DIM = 1 + 4 + 2


@dataclass
class ShopConfig:
    weapon_type: str = "melee"  # weapons the agent may buy: "melee" | "ranged" | "any"
    max_weapons: int = 6
    allow_lock: bool = True
    allow_recycle: bool = True


class Vocab:
    """Fixed id -> index tables built from the catalog the mod sends in hello (mod 0.5.0+).

    0 = empty slot, 1 = an id missing from the catalog; items keep their own table for the
    owned-items counts. Saved with a model so it never changes under it.
    """

    def __init__(self, catalog: dict):
        self.catalog = catalog
        ids = sorted({row[0] for key in ("items", "weapons", "upgrades") for row in catalog.get(key, [])})
        self.index = {name: i + 2 for i, name in enumerate(ids)}
        self.items = {name: i for i, name in enumerate(sorted({row[0] for row in catalog.get("items", [])}))}

    @property
    def size(self) -> int:
        return len(self.index) + 2

    def id(self, name) -> int:
        return self.index.get(name, 1) if name else 0

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.catalog), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def _stats_vector(effects) -> np.ndarray:
    out = np.zeros(len(STAT_KEYS) + 1, np.float32)  # last column: number of non-stat effects
    for key, value in effects or []:
        if isinstance(key, str) and key.startswith("stat_") and key[5:] in STAT_KEYS and isinstance(value, (int, float)):
            out[STAT_KEYS.index(key[5:])] += value / STAT_SCALE.get(key[5:], 10.0)
        else:
            out[-1] += 0.2
    return out


class ShopEncoder:
    def __init__(self, vocab: Vocab, config: ShopConfig = None):
        self.vocab = vocab
        self.config = config or ShopConfig()

    def space(self):
        from gymnasium import spaces
        box = lambda *shape: spaces.Box(-10.0, 10.0, shape, np.float32)
        return spaces.Dict({
            "glob": box(GLOB_DIM), "offers": box(N_OFFERS, OFFER_DIM), "offer_id": box(N_OFFERS),
            "weapons": box(N_WEAPONS, WEAPON_DIM), "weapon_id": box(N_WEAPONS), "items": box(len(self.vocab.items)),
        })

    # -- observation --------------------------------------------------------------------

    def encode(self, d: dict) -> dict:
        kind = d.get("kind")
        build = d.get("build") or {}
        materials = float(d.get("materials", 0))
        stats = build.get("stats") or {}

        glob = np.zeros(GLOB_DIM, np.float32)
        glob[KINDS.index(kind)] = 1.0
        reroll = float(d.get("reroll_price", 0) or 0)
        glob[3:12] = (
            float(d.get("wave", 1)) / 20.0,
            min(materials / 100.0, 10.0),
            np.log1p(materials) / 6.0,
            float(build.get("level", 0)) / 20.0,
            float(build.get("max_hp", 10)) / 100.0,
            float(build.get("free_weapon_slots", 0)) / 6.0,
            min(reroll / max(1.0, materials), 5.0) if kind != "item" else float(d.get("recycle_value", 0)) / 50.0,
            float(reroll <= materials) if kind != "item" else 0.0,
            min(float(d.get("actions_taken", 0)) / 20.0, 3.0),
        )
        for i, key in enumerate(STAT_KEYS):
            glob[12 + i] = np.clip(float(stats.get(key, 0.0)) / STAT_SCALE.get(key, 20.0), -10, 10)

        offers = np.zeros((N_OFFERS, OFFER_DIM), np.float32)
        offer_id = np.zeros(N_OFFERS, np.float32)
        owned_weapons = [w[0] for w in build.get("weapons") or []]
        for slot, offer in self._offers(d):
            row = offers[slot]
            category = 2 if kind == "level_up" else (1 if offer.get("category") == "weapon" else 0)
            price = float(offer.get("price", 0) or 0)
            row[0] = 1.0
            row[1 + category] = 1.0
            row[4 + min(3, max(0, int(offer.get("tier", 0))))] = 1.0
            row[8:14] = (
                min(price / max(1.0, materials), 5.0),
                float(price <= materials),
                price / 50.0,
                float(bool(offer.get("locked"))),
                float(offer.get("weapon_type") == "melee"),
                float(owned_weapons.count(offer.get("id"))) / 2.0,
            )
            row[14:] = _stats_vector(offer.get("effects"))
            offer_id[slot] = self.vocab.id(offer.get("id"))

        weapons = np.zeros((N_WEAPONS, WEAPON_DIM), np.float32)
        weapon_id = np.zeros(N_WEAPONS, np.float32)
        combinable = {w.get("index"): bool(w.get("can_combine")) for w in d.get("owned_weapons") or []}
        for j, w in enumerate((build.get("weapons") or [])[:N_WEAPONS]):
            weapons[j, 0] = 1.0
            weapons[j, 1 + min(3, max(0, int(w[1])))] = 1.0
            weapons[j, 5] = float(w[2] == "melee")
            weapons[j, 6] = float(combinable.get(j, False))
            weapon_id[j] = self.vocab.id(w[0])

        items = np.zeros(len(self.vocab.items), np.float32)
        for name, _tier in build.get("items") or []:
            i = self.vocab.items.get(name)
            if i is not None:
                items[i] = min(items[i] + 0.25, 10.0)
        return {"glob": glob, "offers": offers, "offer_id": offer_id, "weapons": weapons,
                "weapon_id": weapon_id, "items": items}

    @staticmethod
    def _offers(d):
        kind = d.get("kind")
        if kind == "shop":
            return [(int(it["index"]), it) for it in d.get("items") or [] if 0 <= int(it.get("index", -1)) < N_OFFERS]
        if kind == "level_up":
            return list(enumerate((d.get("options") or [])[:N_OFFERS]))
        return [(0, d.get("item") or {})]

    # -- actions ------------------------------------------------------------------------

    def mask(self, d: dict) -> np.ndarray:
        m = np.zeros(N_ACTIONS, bool)
        kind = d.get("kind")
        materials = d.get("materials", 0)
        if kind == "item":
            m[GO] = m[TAKE] = True
            return m
        if kind == "level_up":
            m[PICK:PICK + min(N_OFFERS, len(d.get("options") or []))] = True
            m[REROLL] = d.get("reroll_price", 0) <= materials
            if not m[PICK:PICK + N_OFFERS].any():
                m[PICK] = True  # the mod falls back to option 0
            return m
        m[GO] = True
        m[REROLL] = d.get("reroll_price", 0) <= materials
        owned = d.get("owned_weapons") or []
        for slot, it in self._offers(d):
            if not it.get("active"):
                continue
            m[PICK + slot] = it["price"] <= materials and (it["category"] == "item" or self._weapon_ok(d, it))
            m[LOCK + slot] = self.config.allow_lock
        if not d.get("weapons_locked"):
            for w in owned[:N_WEAPONS]:
                j = int(w.get("index", -1))
                if 0 <= j < N_WEAPONS:
                    m[COMBINE + j] = bool(w.get("can_combine"))
                    m[RECYCLE + j] = self.config.allow_recycle and len(owned) > 1
        return m

    def _weapon_ok(self, d, it) -> bool:
        c = self.config
        if c.weapon_type != "any" and it.get("weapon_type", c.weapon_type) != c.weapon_type:
            return False
        owned_ids = [w["id"] for w in d.get("owned_weapons") or []] or d.get("weapons", [])
        free_slots = d.get("free_weapon_slots", 0)
        if it["id"] in owned_ids:  # with full slots the game merges it into the owned copy
            return free_slots > 0 or it.get("tier", 0) < 3
        return free_slots > 0 and len(owned_ids) < c.max_weapons

    @staticmethod
    def reply(d: dict, action: int) -> dict:
        kind = d.get("kind")
        if kind == "item":
            return {"action": "take"} if action == TAKE else {"action": "discard"}
        if kind == "level_up":
            if action == REROLL:
                return {"action": "reroll"}
            return {"action": "choose", "index": action - PICK if PICK <= action < PICK + N_OFFERS else 0}
        if PICK <= action < PICK + N_OFFERS:
            return {"action": "buy", "index": action - PICK}
        if action == REROLL:
            return {"action": "reroll"}
        if LOCK <= action < LOCK + N_OFFERS:
            slot = action - LOCK
            locked = any(int(it.get("index", -1)) == slot and it.get("locked") for it in d.get("items") or [])
            return {"action": "unlock" if locked else "lock", "index": slot}
        if COMBINE <= action < COMBINE + N_WEAPONS:
            return {"action": "combine", "weapon_index": action - COMBINE}
        if RECYCLE <= action < RECYCLE + N_WEAPONS:
            return {"action": "recycle", "weapon_index": action - RECYCLE}
        return {"action": "go"}

    @staticmethod
    def action_of(d: dict, reply: dict):
        """The action index of a MetaPolicy reply (behaviour cloning), None if it has none (ban)."""
        kind, action = d.get("kind"), reply.get("action")
        if kind == "item":
            return {"take": TAKE, "discard": GO}.get(action)
        if kind == "level_up":
            return REROLL if action == "reroll" else PICK + int(reply.get("index", 0))
        if action == "buy":
            return PICK + int(reply["index"])
        if action in ("lock", "unlock"):
            return LOCK + int(reply["index"])
        if action == "combine":
            return COMBINE + int(reply["weapon_index"])
        if action == "recycle":
            return RECYCLE + int(reply["weapon_index"])
        return {"reroll": REROLL, "go": GO}.get(action)


def config_dict(config: ShopConfig) -> dict:
    return asdict(config)


class RecordingMetaPolicy(MetaPolicy):
    """Wraps a MetaPolicy and records (observation, mask, action) of every decision it makes."""

    def __init__(self, inner: MetaPolicy, encoder: ShopEncoder):
        self.inner, self.encoder = inner, encoder
        self.records = []

    def seed(self, rng):
        self.inner.seed(rng)

    def _record(self, d, reply):
        action = self.encoder.action_of(d, reply)
        mask = self.encoder.mask(d)
        if action is not None and mask[action]:
            self.records.append((self.encoder.encode(d), mask, action))
        return reply

    def level_up(self, d):
        return self._record(d, self.inner.level_up(d))

    def item(self, d):
        return self._record(d, self.inner.item(d))

    def shop(self, d):
        return self._record(d, self.inner.shop(d))
