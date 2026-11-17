"""Decisions between waves: level-up upgrades, crate items and the shop.

The movement agent never sees these as steps. The environment asks a MetaPolicy and
records what it chose in info["decisions"]. Subclass MetaPolicy (or pass any object with
the same three methods) to plug in something smarter later.

Replies use the bridge vocabulary:
  level_up -> {"action": "choose", "index": i} | {"action": "reroll"}
  item     -> {"action": "take"} | {"action": "discard"} | {"action": "ban"}
  shop     -> {"action": "buy" | "lock" | "unlock" | "ban", "index": i}
              | {"action": "recycle" | "combine", "weapon_index": j}   (j = index in owned_weapons)
              | {"action": "reroll"} | {"action": "go"}
"ban" needs ban mode (Abyssal Terrors) and tokens: see decision["ban"] = {"active", "tokens"}.
"""
from dataclasses import dataclass, field

import numpy as np

STAT_KEYS = ("max_hp", "hp_regeneration", "lifesteal", "percent_damage", "melee_damage", "ranged_damage",
             "elemental_damage", "attack_speed", "crit_chance", "engineering", "range", "armor", "dodge",
             "speed", "luck", "harvesting")


class MetaPolicy:
    def seed(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def level_up(self, decision: dict) -> dict:
        return {"action": "choose", "index": 0}

    def item(self, decision: dict) -> dict:
        return {"action": "take"}

    def shop(self, decision: dict) -> dict:
        return {"action": "go"}


class RandomMetaPolicy(MetaPolicy):
    """Random upgrade, always take crate items, buy random affordable things, never reroll."""

    def __init__(self, shop_spend_fraction: float = 1.0):
        self.shop_spend_fraction = shop_spend_fraction
        self.rng = np.random.default_rng()
        self._budget = None

    def level_up(self, decision):
        return {"action": "choose", "index": int(self.rng.integers(len(decision.get("options") or [0])))}

    def shop(self, decision):
        # Budget = how much this shop visit may still spend.
        if decision.get("actions_taken", 0) == 0 or self._budget is None:
            self._budget = decision["materials"] * self.shop_spend_fraction
        limit = min(decision["materials"], self._budget)
        affordable = [
            item for item in decision.get("items", [])
            if item["active"] and item["price"] <= limit
            and (item["category"] == "item" or decision.get("free_weapon_slots", 0) > 0
                 or item["id"] in decision.get("weapons", []))
        ]
        if not affordable:
            return {"action": "go"}
        choice = affordable[int(self.rng.integers(len(affordable)))]
        self._budget -= choice["price"]
        return {"action": "buy", "index": choice["index"]}


@dataclass
class MetaConfig:
    """Parameters of RuleMetaPolicy. Every between-wave mechanic has its own group."""
    # Item score = tier_weight * (tier + 1) + sum(stat_weights[stat] * value) - price_weight * price
    #              (+ weapon_bonus for weapons, + other_effect_weight per non-stat effect)
    stat_weights: dict = field(default_factory=lambda: {k: 1.0 for k in STAT_KEYS})
    tier_weight: float = 1.0
    price_weight: float = 0.0
    weapon_bonus: float = 2.0
    other_effect_weight: float = 0.0

    # Level-up: "random" | "first" | "score"
    level_up_strategy: str = "score"
    level_up_max_rerolls: int = 0          # rerolls per level-up
    level_up_reroll_below: float = 0.0     # reroll while the best score is below this

    # Crate items: "take" | "discard" | "score"
    item_mode: str = "take"
    item_min_tier: int = 0
    item_min_score: float = -1e9
    item_ban_rejected: bool = False        # ban instead of recycle (needs ban mode and tokens)

    # Shop: "random" | "score" | "none" (leave at once)
    shop_strategy: str = "score"
    spend_fraction: float = 1.0            # share of the materials one visit may spend
    keep_materials: int = 0                # never go below this many materials
    shop_min_score: float = -1e9
    buy_weapons: bool = True
    weapon_type: str = "any"               # "any" | "melee" | "ranged"
    max_weapons: int = 6
    auto_combine: bool = True              # combine two identical weapons when possible
    recycle_below_tier: int = -1           # recycle owned weapons below this tier (-1 = never)
    max_rerolls: int = 0                   # shop rerolls per visit
    reroll_max_price: int = 10
    lock_unaffordable: bool = False        # lock the best item that is too expensive now
    lock_min_tier: int = 2
    ban_ids: list = field(default_factory=list)  # shop item ids to ban on sight


class RuleMetaPolicy(MetaPolicy):
    """Configurable, deterministic-by-default meta policy (see MetaConfig)."""

    def __init__(self, config: MetaConfig = None, **overrides):
        self.config = config or MetaConfig()
        for key, value in overrides.items():
            if not hasattr(self.config, key):
                raise TypeError(f"unknown MetaConfig field {key!r}")
            setattr(self.config, key, value)
        self.rng = np.random.default_rng()
        self._level_key, self._level_rerolls = None, 0
        self._visit = {}

    # -- scoring ------------------------------------------------------------------------

    def score(self, item: dict) -> float:
        c = self.config
        s = c.tier_weight * (int(item.get("tier", 0)) + 1) - c.price_weight * float(item.get("price", 0))
        if item.get("category") == "weapon":
            s += c.weapon_bonus
        for key, value in item.get("effects") or []:
            if isinstance(key, str) and key.startswith("stat_") and isinstance(value, (int, float)):
                s += float(c.stat_weights.get(key[5:], 0.0)) * value
            else:
                s += c.other_effect_weight
        return s

    # -- level-up -----------------------------------------------------------------------

    def level_up(self, d):
        c, options = self.config, d.get("options") or []
        if not options:
            return {"action": "choose", "index": 0}
        key = (d.get("wave"), d.get("level"))
        if key != self._level_key:
            self._level_key, self._level_rerolls = key, 0
        if c.level_up_strategy == "random":
            return {"action": "choose", "index": int(self.rng.integers(len(options)))}
        if c.level_up_strategy == "first":
            return {"action": "choose", "index": 0}
        scores = [self.score(o) for o in options]
        best = int(np.argmax(scores))
        if (scores[best] < c.level_up_reroll_below and self._level_rerolls < c.level_up_max_rerolls
                and d.get("reroll_price", 0) <= d.get("materials", 0)):
            self._level_rerolls += 1
            return {"action": "reroll"}
        return {"action": "choose", "index": best}

    # -- crates -------------------------------------------------------------------------

    def item(self, d):
        c, item = self.config, d.get("item") or {}
        if c.item_mode == "take":
            return {"action": "take"}
        keep = c.item_mode == "score" and item.get("tier", 0) >= c.item_min_tier and self.score(item) >= c.item_min_score
        if keep:
            return {"action": "take"}
        if c.item_ban_rejected and (d.get("ban") or {}).get("active"):
            return {"action": "ban"}
        return {"action": "discard"}

    # -- shop ---------------------------------------------------------------------------

    def shop(self, d):
        c = self.config
        if c.shop_strategy == "none":
            return {"action": "go"}
        materials = d.get("materials", 0)
        if d.get("actions_taken", 0) == 0 or not self._visit:
            self._visit = {"budget": materials * c.spend_fraction, "rerolls": 0, "locked": set(), "banned": set()}
        v = self._visit
        items = [i for i in d.get("items", []) if i.get("active")]
        owned = d.get("owned_weapons") or []
        ban = d.get("ban") or {}

        # 1. Ban unwanted items.
        if ban.get("active") and c.ban_ids:
            for it in items:
                if it["id"] in c.ban_ids and it["category"] == "item" and it["id"] not in v["banned"]:
                    v["banned"].add(it["id"])
                    return {"action": "ban", "index": it["index"]}

        # 2. Weapon maintenance (not possible when an effect locks the current weapons).
        if not d.get("weapons_locked"):
            if c.auto_combine:
                for w in owned:
                    if w.get("can_combine"):
                        return {"action": "combine", "weapon_index": w["index"]}
            if c.recycle_below_tier >= 0 and len(owned) > 1:
                for w in owned:
                    if w.get("tier", 0) < c.recycle_below_tier:
                        return {"action": "recycle", "weapon_index": w["index"]}

        # 3. Buy.
        limit = min(materials - c.keep_materials, v["budget"])
        owned_ids = [w["id"] for w in owned] or d.get("weapons", [])
        free_slots = d.get("free_weapon_slots", 0)

        def weapon_ok(it):
            if not c.buy_weapons:
                return False
            if c.weapon_type != "any" and it.get("weapon_type", c.weapon_type) != c.weapon_type:
                return False
            if it["id"] in owned_ids:  # with full slots the game merges it into the owned copy
                return free_slots > 0 or it.get("tier", 0) < 3  # tier IV (3) has nothing to merge into
            return free_slots > 0 and len(owned_ids) < c.max_weapons

        candidates = [it for it in items
                      if it["price"] <= limit and (it["category"] == "item" or weapon_ok(it))
                      and self.score(it) >= c.shop_min_score]
        if candidates:
            if c.shop_strategy == "random":
                choice = candidates[int(self.rng.integers(len(candidates)))]
            else:
                choice = max(candidates, key=self.score)
            v["budget"] -= choice["price"]
            return {"action": "buy", "index": choice["index"]}

        # 4. Lock the best item we cannot afford yet.
        if c.lock_unaffordable:
            lockable = [it for it in items if not it.get("locked") and it["tier"] >= c.lock_min_tier
                        and it["price"] > limit and it["id"] not in v["locked"]
                        and (it["category"] == "item" or weapon_ok(it))]
            if lockable:
                best = max(lockable, key=self.score)
                v["locked"].add(best["id"])
                return {"action": "lock", "index": best["index"]}

        # 5. Reroll.
        price = d.get("reroll_price", 0)
        if v["rerolls"] < c.max_rerolls and price <= c.reroll_max_price and materials - price >= c.keep_materials:
            v["rerolls"] += 1
            v["budget"] -= price
            return {"action": "reroll"}
        return {"action": "go"}
