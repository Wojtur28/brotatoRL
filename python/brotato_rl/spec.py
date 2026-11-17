"""Run spec: a JSON-friendly description of the environment, shared by training and evaluation.

train.py saves it as spec.json next to the model; evaluate.py / play.py / the panel read it back,
so a model always plays in the environment it was trained in (same observation layout, action
type, frame_skip, reward and meta policy).

    {
      "character": "character_well_rounded", "weapon": null, "danger": 0, "zone": 0, "start_wave": 1,
      "frame_skip": 4, "action_type": "discrete", "dead_zone": 0.25, "max_episode_steps": 20000,
      "game_options": {"enemy_scaling": {...}, "wave_duration": 20, ...},
      "reward":  {RewardConfig fields},
      "encoder": {EncoderConfig fields},
      "meta":    {"type": "rule" | "random" | "first" | "learned", MetaConfig fields (rule),
                  "shop_spend_fraction" (random), "path" + "stochastic" (learned: a shop model, shop_net.py)}
    }
Missing keys take the defaults below.
"""
import copy
import json
from dataclasses import asdict, fields
from pathlib import Path

from .encoding import EncoderConfig
from .meta import MetaConfig, MetaPolicy, RandomMetaPolicy, RuleMetaPolicy
from .rewards import RewardConfig

DEFAULT_SPEC = {
    "character": "character_well_rounded",
    "weapon": None,
    "danger": 0,
    "zone": 0,
    "start_wave": 1,
    "frame_skip": 4,
    "action_type": "discrete",
    "dead_zone": 0.25,
    "max_episode_steps": None,
    "game_options": {},
    "reward": {},
    "encoder": {},
    "meta": {"type": "rule"},
}
SPEC_FILE = "spec.json"


def _dataclass(cls, values: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def normalize(spec: dict = None) -> dict:
    """Defaults filled in; reward/encoder/meta expanded to every field (self-describing files)."""
    out = copy.deepcopy(DEFAULT_SPEC)
    for key, value in (spec or {}).items():
        if key not in out:
            raise ValueError(f"unknown spec key {key!r}")
        out[key] = copy.deepcopy(value)
    out["reward"] = asdict(_dataclass(RewardConfig, out["reward"] or {}))
    out["encoder"] = asdict(_dataclass(EncoderConfig, out["encoder"] or {}))
    meta = dict(out["meta"] or {})
    kind = meta.pop("type", "rule")
    if kind == "rule":
        out["meta"] = {"type": "rule", **asdict(_dataclass(MetaConfig, meta))}
    elif kind == "random":
        out["meta"] = {"type": "random", "shop_spend_fraction": float(meta.get("shop_spend_fraction", 1.0))}
    elif kind == "first":
        out["meta"] = {"type": "first"}
    elif kind == "learned":
        if not meta.get("path"):
            raise ValueError('meta type "learned" needs "path" (a shop model .pt, see brotato_rl/shop_net.py)')
        out["meta"] = {"type": "learned", "path": str(meta["path"]), "stochastic": bool(meta.get("stochastic", False))}
    else:
        raise ValueError(f"meta type must be rule, random, first or learned, not {kind!r}")
    return out


def build_meta(meta: dict) -> MetaPolicy:
    meta = dict(meta or {"type": "rule"})
    kind = meta.pop("type", "rule")
    if kind == "random":
        return RandomMetaPolicy(shop_spend_fraction=float(meta.get("shop_spend_fraction", 1.0)))
    if kind == "first":
        return MetaPolicy()
    if kind == "learned":
        from .shop_net import LearnedMetaPolicy  # torch only when a learned shop is used
        return LearnedMetaPolicy(meta["path"], stochastic=bool(meta.get("stochastic", False)))
    return RuleMetaPolicy(_dataclass(MetaConfig, meta))


def env_kwargs(spec: dict) -> dict:
    """BrotatoEnv keyword arguments for a spec (rendering/instance options are added by the caller)."""
    s = normalize(spec)
    return {
        "character": s["character"], "weapon": s["weapon"], "danger": int(s["danger"]), "zone": int(s["zone"]),
        "start_wave": int(s["start_wave"]), "frame_skip": int(s["frame_skip"]), "action_type": s["action_type"],
        "dead_zone": float(s["dead_zone"]), "max_episode_steps": s["max_episode_steps"],
        "game_options": dict(s["game_options"]),
        "reward_config": RewardConfig(**s["reward"]),
        "encoder_config": EncoderConfig(**s["encoder"]),
        "meta_policy": build_meta(s["meta"]),
    }


def save(spec: dict, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(normalize(spec), indent=1), encoding="utf-8")
    return path


def load(path) -> dict:
    return normalize(json.loads(Path(path).read_text(encoding="utf-8")))


def find_for_model(model_path) -> Path | None:
    """spec.json in the model's folder or the one above (runs/<name>/best/model.zip -> runs/<name>/spec.json)."""
    model_path = Path(model_path).resolve()
    for folder in (model_path.parent, model_path.parent.parent):
        candidate = folder / SPEC_FILE
        if candidate.is_file():
            return candidate
    return None
