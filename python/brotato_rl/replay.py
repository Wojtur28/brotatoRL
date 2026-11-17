"""Replays: a compact record of one episode, for viewer/replay.html (no game needed to watch).

The game is not deterministic after the start, so a run cannot be re-simulated from its seed and
actions. A replay therefore stores what the agent saw at every step: positions of the character,
enemies (with attack kind and wind-up/dash state), projectiles, spawn markers, materials and
crates, plus HP, wave, the action and the decisions made between waves.

    rec = ReplayRecorder(meta={"title": "ppo1 @ 2.5M", "seed": 10001})
    obs, info = env.reset(seed=10001); rec.add(env.raw_state, None, info)
    ... rec.add(env.raw_state, action, info) after every step ...
    rec.save("replays/ppo1_2500k_seed10001.json.gz")

File: gzip JSON {"version": 1, "meta": {...}, "frames": [frame, ...]}, one frame per step:
  t frame (1/60 s), w wave, tl wave time left (s), p [x, y, hp, max_hp], a action (int, or [x, y]),
  lv level, m materials, e [x, y, kind, attack class, attack state, hp %] per enemy (flat),
  pr [x, y] per projectile, b [x, y] per enemy spawn marker, g [x, y] per material on the ground,
  c [x, y, kind] per consumable, d decisions made just before this frame (list), z arena (on change).
"""
import gzip
import json
from pathlib import Path

import numpy as np

REPLAY_VERSION = 1


def _ints(values):
    return [int(round(v)) for v in values]


class ReplayRecorder:
    def __init__(self, meta: dict = None):
        self.meta = dict(meta or {})
        self.frames = []
        self._zone = None

    def add(self, state: dict, action=None, info: dict = None) -> None:
        state = state or {}
        player = state.get("player") or {}
        if not player.get("pos"):
            return  # end state without a scene (or between waves): nothing to draw
        frame = {
            "t": int(state.get("frame", 0)),
            "w": int(state.get("wave", 0)),
            "tl": round(float(state.get("wave_time_left", 0.0)), 1),
            "p": _ints(player["pos"]) + [int(player.get("hp", 0)), int(player.get("max_hp", 0))],
            "lv": int(player.get("level", 0)),
            "m": int(player.get("materials", 0)),
        }
        if action is not None:
            a = np.asarray(action).reshape(-1)
            frame["a"] = int(a[0]) if a.size == 1 else [round(float(a[0]), 2), round(float(a[1]), 2)]
        zone = state.get("zone")
        if zone and zone != self._zone:
            frame["z"] = _ints(zone)
            self._zone = zone
        cols = int(state.get("enemy_cols", 8))
        enemies = np.asarray(state.get("enemies") or [], dtype=np.float64).reshape(-1, cols)
        if len(enemies):
            cls = enemies[:, 8] if cols > 8 else np.zeros(len(enemies))
            st = enemies[:, 9] if cols > 9 else np.zeros(len(enemies))
            hp = np.clip(100 * enemies[:, 4] / np.maximum(enemies[:, 5], 1), 0, 100)
            frame["e"] = np.rint(np.column_stack([enemies[:, 0], enemies[:, 1], enemies[:, 7], cls, st, hp])) \
                .astype(int).ravel().tolist()
        for key, src, cols_n, keep in (("pr", "projectiles", 4, 2), ("g", "materials", 3, 2),
                                       ("c", "consumables", 3, 3)):
            rows = np.asarray(state.get(src) or [], dtype=np.float64).reshape(-1, cols_n)
            if len(rows):
                frame[key] = np.rint(rows[:, :keep]).astype(int).ravel().tolist()
        births = np.asarray(state.get("births") or [], dtype=np.float64).reshape(-1, 4)
        births = births[np.isin(births[:, 3], (1, 4))]
        if len(births):
            frame["b"] = np.rint(births[:, :2]).astype(int).ravel().tolist()
        decisions = (info or {}).get("decisions")
        if decisions:
            frame["d"] = [{k: v for k, v in d.items() if k in ("kind", "action", "index", "weapon_index", "wave")}
                          for d in decisions]
        self.frames.append(frame)

    def finish(self, info: dict = None, **extra) -> None:
        info = info or {}
        self.meta.update({k: info.get(k) for k in ("outcome", "wave", "kills", "materials_collected", "damage_taken")
                          if k in info}, **extra)

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": REPLAY_VERSION, "meta": {**self.meta, "steps": len(self.frames)}, "frames": self.frames}
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump(payload, fh, separators=(",", ":"))
        return path


def load(path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)
