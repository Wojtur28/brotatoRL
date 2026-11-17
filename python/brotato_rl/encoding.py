"""Turns the raw game state (dict from the mod) into fixed-size numpy observations.

Every entity table holds the K nearest objects to the player, sorted by distance. The
first column is a validity flag (1 = real row, 0 = padding). Positions are relative to
the player and divided by distance_scale; velocities are divided by speed_scale.
"""
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from .space import free_lengths

# Column layouts of the flat arrays sent by bridge.gd.
ENEMY_COLS = 8        # x, y, vx, vy, hp, max_hp, type, kind (mod < 0.4.0; the state says "enemy_cols")
                      # mod 0.4.0+: + attack class (0 contact, 1 shoots, 2 charges, 3 spawns),
                      #             + attack state (0 none, 1 winding up, 2 dashing)
PROJECTILE_COLS = 4   # x, y, vx, vy
BIRTH_COLS = 4        # x, y, time_left, entity_type
MATERIAL_COLS = 3     # x, y, value
CONSUMABLE_COLS = 3   # x, y, kind (0 fruit, 1 crate, 2 legendary crate, 3 other)
ENEMY_BIRTH_TYPES = (1, 4)  # EntityType.ENEMY, EntityType.BOSS

PLAYER_FEATURES = (
    "hp_frac", "max_hp", "armor", "dodge", "speed_stat", "hp_regeneration", "lifesteal", "move_speed",
    "iframes", "level", "xp_frac", "materials", "bonus_materials", "wave", "wave_time_left",
    "wave_progress", "elite_wave", "horde_wave", "pos_x", "pos_y", "edge_left", "edge_right",
    "edge_top", "edge_bottom", "harvesting", "luck", "range", "percent_damage", "attack_speed",
    "crit_chance",
)
ENEMY_FEATURES = ("valid", "dx", "dy", "dist", "vx", "vy", "hp_frac", "max_hp_log", "kind")
PROJECTILE_FEATURES = ("valid", "dx", "dy", "dist", "vx", "vy")
# threat_features=True appends these (they need mod 0.4.0 for the attack columns):
ENEMY_THREAT_FEATURES = ("shooter", "charger", "spawner", "winding_up", "dashing", "approach", "contact_soon")
PROJECTILE_THREAT_FEATURES = ("approach", "contact_soon")
BIRTH_FEATURES = ("valid", "dx", "dy", "dist", "time_left")
MATERIAL_FEATURES = ("valid", "dx", "dy", "dist", "value_log")
CONSUMABLE_FEATURES = ("valid", "dx", "dy", "dist", "is_fruit", "is_crate")
OBS_LIMIT = 10.0


@dataclass
class EncoderConfig:
    max_enemies: int = 32
    max_projectiles: int = 32
    max_births: int = 16
    max_materials: int = 16
    max_consumables: int = 8
    distance_scale: float = 1000.0
    speed_scale: float = 500.0
    # Optional "situational awareness" features (0 = off, observation space unchanged).
    # free_rays: obs["free"] (free_rays,) = free length of evenly spread rays (east first,
    #   then clockwise on screen) stopped by walls, enemies and spawn markers, / free_ray_len.
    #   Shows escape routes and corners directly (see space.py).
    # sectors: obs["sectors"] (sectors, 2) = ALL enemies (not only the K nearest) per angular
    #   sector, split into near (< sector_radius) and far, / 10. Shows a ring forming.
    free_rays: int = 0
    free_ray_len: float = 400.0
    free_ray_body: float = 60.0
    sectors: int = 0
    sector_radius: float = 400.0
    # threat_features: per enemy its attack kind (shoots / charges / spawns; none = contact only),
    #   whether it is winding up an attack or dashing (mod 0.4.0+), and for enemies and projectiles
    #   "approach" = closing speed / speed_scale (negative: moving away) and "contact_soon" =
    #   1 / (1 + seconds until it reaches the character at that speed; 0 when not closing in).
    #   The game introduces shooters (wave 4), chargers (6-7) and bruisers (8, charge after a
    #   wind-up); without these columns the agent cannot tell them apart.
    threat_features: bool = False


def _rows(flat, cols):
    arr = np.asarray(flat if flat else [], dtype=np.float32)
    return arr.reshape(-1, cols)


class ObservationEncoder:
    def __init__(self, config: EncoderConfig = None):
        self.config = config or EncoderConfig()
        c = self.config
        threat = bool(c.threat_features)
        self.shapes = {
            "player": (len(PLAYER_FEATURES),),
            "enemies": (c.max_enemies, len(ENEMY_FEATURES) + threat * len(ENEMY_THREAT_FEATURES)),
            "projectiles": (c.max_projectiles, len(PROJECTILE_FEATURES) + threat * len(PROJECTILE_THREAT_FEATURES)),
            "births": (c.max_births, len(BIRTH_FEATURES)),
            "materials": (c.max_materials, len(MATERIAL_FEATURES)),
            "consumables": (c.max_consumables, len(CONSUMABLE_FEATURES)),
        }
        if c.free_rays > 0:
            self.shapes["free"] = (c.free_rays,)
        if c.sectors > 0:
            self.shapes["sectors"] = (c.sectors, 2)
        self.space = spaces.Dict({
            key: spaces.Box(-OBS_LIMIT, OBS_LIMIT, shape, dtype=np.float32) for key, shape in self.shapes.items()
        })

    def zeros(self):
        return {key: np.zeros(shape, dtype=np.float32) for key, shape in self.shapes.items()}

    def encode(self, state: dict) -> dict:
        obs = self.zeros()
        player = state.get("player") or {}
        if not player.get("pos"):
            return obs
        c = self.config
        px, py = player["pos"]
        stats = player.get("stats", {})
        zone = state.get("zone") or [px - 1, py - 1, px + 1, py + 1]
        x0, y0, x1, y1 = zone
        half_w, half_h = max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
        max_hp = max(1.0, float(player.get("max_hp", 1)))
        duration = max(1e-3, float(state.get("wave_duration", 1)))
        left = float(state.get("wave_time_left", 0))
        xp_next = max(1e-3, float(player.get("xp_next", 1)))
        nb_waves = max(1, int(state.get("nb_waves", 20)))
        obs["player"][:] = [
            player.get("hp", 0) / max_hp,
            max_hp / 100.0,
            stats.get("armor", 0) / 20.0,
            stats.get("dodge", 0) / 100.0,
            stats.get("speed", 0) / 100.0,
            stats.get("hp_regeneration", 0) / 20.0,
            stats.get("lifesteal", 0) / 100.0,
            player.get("move_speed", 0) / c.speed_scale,
            player.get("iframes", 0),
            player.get("level", 0) / 20.0,
            player.get("xp", 0) / xp_next,
            np.log1p(max(0.0, player.get("materials", 0))) / 5.0,
            np.log1p(max(0.0, player.get("bonus_materials", 0))) / 5.0,
            state.get("wave", 0) / nb_waves,
            left / 90.0,
            1.0 - left / duration,
            float(bool(state.get("elite_wave"))),
            float(bool(state.get("horde_wave"))),
            (px - (x0 + x1) / 2) / half_w,
            (py - (y0 + y1) / 2) / half_h,
            (px - x0) / c.distance_scale,
            (x1 - px) / c.distance_scale,
            (py - y0) / c.distance_scale,
            (y1 - py) / c.distance_scale,
            stats.get("harvesting", 0) / 50.0,
            stats.get("luck", 0) / 100.0,
            stats.get("range", 0) / 200.0,
            stats.get("percent_damage", 0) / 100.0,
            stats.get("attack_speed", 0) / 100.0,
            stats.get("crit_chance", 0) / 100.0,
        ]

        enemy_cols = int(state.get("enemy_cols", ENEMY_COLS))
        enemies = _rows(state.get("enemies"), enemy_cols)
        threat = bool(c.threat_features)

        def enemy_extra(r):
            base = [r[:, 2] / c.speed_scale, r[:, 3] / c.speed_scale,
                    r[:, 4] / np.maximum(r[:, 5], 1.0), np.log1p(np.maximum(r[:, 5], 0.0)) / 8.0, r[:, 7] / 2.0]
            if threat:
                cls = r[:, 8] if enemy_cols > 8 else np.zeros(len(r), dtype=np.float32)
                st = r[:, 9] if enemy_cols > 9 else np.zeros(len(r), dtype=np.float32)
                base += [(cls == 1), (cls == 2), (cls == 3), (st == 1), (st == 2), *self._closing(r, px, py)]
            return np.stack([np.asarray(b, dtype=np.float32) for b in base], axis=1)

        self._fill(obs["enemies"], enemies, px, py, enemy_extra)
        projectiles = _rows(state.get("projectiles"), PROJECTILE_COLS)
        self._fill(obs["projectiles"], projectiles, px, py, lambda r: np.concatenate(
            [r[:, 2:4] / c.speed_scale] + ([np.stack(self._closing(r, px, py), axis=1)] if threat else []), axis=1))
        births = _rows(state.get("births"), BIRTH_COLS)
        births = births[np.isin(births[:, 3], ENEMY_BIRTH_TYPES)] if len(births) else births
        self._fill(obs["births"], births, px, py, lambda r: r[:, 2:3] / 2.0)
        materials = _rows(state.get("materials"), MATERIAL_COLS)
        self._fill(obs["materials"], materials, px, py, lambda r: np.log1p(np.maximum(r[:, 2:3], 0.0)) / 3.0)
        consumables = _rows(state.get("consumables"), CONSUMABLE_COLS)
        self._fill(obs["consumables"], consumables, px, py, lambda r: np.stack([
            (r[:, 2] == 0).astype(np.float32), ((r[:, 2] == 1) | (r[:, 2] == 2)).astype(np.float32),
        ], axis=1))

        if c.free_rays > 0 or c.sectors > 0:
            obstacles = np.concatenate([enemies[:, :2], births[:, :2]]) - (px, py) if len(enemies) or len(births) \
                else np.zeros((0, 2))
            if c.free_rays > 0:
                t = free_lengths(obstacles, (px - x0, x1 - px, py - y0, y1 - py), c.free_rays,
                                 c.free_ray_len, c.free_ray_body)
                obs["free"][:] = t / c.free_ray_len
            if c.sectors > 0 and len(enemies):
                rel = enemies[:, :2] - (px, py)
                angle = np.mod(np.arctan2(rel[:, 1], rel[:, 0]) + np.pi / c.sectors, 2 * np.pi)
                sector = np.minimum((angle / (2 * np.pi) * c.sectors).astype(int), c.sectors - 1)
                far = (np.hypot(rel[:, 0], rel[:, 1]) >= c.sector_radius).astype(int)
                np.add.at(obs["sectors"], (sector, far), 0.1)

        for value in obs.values():
            np.clip(value, -OBS_LIMIT, OBS_LIMIT, out=value)
        return obs

    def _closing(self, rows, px, py):
        """(approach, contact_soon) of rows with x, y, vx, vy in columns 0-3 (game pixels)."""
        rel = rows[:, 0:2] - (px, py)
        dist = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1.0)
        closing = -(rel[:, 0] * rows[:, 2] + rel[:, 1] * rows[:, 3]) / dist  # px/s towards the character
        seconds = np.where(closing > 1.0, dist / np.maximum(closing, 1.0), np.inf)
        return closing / self.config.speed_scale, 1.0 / (1.0 + seconds)

    def _fill(self, out, rows, px, py, extra):
        if not len(rows):
            return
        scale = self.config.distance_scale
        dx = (rows[:, 0] - px) / scale
        dy = (rows[:, 1] - py) / scale
        dist = np.hypot(dx, dy)
        order = np.argsort(dist, kind="stable")[: out.shape[0]]
        n = len(order)
        out[:n, 0] = 1.0
        out[:n, 1] = dx[order]
        out[:n, 2] = dy[order]
        out[:n, 3] = dist[order]
        out[:n, 4:] = extra(rows[order])
