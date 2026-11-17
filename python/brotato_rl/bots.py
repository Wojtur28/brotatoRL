"""Baseline players without machine learning. Both read the same observations an agent gets."""
import numpy as np

from .env import DIRECTIONS
from .space import free_lengths


class RandomBot:
    """Random direction, held for a few steps so the character actually travels."""

    def __init__(self, hold_steps=8, seed=None):
        self.hold_steps = hold_steps
        self.rng = np.random.default_rng(seed)
        self._action, self._left = 0, 0

    def act(self, obs) -> int:
        if self._left <= 0:
            self._action, self._left = int(self.rng.integers(len(DIRECTIONS))), self.hold_steps
        self._left -= 1
        return self._action


class HeuristicBot:
    """Potential field: pushed away from enemies, projectiles, spawn markers and walls,
    pulled towards materials, crates and (when hurt) fruit.

    Distances are in observation units (game pixels / distance_scale, 1000 by default).
    """

    def __init__(self, enemy_radius=0.45, projectile_radius=0.3, birth_radius=0.15, wall_margin=0.2,
                 enemy_weight=1.0, projectile_weight=1.5, birth_weight=0.6, wall_weight=1.2,
                 material_weight=0.35, fruit_weight=0.8, crate_weight=0.5, center_weight=0.05,
                 escape_weight=2.0, escape_ray_len=0.4, escape_body=0.06):
        self.enemy_radius = enemy_radius
        self.projectile_radius = projectile_radius
        self.birth_radius = birth_radius
        self.wall_margin = wall_margin
        self.enemy_weight = enemy_weight
        self.projectile_weight = projectile_weight
        self.birth_weight = birth_weight
        self.wall_weight = wall_weight
        self.material_weight = material_weight
        self.fruit_weight = fruit_weight
        self.crate_weight = crate_weight
        self.center_weight = center_weight
        # Escape term: each of the 8 directions also scores by how far it stays free of walls,
        # enemies and spawn markers (see space.py). Breaks the corner trap where the push from
        # enemies and the push from the wall cancel out. 0 = behaviour before the fix; measured
        # (Danger 0, 10-16 runs each): 0 -> mean wave 9.4, 2 -> 12.8, 3 -> 12.6, 5 -> 10.8.
        self.escape_weight = escape_weight
        self.escape_ray_len = escape_ray_len
        self.escape_body = escape_body

    @staticmethod
    def _repel(table, radius, weight):
        rows = table[table[:, 0] > 0]
        if not len(rows):
            return np.zeros(2), 0.0
        dist = np.maximum(rows[:, 3], 1e-3)
        close = dist < radius
        if not close.any():
            return np.zeros(2), 0.0
        rows, dist = rows[close], dist[close]
        strength = weight * (1.0 - dist / radius) ** 2 / dist
        force = -(rows[:, 1:3] * strength[:, None]).sum(axis=0)
        return force, float(strength.sum())

    @staticmethod
    def _attract(table, weight, mask=None):
        rows = table[table[:, 0] > 0]
        if mask is not None:
            rows = rows[mask(rows)]
        if not len(rows):
            return np.zeros(2)
        nearest = rows[np.argmin(rows[:, 3])]
        return weight * nearest[1:3] / max(nearest[3], 1e-3)

    def direction(self, obs) -> np.ndarray:
        player = obs["player"]
        threat = np.zeros(2)
        danger = 0.0
        for key, radius, weight in (("enemies", self.enemy_radius, self.enemy_weight),
                                    ("projectiles", self.projectile_radius, self.projectile_weight),
                                    ("births", self.birth_radius, self.birth_weight)):
            force, level = self._repel(obs[key], radius, weight)
            threat += force
            danger += level

        # Walls: player features 20..23 are distances to the left/right/top/bottom edges.
        left, right, top, bottom = player[20:24]
        m = self.wall_margin
        wall = np.array([max(0, m - left) - max(0, m - right), max(0, m - top) - max(0, m - bottom)]) / m
        threat += self.wall_weight * wall

        # Go for pickups when it is calm; the calmer, the stronger the pull.
        calm = 1.0 / (1.0 + danger)
        hp_frac = player[0]
        pull = self._attract(obs["materials"], self.material_weight)
        pull += self._attract(obs["consumables"], self.crate_weight, lambda r: r[:, 5] > 0)
        pull += self._attract(obs["consumables"], self.fruit_weight * (1.0 - hp_frac), lambda r: r[:, 4] > 0)
        pull += -self.center_weight * player[18:20]
        return threat + calm * pull

    def free_lengths(self, obs) -> np.ndarray:
        """Free length (obs units) along each of the 8 movement directions."""
        rows = [t[t[:, 0] > 0, 1:3] for t in (obs["enemies"], obs["births"])]
        return free_lengths(np.concatenate(rows), obs["player"][20:24], ray_len=self.escape_ray_len,
                            body=self.escape_body, dirs=DIRECTIONS[1:])

    def act(self, obs) -> int:
        force = self.direction(obs)
        if self.escape_weight:
            norm = np.hypot(*force)
            unit = force / norm if norm > 1e-3 else np.zeros(2)
            # Force strength saturates so the escape term still matters next to strong pushes.
            score = DIRECTIONS[1:] @ unit * min(norm, 1.0) + self.escape_weight * self.free_lengths(obs) / self.escape_ray_len
            return int(np.argmax(score)) + 1
        if np.hypot(*force) < 1e-3:
            return 0
        return int(np.argmax(DIRECTIONS[1:] @ force)) + 1
