"""Escape freedom: how much open space the character has around it.

Casts n_rays rays from the character. Each ray stops at the arena wall or where it passes
within `body` pixels of an enemy, and is capped at `ray_len`. Freedom = mean free length /
ray_len, in [0, 1]: 1 in the open, ~0.56 against a wall, ~0.31 in an empty corner and near
0 when boxed in. Unlike "distance from the nearest enemy", it falls when the character is
pushed against a wall, so it tells a corner apart from an open escape route.
"""
import numpy as np

_RAY_CACHE = {}


def _rays(n):
    if n not in _RAY_CACHE:
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        _RAY_CACHE[n] = np.stack([np.cos(a), np.sin(a)], axis=1)
    return _RAY_CACHE[n]


def free_lengths(rel_enemies, edges, n_rays=16, ray_len=400.0, body=60.0, dirs=None):
    """Free length of every ray, shape (n_rays,).

    rel_enemies: (N, 2) obstacle positions relative to the character.
    edges: distances to the left, right, top and bottom wall (same units, all >= 0).
    dirs: optional (R, 2) unit directions instead of n_rays evenly spread rays.
    """
    d = _rays(n_rays) if dirs is None else np.asarray(dirs, dtype=np.float64)
    left, right, top, bottom = (max(0.0, float(e)) for e in edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[:, 0] > 1e-9, right / d[:, 0], np.where(d[:, 0] < -1e-9, left / -d[:, 0], np.inf))
        ty = np.where(d[:, 1] > 1e-9, bottom / d[:, 1], np.where(d[:, 1] < -1e-9, top / -d[:, 1], np.inf))
    t = np.minimum(np.minimum(tx, ty), ray_len)

    e = np.asarray(rel_enemies, dtype=np.float64).reshape(-1, 2)
    if len(e):
        along = e @ d.T                                   # (N, R) distance along each ray
        perp2 = (e ** 2).sum(axis=1, keepdims=True) - along ** 2
        hit = (along > 0) & (perp2 < body ** 2)
        t_hit = np.where(hit, np.maximum(0.0, along - np.sqrt(np.maximum(0.0, body ** 2 - perp2))), np.inf)
        t = np.minimum(t, t_hit.min(axis=0))
    return t


def freedom(rel_enemies, edges, n_rays=16, ray_len=400.0, body=60.0) -> float:
    return float(free_lengths(rel_enemies, edges, n_rays, ray_len, body).mean() / ray_len)


def freedom_from_raw(state: dict, n_rays=16, ray_len=400.0, body=60.0):
    """Freedom from a raw state of the mod (pixels). Enemies and enemy spawn markers block rays.
    None outside combat."""
    player, zone = state.get("player") or {}, state.get("zone")
    if not zone or not player.get("pos"):
        return None
    px, py = player["pos"]
    x0, y0, x1, y1 = zone
    cols = int(state.get("enemy_cols", 8))  # 10 from mod 0.4.0 on
    enemies = np.asarray(state.get("enemies") or [], dtype=np.float64).reshape(-1, cols)[:, :2]
    births = np.asarray(state.get("births") or [], dtype=np.float64).reshape(-1, 4)
    births = births[np.isin(births[:, 3], (1, 4)), :2]  # EntityType.ENEMY, EntityType.BOSS
    obstacles = np.concatenate([enemies, births]) - (px, py)
    return freedom(obstacles, (px - x0, x1 - px, py - y0, y1 - py), n_rays, ray_len, body)


def wall_closeness(state: dict, margin=200.0):
    """0 when at least `margin` px from every wall, grows to 1 per axis at the wall (corner: up to 2)."""
    player, zone = state.get("player") or {}, state.get("zone")
    if not zone or not player.get("pos") or margin <= 0:
        return None
    px, py = player["pos"]
    x0, y0, x1, y1 = zone
    cx = max(0.0, margin - min(px - x0, x1 - px)) / margin
    cy = max(0.0, margin - min(py - y0, y1 - py)) / margin
    return min(1.0, cx) ** 2 + min(1.0, cy) ** 2
