"""Default reward. Every term comes from what the mod reports, so changing the reward never
requires touching the mod.

Positional shaping (freedom, wall) is potential-based (Ng, Harada, Russell 1999):
    F = gamma * Phi(s') - Phi(s),   Phi = freedom_weight * freedom - wall_weight * wall_closeness
Summed over an episode it telescopes to gamma^T * 0 - Phi(s0) plus the (gamma-1)*Phi drift, so it
cannot be farmed by walking back and forth and it does not change which policy is optimal. It
only gives the agent an immediate signal that it is being boxed in, long before the damage (and
death) arrives. Wave boundaries are ordinary transitions; only the terminal state has Phi = 0.
"""
from dataclasses import dataclass

from .space import freedom_from_raw, wall_closeness


@dataclass
class RewardConfig:
    alive: float = 0.01          # every agent step survived
    damage_taken: float = -1.0   # per max HP worth of damage
    material: float = 0.02       # per material picked up
    kill: float = 0.0            # per enemy killed (weapons aim and fire on their own)
    wave_cleared: float = 1.0
    death: float = -2.0
    win: float = 5.0
    # How damage is measured. "counter": the hits the mod reports (the lethal hit counts in full,
    # overkill included; a damage source that does not go through Player.take_damage is missed).
    # "hp": HP lost between two observations of the same wave (every source, never more than the
    # HP left; healing in the same step hides that much damage).
    damage_source: str = "counter"
    damage_low_hp: float = 0.0   # damage penalty x (1 + w * (1 - HP left / max HP)); 0 = flat
    # Positional shaping (0 = off). Suggested start: freedom_weight 0.5, wall_weight 0.2.
    freedom_weight: float = 0.0  # Phi += w * freedom (open escape routes, 0..1)
    wall_weight: float = 0.0     # Phi -= w * wall_closeness (0 in the open, up to 2 in a corner)
    shaping_gamma: float = 0.99  # should equal the discount of the RL algorithm
    freedom_rays: int = 16
    freedom_ray_len: float = 400.0  # px; ~1.3 s of walking at base speed
    freedom_body: float = 60.0      # px; a ray is blocked this close to an enemy
    wall_margin: float = 200.0      # px


def potential(state: dict, config: RewardConfig) -> float:
    phi = 0.0
    if config.freedom_weight:
        f = freedom_from_raw(state, config.freedom_rays, config.freedom_ray_len, config.freedom_body)
        phi += config.freedom_weight * (f or 0.0)
    if config.wall_weight:
        phi -= config.wall_weight * (wall_closeness(state, config.wall_margin) or 0.0)
    return phi


def damage_taken(prev: dict, state: dict, outcome, config: RewardConfig):
    """(damage in HP points, HP left after it) for prev -> state, as config.damage_source says."""
    p0, p1 = prev.get("player") or {}, state.get("player") or {}
    hp0 = p0.get("hp")
    hp1 = p1.get("hp")
    if hp1 is None:
        hp1 = 0.0 if outcome == "died" else hp0
    if config.damage_source == "hp":
        if hp0 is None or hp1 is None or prev.get("wave") != state.get("wave"):
            return 0.0, hp1  # a new wave starts with full HP: not damage
        return max(0.0, float(hp0) - float(hp1)), hp1
    if config.damage_source != "counter":
        raise ValueError(f"damage_source must be 'counter' or 'hp', not {config.damage_source!r}")
    before, after = prev.get("counters", {}), state.get("counters", {})
    return float(after.get("damage_taken", 0)) - float(before.get("damage_taken", 0)), hp1


def compute_reward(prev: dict, state: dict, outcome, config: RewardConfig):
    """Return (reward, components) for the transition prev -> state."""
    before, after = prev.get("counters", {}), state.get("counters", {})
    max_hp = max(1.0, float((prev.get("player") or {}).get("max_hp", 1)))

    def delta(key):
        return float(after.get(key, 0)) - float(before.get(key, 0))

    damage, hp_left = damage_taken(prev, state, outcome, config)
    damage_factor = 1.0
    if config.damage_low_hp and damage > 0 and hp_left is not None:
        damage_factor += config.damage_low_hp * (1.0 - min(1.0, max(0.0, float(hp_left) / max_hp)))

    waves = max(0, int(state.get("wave", 0)) - int(prev.get("wave", 0)))
    if outcome == "won":
        waves += 1  # the final wave never increments the counter

    shaping = 0.0
    if config.freedom_weight or config.wall_weight:
        # Both states are combat observations (also across a wave boundary: the last frame of
        # wave N -> the first frame of wave N+1), so the ordinary gamma*Phi(s') - Phi(s) applies.
        # Only the terminal state has Phi = 0. Treating a wave boundary as terminal would cost
        # Phi(start of wave) at every wave (~0.5 with the suggested weights) and break invariance.
        if outcome is None and "zone" in prev and "zone" in state:
            shaping = config.shaping_gamma * potential(state, config) - potential(prev, config)
        elif "zone" in prev:
            shaping = -potential(prev, config)  # terminal transition: Phi(terminal) = 0

    parts = {
        "alive": config.alive if outcome != "died" else 0.0,
        "damage_taken": config.damage_taken * damage_factor * damage / max_hp,
        "material": config.material * delta("materials_collected"),
        "kill": config.kill * delta("kills"),
        "wave_cleared": config.wave_cleared * waves,
        "death": config.death if outcome == "died" else 0.0,
        "win": config.win if outcome == "won" else 0.0,
        "position": shaping,
    }
    return sum(parts.values()), parts
