"""Feature extractors for the SB3 policies.

The default (CombinedExtractor) flattens every table: 32 enemies x 16 features become 512 numbers in
fixed slots, and the MLP has to learn the same per-enemy rule 32 times over, in every slot. The bot
does something simpler: a force per enemy / projectile / spawn marker, summed, plus the nearest
pickup and the free length per direction (a minimum over obstacles).

EntitySetExtractor (DeepSets) mirrors that: one small network per table, shared by its rows, turns
every present row (column 0 > 0) into a vector; the rows are pooled by a sum (scaled down) and a
max, so the result does not depend on row order and every row trains the same weights. The tables
without rows (player, free rays, sectors) are flattened and appended.

    policy_kwargs={"features_extractor_class": EntitySetExtractor,
                   "features_extractor_kwargs": {"hidden": 64, "out": 64}}
In a training config: ppo.extractor = {"type": "entity_set", "hidden": 64, "out": 64}.
"""
import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

ENTITY_KEYS = ("enemies", "projectiles", "births", "materials", "consumables")


class EntitySetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, hidden: int = 64, out: int = 64, sum_scale: float = 0.125):
        sets = [k for k in ENTITY_KEYS if k in observation_space.spaces]
        flat = [k for k in observation_space.spaces if k not in sets]
        flat_dim = sum(int(th.tensor(observation_space[k].shape).prod()) for k in flat)
        super().__init__(observation_space, features_dim=len(sets) * 2 * out + flat_dim)
        self.sets, self.flat, self.sum_scale = sets, flat, sum_scale
        self.encoders = nn.ModuleDict({
            k: nn.Sequential(nn.Linear(observation_space[k].shape[-1], hidden), nn.ReLU(),
                             nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))
            for k in sets
        })

    def forward(self, obs: dict) -> th.Tensor:
        parts = []
        for k in self.sets:
            rows = obs[k]                                   # (batch, n, cols)
            mask = (rows[..., :1] > 0).float()              # present rows
            h = self.encoders[k](rows)                      # (batch, n, out)
            parts.append((h * mask).sum(dim=1) * self.sum_scale)
            masked = h.masked_fill(mask == 0, -1e4).max(dim=1).values
            parts.append(th.where(mask.sum(dim=1) > 0, masked, th.zeros_like(masked)))
        parts += [obs[k].flatten(start_dim=1) for k in self.flat]
        return th.cat(parts, dim=1)


EXTRACTORS = {"entity_set": EntitySetExtractor}


def policy_kwargs_for(extractor: dict | None) -> dict:
    """policy_kwargs entries for ppo.extractor of a training config ({} = SB3's default)."""
    if not extractor:
        return {}
    kwargs = {k: v for k, v in extractor.items() if k != "type"}
    return {"features_extractor_class": EXTRACTORS[extractor["type"]], "features_extractor_kwargs": kwargs}
