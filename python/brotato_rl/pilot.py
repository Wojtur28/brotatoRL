"""A frozen movement policy in plain numpy (no torch in the game workers).

The shop agent's environment plays every wave with a trained PPO movement model (EntitySetExtractor +
MLP). Through SB3 that costs ~1.8 ms per step (model.predict overhead) and ~200 MB of torch per worker;
the same forward pass in numpy takes a fraction of that. The weights are exported once (this needs
torch) into an .npz that also holds the run spec, so the environment encodes observations exactly as
in training.

    python -m brotato_rl.pilot runs/ppo5/final/model.zip models/pilot_ppo5_10M.npz    # export
    pilot = NumpyPilot("models/pilot_ppo5_10M.npz")
    env = BrotatoEnv(**spec.env_kwargs(pilot.spec)); action = pilot.act(obs)
"""
import json
import sys
from pathlib import Path

import numpy as np


def export_pilot(model_path, out_path) -> Path:
    """Writes the actor of an SB3 PPO model with EntitySetExtractor (deterministic actions only)."""
    from stable_baselines3 import PPO
    from torch import nn

    from . import spec as specs
    from .extractors import EntitySetExtractor

    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy
    extractor = policy.pi_features_extractor if hasattr(policy, "pi_features_extractor") else policy.features_extractor
    if not isinstance(extractor, EntitySetExtractor):
        raise ValueError(f"expected EntitySetExtractor, the model has {type(extractor).__name__}")

    arrays, meta = {}, {"sets": list(extractor.sets), "flat": list(extractor.flat),
                        "sum_scale": float(extractor.sum_scale), "encoder_layers": {}, "policy_layers": 0}

    def linears(seq, prefix, acts):
        n = 0
        for layer in seq:
            if isinstance(layer, nn.Linear):
                arrays[f"{prefix}{n}_w"] = layer.weight.detach().numpy().T.astype(np.float32)
                arrays[f"{prefix}{n}_b"] = layer.bias.detach().numpy().astype(np.float32)
                n += 1
            elif not isinstance(layer, acts):
                raise ValueError(f"unsupported layer {layer!r} in {prefix}")
        return n

    for key in extractor.sets:
        meta["encoder_layers"][key] = linears(extractor.encoders[key], f"enc_{key}_", (nn.ReLU,))
    meta["policy_layers"] = linears(policy.mlp_extractor.policy_net, "pi_", (nn.Tanh,))
    linears([policy.action_net], "act_", ())

    spec_path = specs.find_for_model(model_path)
    meta["spec"] = specs.load(spec_path) if spec_path else None
    meta["source"] = str(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, meta=np.array(json.dumps(meta)), **arrays)
    return out_path


class NumpyPilot:
    """act(obs) -> discrete action, the argmax of the exported policy (same as predict(deterministic=True))."""

    def __init__(self, path):
        data = np.load(str(path))
        meta = json.loads(str(data["meta"]))
        self.spec = meta["spec"]
        self.sets, self.flat, self.sum_scale = meta["sets"], meta["flat"], np.float32(meta["sum_scale"])
        self.encoders = {k: [(data[f"enc_{k}_{i}_w"], data[f"enc_{k}_{i}_b"]) for i in range(n)]
                         for k, n in meta["encoder_layers"].items()}
        self.policy = [(data[f"pi_{i}_w"], data[f"pi_{i}_b"]) for i in range(meta["policy_layers"])]
        self.action = (data["act_0_w"], data["act_0_b"])
        self.out = self.encoders[self.sets[0]][-1][0].shape[1]

    def logits(self, obs: dict) -> np.ndarray:
        parts = []
        for key in self.sets:
            rows = obs[key]
            rows = rows[rows[:, 0] > 0]  # present rows only; padding rows do not count in sum or max
            if len(rows):
                h = rows
                layers = self.encoders[key]
                for i, (w, b) in enumerate(layers):
                    h = h @ w + b
                    if i < len(layers) - 1:
                        np.maximum(h, 0, out=h)
                parts += [h.sum(axis=0) * self.sum_scale, h.max(axis=0)]
            else:
                parts += [np.zeros(self.out, np.float32)] * 2
        parts += [obs[key].ravel() for key in self.flat]
        x = np.concatenate(parts).astype(np.float32, copy=False)
        for w, b in self.policy:
            x = np.tanh(x @ w + b)
        return x @ self.action[0] + self.action[1]

    def act(self, obs: dict) -> int:
        return int(np.argmax(self.logits(obs)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m brotato_rl.pilot MODEL.zip OUT.npz")
    print(export_pilot(sys.argv[1], sys.argv[2]))
