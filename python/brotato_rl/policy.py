"""Loading trained Stable-Baselines3 models and turning them into act(obs) functions."""
import importlib


def load_sb3(path, algo="PPO", threads=None):
    if threads:
        import torch
        torch.set_num_threads(int(threads))
    for module in ("stable_baselines3", "sb3_contrib"):
        try:
            cls = getattr(importlib.import_module(module), algo, None)
        except ImportError:
            continue
        if cls is not None:
            return cls.load(str(path), device="cpu")
    raise ImportError(f"no {algo} in stable_baselines3 / sb3_contrib")


def check_spaces(model, env):
    if model.observation_space != env.observation_space:
        mk = set(getattr(model.observation_space, "spaces", {}) or {})
        ek = set(getattr(env.observation_space, "spaces", {}) or {})
        detail = f"keys only in model: {sorted(mk - ek)}, only in env: {sorted(ek - mk)}" if mk != ek else \
            "same keys, different shapes (EncoderConfig differs from training)"
        raise ValueError(f"model observation space does not match the environment ({detail}); "
                         "use the spec.json the model was trained with")
    if model.action_space != env.action_space:
        raise ValueError(f"model action space {model.action_space} != env {env.action_space} (action_type)")


def actor(model, deterministic=True):
    return lambda obs: model.predict(obs, deterministic=deterministic)[0]
