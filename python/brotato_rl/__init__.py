"""Brotato as a Gymnasium environment (game side: the RLBridge mod in mod/rl_bridge)."""
from gymnasium.envs.registration import register

from .encoding import EncoderConfig
from .env import ACTION_NAMES, DIRECTIONS, BrotatoEnv, RunConfigError
from .meta import MetaConfig, MetaPolicy, RandomMetaPolicy, RuleMetaPolicy
from .protocol import BridgeError
from .rewards import RewardConfig

register(id="BrotatoRL-v0", entry_point="brotato_rl.env:BrotatoEnv")

__all__ = [
    "ACTION_NAMES", "DIRECTIONS", "BrotatoEnv", "BridgeError", "EncoderConfig", "MetaConfig", "MetaPolicy",
    "RandomMetaPolicy", "RewardConfig", "RuleMetaPolicy", "RunConfigError",
]
