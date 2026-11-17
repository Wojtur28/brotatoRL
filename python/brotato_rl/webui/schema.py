"""Parameter definitions shown in the web panel.

Defaults are read from the real classes (RewardConfig, EncoderConfig, MetaConfig, HeuristicBot,
BrotatoEnv), so the panel stays in sync with the code. The browser renders the form from
SCHEMA; runner.build_* turns the submitted values back into Python objects.
"""
import inspect
from dataclasses import fields

from ..bots import HeuristicBot, RandomBot
from ..encoding import EncoderConfig
from ..env import BrotatoEnv
from ..meta import STAT_KEYS, MetaConfig
from ..rewards import RewardConfig


def _defaults(callable_):
    return {n: p.default for n, p in inspect.signature(callable_).parameters.items()
            if p.default is not inspect.Parameter.empty}


ENV = _defaults(BrotatoEnv.__init__)
HEUR = _defaults(HeuristicBot.__init__)
RAND = _defaults(RandomBot.__init__)
META = MetaConfig()
REWARD = RewardConfig()
ENC = EncoderConfig()


def f(key, label, type_, default, help_="", **extra):
    return {"key": key, "label": label, "type": type_, "default": default, "help": help_, **extra}


STAT_LABELS = {
    "max_hp": "Max HP", "hp_regeneration": "HP regeneration", "lifesteal": "Life steal",
    "percent_damage": "% damage", "melee_damage": "Melee damage", "ranged_damage": "Ranged damage",
    "elemental_damage": "Elemental damage", "attack_speed": "Attack speed", "crit_chance": "Crit chance",
    "engineering": "Engineering", "range": "Range", "armor": "Armor", "dodge": "Dodge", "speed": "Speed",
    "luck": "Luck", "harvesting": "Harvesting",
}

REWARD_HELP = {
    "freedom_weight": "Reward for open escape routes (e.g. 0.5). The penalty starts as soon as the character lets itself get boxed in.",
    "wall_weight": "Penalty for being close to a wall, ×2 in a corner (e.g. 0.2).",
    "shaping_gamma": "Set equal to the RL algorithm's gamma (PPO usually 0.99).",
    "freedom_rays": "Number of rays in the freedom measure.",
    "freedom_ray_len": "Ray length [px] (~1.3 s of walking).",
    "freedom_body": "Radius within which an enemy blocks a ray [px].",
    "wall_margin": "Distance from the wall at which the penalty starts [px].",
}
ENCODER_HELP = {
    "free_rays": "Adds obs['free']: open paths in N directions (0 = off; changes the observation space).",
    "sectors": "Adds obs['sectors']: count of ALL enemies in N sectors, near/far (0 = off).",
    "threat_features": "1 = enemy attack kind, windup/dash and time to contact (mod 0.4.0+; changes the observation space).",
}
REWARD_HELP.update({
    "damage_source": "counter = hits reported by the mod; hp = HP lost (any source, no overkill on death).",
    "damage_low_hp": "Damage penalty × (1 + w × missing HP); 0 = flat.",
})
SELECT_OPTIONS = {"damage_source": [["counter", "counter (hits)"], ["hp", "hp (HP lost)"]]}


def config_field(obj, name, help_map):
    """Form field for a dataclass field: select for known text options, 0/1 for flags, else a number."""
    value, help_ = getattr(obj, name), help_map.get(name, "")
    if name in SELECT_OPTIONS:
        return f(name, name, "select", value, help_, options=SELECT_OPTIONS[name])
    if isinstance(value, bool):
        return f(name, name, "int", int(value), help_, min=0, max=1)
    if isinstance(value, int):
        return f(name, name, "int", value, help_, min=0, step=1)
    return f(name, name, "float", value, help_, step=0.01)

SCHEMA = [
    {"id": "run", "title": "Run", "help": "Run settings sent with every reset().", "fields": [
        f("character", "Character", "character", "character_well_rounded",
          "List from the game (“Refresh catalog from game” button)."),
        f("weapon", "Starting weapon / item", "weapon", "", "Empty = the character's first option."),
        f("danger", "Danger (difficulty)", "danger", 0, "0–5; 6 = Nightmare (extra events: fog of war, spawns, bullet hell)."),
        f("zone", "Zone", "zone", 0, "0 = base game, 1 = DLC zone (Abyssal Terrors)."),
        f("start_wave", "Starting wave", "int", 1, "Start from a later wave (1–20).", min=1, max=20),
        f("episodes", "Number of episodes", "int", 3, "", min=1, max=10000),
        f("seed", "First episode seed", "int", 0, "Episode i gets seed + i. Empty = random.", nullable=True),
        f("max_episode_steps", "Step limit", "int", None, "Empty = no limit (truncated after the limit).",
          nullable=True, min=1),
    ]},
    {"id": "game", "title": "Game rules (mod)", "help": "Game-side mechanics overrides. Applied again on every reset.",
     "fields": [
         f("enemy_health", "Enemy HP multiplier", "float", 1.0, "Game accessibility option (enemy_scaling.health).",
           min=0.01, max=10, step=0.05),
         f("enemy_damage", "Enemy damage multiplier", "float", 1.0, "enemy_scaling.damage", min=0.01, max=10, step=0.05),
         f("enemy_speed", "Enemy speed multiplier", "float", 1.0, "enemy_scaling.speed", min=0.01, max=10, step=0.05),
         f("wave_duration", "Wave length [s]", "int", None, "Empty = the wave's default (20–60 s).", nullable=True,
           min=1, max=600),
         f("starting_materials", "Starting materials", "int", 0, "Added to the character's starting materials.", min=0),
         f("invulnerable", "Invulnerable", "bool", False, "For testing: the character does not lose HP."),
         f("endless", "Endless mode", "bool", False, "The run goes on after wave 20 (experimental)."),
         f("ban_mode", "Ban mode", "select", "default", "Abyssal Terrors: item banning.",
           options=[["default", "default (game setting)"], ["on", "on"], ["off", "off"]]),
         f("ban_tokens", "Ban tokens", "int", 8, "Number of bans per run.", min=0, max=99),
         f("keep_lock", "Locked items carry over to the next shop", "bool", True, ""),
     ]},
    {"id": "env", "title": "Environment", "help": "BrotatoEnv parameters and the number of parallel instances.", "fields": [
        f("instances", "Game instances at once", "int", 1,
          "Each instance is a separate game process: ~0.5 GB RAM (no window ~410–530 MB, with a window ~460–580 MB). "
          "Episodes are split between instances.", min=1, max=16),
        f("watch", "Show game windows (render_mode=human)", "bool", False, "Each instance gets its own window."),
        f("tile_windows", "Tile windows in a grid", "bool", True,
          "With >1 window: size and position are fitted to the screen (the “Window resolution” field is then ignored).",
          show_if=["watch", True]),
        f("speed", "Game speed", "select", "auto",
          "With a window: 1× = normal speed; 2×/4×/8× = faster (if the CPU keeps up); max = no limit. "
          "Without a window, max is always the best choice.",
          options=[["auto", "auto (1× with a window, max without)"], ["0.5", "0.5× (slowed down)"], ["1", "1× (real time)"],
                   ["2", "2×"], ["4", "4×"], ["8", "8×"], ["max", "max (no limit)"]]),
        f("frame_skip", "frame_skip", "int", ENV["frame_skip"], "Game frames per decision (4 = 15 decisions/s).", min=1, max=60),
        f("action_type", "Action type", "select", ENV["action_type"], "",
          options=[["discrete", "discrete: Discrete(9)"], ["continuous", "continuous: Box(-1,1,(2,))"]]),
        f("dead_zone", "dead_zone", "float", ENV["dead_zone"], "Continuous only: a shorter vector = stand still.",
          min=0, max=1, step=0.05),
        f("sound", "Sound", "bool", ENV["sound"], ""),
        f("lite_visuals", "Lite visuals", "select", "auto", "No screen shake, effects or damage numbers.",
          options=[["auto", "auto (on without a window)"], ["on", "on"], ["off", "off"]]),
        f("resolution", "Window resolution", "select", "1280x720", "For a single window, or when windows are not tiled.",
          options=[["640x360", "640×360"], ["960x540", "960×540"], ["1280x720", "1280×720"], ["1600x900", "1600×900"],
                   ["1920x1080", "1920×1080"]], show_if=["watch", True]),
        f("reply_timeout", "Game reply timeout [s]", "float", ENV["reply_timeout"], "", min=1),
        f("connect_timeout", "Game start timeout [s]", "float", ENV["connect_timeout"], "", min=5),
        f("keep_files", "Keep instance files (logs)", "bool", ENV["keep_files"], ""),
    ]},
    {"id": "agent", "title": "Movement agent (model)", "help": "Who controls the character during a wave.", "fields": [
        f("type", "Agent", "select", "heuristic", "", options=[
            ["heuristic", "HeuristicBot (potential field)"], ["random", "RandomBot"], ["still", "Stands still"],
            ["sb3", "stable-baselines3 model (.zip)"], ["custom", "Custom Python class"]]),
        f("hold_steps", "RandomBot: steps in the same direction", "int", RAND["hold_steps"], "", min=1,
          show_if=["type", "random"]),
        *[f(k, f"HeuristicBot: {k}", "float", v,
            "Weight of open paths when picking a direction (0 = old behaviour, gets stuck in corners)."
            if k == "escape_weight" else "",
            step=0.05, show_if=["type", "heuristic"]) for k, v in HEUR.items()],
        f("sb3_algo", "Algorithm", "select", "PPO", "Class from stable_baselines3 / sb3_contrib.",
          options=[[a, a] for a in ("PPO", "A2C", "DQN", "SAC", "TD3", "DDPG", "RecurrentPPO", "QRDQN", "TRPO")],
          show_if=["type", "sb3"]),
        f("sb3_path", "Model path (.zip)", "text", "", "Relative to the project folder, or absolute.",
          show_if=["type", "sb3"]),
        f("flatten", "Flatten observation (FlattenObservation)", "bool", False,
          "Check if the model was trained on a flattened observation (MlpPolicy).", show_if=["type", "sb3"]),
        f("deterministic", "Deterministic actions", "bool", True, "", show_if=["type", "sb3"]),
        f("device", "Device", "select", "cpu", "", options=[["cpu", "cpu"], ["cuda", "cuda"], ["auto", "auto"]],
          show_if=["type", "sb3"]),
        f("custom_path", "Class: module:name", "text", "", "E.g. my_agents:MyAgent. The object must have act(obs). "
          "The constructor may take env.", show_if=["type", "custom"]),
    ]},
    {"id": "meta", "title": "Between waves (meta policy)", "help": "Level-ups, crates and the shop.", "fields": [
        f("type", "Meta policy", "select", "rule", "", options=[
            ["rule", "RuleMetaPolicy (parameters below)"], ["random", "RandomMetaPolicy (env default)"],
            ["first", "Always the first option, shop: move on at once"]]),
        f("random_spend_fraction", "Random: share of materials to spend", "float", 1.0, "", min=0, max=1, step=0.05,
          show_if=["type", "random"]),
        # level-up
        f("level_up_strategy", "Level-up: strategy", "select", META.level_up_strategy, "",
          options=[["score", "highest score (stat weights)"], ["random", "random"], ["first", "first option"]],
          show_if=["type", "rule"], section="Level-up"),
        f("level_up_max_rerolls", "Level-up: max rerolls", "int", META.level_up_max_rerolls, "", min=0,
          show_if=["type", "rule"]),
        f("level_up_reroll_below", "Level-up: reroll when best score <", "float", META.level_up_reroll_below,
          "", step=0.5, show_if=["type", "rule"]),
        # crates
        f("item_mode", "Crates: mode", "select", META.item_mode, "",
          options=[["take", "always take"], ["discard", "always recycle"], ["score", "by threshold"]],
          show_if=["type", "rule"], section="Crates"),
        f("item_min_tier", "Crates: min tier (0=I … 3=IV)", "int", META.item_min_tier, "", min=0, max=3,
          show_if=["type", "rule"]),
        f("item_min_score", "Crates: min score", "float", -100.0, "", step=0.5, show_if=["type", "rule"]),
        f("item_ban_rejected", "Crates: ban instead of recycling", "bool", META.item_ban_rejected,
          "Needs ban mode and ban tokens.", show_if=["type", "rule"]),
        # shop
        f("shop_strategy", "Shop: strategy", "select", META.shop_strategy, "",
          options=[["score", "highest score"], ["random", "random"], ["none", "buy nothing"]],
          show_if=["type", "rule"], section="Shop"),
        f("spend_fraction", "Shop: share of materials per visit", "float", META.spend_fraction, "", min=0, max=1,
          step=0.05, show_if=["type", "rule"]),
        f("keep_materials", "Shop: keep at least this many materials", "int", META.keep_materials, "", min=0,
          show_if=["type", "rule"]),
        f("shop_min_score", "Shop: min purchase score", "float", -100.0, "", step=0.5, show_if=["type", "rule"]),
        f("buy_weapons", "Shop: buy weapons", "bool", META.buy_weapons, "", show_if=["type", "rule"]),
        f("weapon_type", "Shop: weapon type", "select", META.weapon_type, "",
          options=[["any", "any"], ["melee", "melee"], ["ranged", "ranged"]], show_if=["type", "rule"]),
        f("max_weapons", "Shop: max weapons", "int", META.max_weapons, "", min=1, max=12, show_if=["type", "rule"]),
        f("auto_combine", "Shop: combine identical weapons", "bool", META.auto_combine, "", show_if=["type", "rule"]),
        f("recycle_below_tier", "Shop: recycle weapons with tier <", "int", META.recycle_below_tier,
          "-1 = never. At least 1 weapon is always kept.", min=-1, max=4, show_if=["type", "rule"]),
        f("max_rerolls", "Shop: max rerolls per visit", "int", META.max_rerolls, "", min=0,
          show_if=["type", "rule"]),
        f("reroll_max_price", "Shop: max reroll price", "int", META.reroll_max_price, "", min=0,
          show_if=["type", "rule"]),
        f("lock_unaffordable", "Shop: lock the best item you cannot afford", "bool", META.lock_unaffordable, "",
          show_if=["type", "rule"]),
        f("lock_min_tier", "Shop: min tier to lock", "int", META.lock_min_tier, "", min=0, max=3,
          show_if=["type", "rule"]),
        f("ban_ids", "Shop: ban items (ids, comma-separated)", "text", "", "E.g. item_jelly, item_duct_tape",
          show_if=["type", "rule"]),
        # scoring
        f("tier_weight", "Score: tier weight", "float", META.tier_weight, "", step=0.5, show_if=["type", "rule"],
          section="Item score"),
        f("price_weight", "Score: price penalty (per material)", "float", META.price_weight, "", step=0.01,
          show_if=["type", "rule"]),
        f("weapon_bonus", "Score: weapon bonus", "float", META.weapon_bonus, "", step=0.5, show_if=["type", "rule"]),
        f("other_effect_weight", "Score: other effects (each)", "float", META.other_effect_weight, "", step=0.5,
          show_if=["type", "rule"]),
        *[f(f"w_{k}", f"Weight: {STAT_LABELS[k]}", "float", META.stat_weights[k], "", step=0.25,
            show_if=["type", "rule"], compact=True) for k in STAT_KEYS],
    ]},
    {"id": "reward", "title": "Reward", "help": "RewardConfig: affects the reported return (and training). "
     "freedom_weight / wall_weight = position shaping (potential-based, 0 = off).", "fields": [
        config_field(REWARD, x.name, REWARD_HELP) for x in fields(RewardConfig)
    ]},
    {"id": "encoder", "title": "Observation", "help": "EncoderConfig: must match what the model was trained with.",
     "fields": [config_field(ENC, x.name, ENCODER_HELP) for x in fields(EncoderConfig)]},
]


def defaults() -> dict:
    return {g["id"]: {fd["key"]: fd["default"] for fd in g["fields"]} for g in SCHEMA}
