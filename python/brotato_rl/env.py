"""Gymnasium environment: one Brotato run per episode, the agent controls movement."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .encoding import EncoderConfig, ObservationEncoder
from .instance import GameInstance
from .meta import MetaPolicy, RandomMetaPolicy
from .protocol import BridgeError
from .rewards import RewardConfig, compute_reward

# Discrete actions: 0 = stand still, then 8 directions counter-clockwise from east.
# Screen coordinates: +x is right, +y is down.
_DIRS = np.array([[0, 0], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]], dtype=np.float32)
DIRECTIONS = _DIRS / np.maximum(np.linalg.norm(_DIRS, axis=1, keepdims=True), 1.0)
ACTION_NAMES = ("stay", "E", "NE", "N", "NW", "W", "SW", "S", "SE")


class RunConfigError(ValueError):
    """The game rejected the run settings (unknown character, weapon, danger or zone)."""


class BrotatoEnv(gym.Env):
    """Brotato as a Gymnasium environment.

    Observation: dict of float32 arrays (see encoding.py). Action: Discrete(9) by default,
    or Box(-1, 1, (2,)) with action_type="continuous" (a direction; below dead_zone = stand
    still; Brotato always moves at full speed). One step = frame_skip physics frames
    (60 per second of game time). The game waits for every action, so observations
    always match the state the action is applied to.

    Episode = one run. terminated: the character died or won wave 20. truncated:
    max_episode_steps reached, or the game instance crashed or hung (info["error"]).
    Level-ups, crates and the shop are decided by meta_policy between steps.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, character="character_well_rounded", weapon=None, danger=0, zone=0, start_wave=1,
                 frame_skip=4, action_type="discrete", dead_zone=0.25, render_mode=None, realtime=None,
                 sound=False, lite_visuals=None, meta_policy: MetaPolicy = None, reward_config: RewardConfig = None,
                 encoder_config: EncoderConfig = None, max_episode_steps=None, include_raw=False,
                 instance_name=None, reply_timeout=60.0, connect_timeout=120.0, keep_files=False,
                 game_options: dict = None, resolution=(1280, 720), speed=1.0, window_position=None,
                 game_args=()):
        if render_mode not in (None, "human"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        # game_options: extra reset fields understood by the mod (enemy_scaling, wave_duration,
        # endless, ban_mode, ban_tokens, keep_lock, starting_materials, invulnerable).
        self.run_settings = {**(game_options or {}), "character": character, "weapon": weapon, "danger": danger,
                             "zone": zone, "start_wave": start_wave}
        self.frame_skip = int(frame_skip)
        self.action_type = action_type
        self.dead_zone = dead_zone
        self.render_mode = render_mode
        human = render_mode == "human"
        self.realtime = human if realtime is None else realtime
        # Only with realtime: wall-clock pacing factor (1 = normal tempo, 4 = four times faster).
        self.speed = float(speed)
        self.meta_policy = meta_policy or RandomMetaPolicy()
        self.reward_config = reward_config or RewardConfig()
        self.encoder = ObservationEncoder(encoder_config)
        self.max_episode_steps = max_episode_steps
        self.include_raw = include_raw
        self._instance_kwargs = dict(name=instance_name, render=human, realtime=self.realtime, window=human,
                                     sound=sound, lite_visuals=(not human) if lite_visuals is None else lite_visuals,
                                     reply_timeout=reply_timeout, connect_timeout=connect_timeout,
                                     keep_files=keep_files, resolution=tuple(resolution), position=window_position,
                                     extra_args=tuple(game_args))  # extra Godot flags, e.g. "--always-on-top"

        self.observation_space = self.encoder.space
        if action_type == "discrete":
            self.action_space = spaces.Discrete(len(DIRECTIONS))
        elif action_type == "continuous":
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        else:
            raise ValueError("action_type must be 'discrete' or 'continuous'")

        self.instance = None
        self.hello = None
        self.enemy_types = {}
        self._state = None
        self._obs = self.encoder.zeros()
        self._steps = 0
        self._needs_reset = True

    # -- gymnasium API ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.meta_policy.seed(np.random.default_rng(int(self.np_random.integers(0, 2**63 - 1))))
        settings = dict(self.run_settings)
        settings.update(options or {})
        run_seed = settings.pop("seed", None)
        if run_seed is None:
            run_seed = int(self.np_random.integers(0, 2**31 - 1))
        msg = {"type": "reset", "frame_skip": self.frame_skip, "seed": int(run_seed),
               "render": self.render_mode == "human", "realtime": self.realtime, "speed": self.speed,
               **{k: v for k, v in settings.items() if v is not None}}

        for attempt in range(2):
            try:
                self._ensure_instance()
                self.instance.send(msg)
                state, decisions = self._read_until_step()
                break
            except BridgeError:
                self._drop_instance()
                if attempt:
                    raise
        self._steps = 0
        self._needs_reset = state["type"] == "end"
        self._accept_state(state)
        info = self._info(state, decisions)
        info["seed"] = int(run_seed)
        return self._obs, info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before step() (the episode has ended)")
        prev = self._state
        try:
            self.instance.send({"type": "act", "move": self._action_to_move(action)})
            state, decisions = self._read_until_step()
        except BridgeError as exc:
            log = self.instance.log_tail() if self.instance else ""
            self._drop_instance()
            self._needs_reset = True
            info = self._info(prev, [])
            info.update(error=str(exc), game_log=log)
            return self._obs, 0.0, False, True, info

        terminated = state["type"] == "end"
        outcome = state.get("outcome") if terminated else None
        reward, parts = compute_reward(prev, state, outcome, self.reward_config)
        self._accept_state(state)
        self._steps += 1
        truncated = (not terminated and self.max_episode_steps is not None
                     and self._steps >= self.max_episode_steps)
        self._needs_reset = terminated or truncated
        info = self._info(state, decisions, wave_changed=state.get("wave") != prev.get("wave"))
        info["reward_parts"] = parts
        if outcome:
            info["outcome"] = outcome
        return self._obs, float(reward), terminated, truncated, info

    def render(self):
        # In "human" mode the game window itself is the rendering.
        return None

    def close(self):
        if self.instance is not None:
            self.instance.close()
            self.instance = None

    # -- helpers ------------------------------------------------------------------------

    @property
    def raw_state(self) -> dict:
        """Last raw state dict sent by the mod (positions in game pixels)."""
        return self._state

    def _ensure_instance(self):
        if self.instance is not None and self.instance.alive:
            return
        self._drop_instance()
        self.instance = GameInstance(**self._instance_kwargs)
        self.hello = self.instance.start()
        if self.encoder.config.threat_features:
            version = tuple(int(x) for x in str(self.hello.get("mod_version", "0")).split(".") if x.isdigit())
            if version < (0, 4, 0):
                self._drop_instance()
                raise RunConfigError(f"threat_features needs the RLBridge mod 0.4.0+, the lab has "
                                     f"{self.hello.get('mod_version')}; run python\\scripts\\setup_lab.py")

    def _drop_instance(self):
        if self.instance is not None:
            self.instance.kill()
            self.instance.close()
            self.instance = None

    def _read_until_step(self):
        decisions = []
        while True:
            msg = self.instance.recv()
            kind = msg.get("type")
            if kind in ("obs", "end"):
                return msg, decisions
            if kind == "decision":
                handler = getattr(self.meta_policy, msg["kind"])
                reply = handler(msg)
                decisions.append({"kind": msg["kind"], "wave": msg.get("wave"), **reply})
                self.instance.send({"type": "choose", **reply})
            elif kind == "error":
                raise RunConfigError(msg.get("message"))
            else:
                raise BridgeError(f"unexpected message from game: {kind}")

    def _accept_state(self, state):
        self.enemy_types.update(state.get("new_types") or {})
        self._state = state
        self._obs = self.encoder.encode(state)

    def _action_to_move(self, action):
        if self.action_type == "discrete":
            move = DIRECTIONS[int(action)]
        else:
            move = np.asarray(action, dtype=np.float32).reshape(2)
            norm = float(np.hypot(*move))
            move = move / norm if norm >= self.dead_zone else np.zeros(2, dtype=np.float32)
        return [float(move[0]), float(move[1])]

    def _info(self, state, decisions, wave_changed=False):
        player = state.get("player") or {}
        counters = state.get("counters") or {}
        info = {
            "wave": state.get("wave"),
            "wave_changed": wave_changed,
            "frame": state.get("frame"),
            "hp": player.get("hp"),
            "max_hp": player.get("max_hp"),
            "level": player.get("level"),
            "materials": player.get("materials"),
            "kills": counters.get("kills"),
            "materials_collected": counters.get("materials_collected"),
            "damage_taken": counters.get("damage_taken"),
            "decisions": decisions,
        }
        if self.include_raw:
            info["raw"] = state
        return info
