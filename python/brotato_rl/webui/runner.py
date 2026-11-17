"""Turns a panel config (dict of groups, see schema.SCHEMA) into env + agent + meta policy and
plays episodes in a background thread. Also usable without the panel: run_config(config)."""
import ctypes
import importlib
import inspect
import json
import math
import os
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from ctypes import wintypes
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .. import paths
from ..bots import HeuristicBot, RandomBot
from ..encoding import EncoderConfig
from ..env import DIRECTIONS, BrotatoEnv
from ..instance import GameInstance
from ..meta import STAT_KEYS, MetaConfig, MetaPolicy, RandomMetaPolicy, RuleMetaPolicy
from ..rewards import RewardConfig
from .schema import defaults

CATALOG_FILE = paths.LAB / "catalog.json"


def merge_defaults(config: dict) -> dict:
    merged = defaults()
    for group, values in (config or {}).items():
        if group in merged and isinstance(values, dict):
            merged[group].update(values)
    return merged


def _tri(value, auto=None):
    """'on' / 'off' / 'auto' (or 'default') -> True / False / auto."""
    return {"on": True, "off": False}.get(value, auto)


def _resolve(path: str) -> Path:
    p = Path(path.strip().strip('"'))
    return p if p.is_absolute() else paths.ROOT / p


# -- builders -----------------------------------------------------------------------------

def game_options(g: dict) -> dict:
    """Reset fields for the mod (see bridge.gd _setup_run)."""
    opts = {
        "enemy_scaling": {"health": float(g["enemy_health"]), "damage": float(g["enemy_damage"]),
                          "speed": float(g["enemy_speed"])},
        "starting_materials": int(g["starting_materials"] or 0),
        "invulnerable": bool(g["invulnerable"]),
        "endless": bool(g["endless"]),
        "ban_tokens": int(g["ban_tokens"]),
        "keep_lock": bool(g["keep_lock"]),
    }
    if g.get("wave_duration"):
        opts["wave_duration"] = int(g["wave_duration"])
    ban = _tri(g.get("ban_mode"))
    if ban is not None:
        opts["ban_mode"] = ban
    return opts


def build_meta(m: dict) -> MetaPolicy:
    kind = m.get("type", "rule")
    if kind == "random":
        return RandomMetaPolicy(shop_spend_fraction=float(m["random_spend_fraction"]))
    if kind == "first":
        return MetaPolicy()
    cfg = MetaConfig()
    for name in cfg.__dataclass_fields__:
        if name in ("stat_weights", "ban_ids") or name not in m:
            continue
        current = getattr(cfg, name)
        setattr(cfg, name, bool(m[name]) if isinstance(current, bool) else type(current)(m[name]))
    cfg.stat_weights = {k: float(m.get(f"w_{k}", cfg.stat_weights[k])) for k in STAT_KEYS}
    cfg.ban_ids = [s.strip() for s in str(m.get("ban_ids") or "").split(",") if s.strip()]
    return RuleMetaPolicy(cfg)


SPEED_MAX = "max"


def speed_settings(env: dict):
    """Panel 'speed' -> (realtime, speed) for BrotatoEnv. auto: 1x with a window, max without."""
    value = str(env.get("speed", "auto"))
    if value == "auto":
        return (True, 1.0) if env.get("watch") else (False, 1.0)
    if value == SPEED_MAX:
        return False, 1.0
    return True, max(0.05, float(value))


def build_env(cfg: dict, position=None, resolution=None, instance_name=None) -> BrotatoEnv:
    run, env, g = cfg["run"], cfg["env"], cfg["game"]
    w, h = resolution or (int(x) for x in str(env["resolution"]).split("x"))
    enc_fields = EncoderConfig.__dataclass_fields__
    realtime, speed = speed_settings(env)
    return BrotatoEnv(
        character=run["character"] or "character_well_rounded",
        weapon=run["weapon"] or None,
        danger=int(run["danger"]),
        zone=int(run["zone"]),
        start_wave=int(run["start_wave"]),
        frame_skip=int(env["frame_skip"]),
        action_type=env["action_type"],
        dead_zone=float(env["dead_zone"]),
        render_mode="human" if env["watch"] else None,
        realtime=realtime,
        speed=speed,
        sound=bool(env["sound"]),
        lite_visuals=_tri(env["lite_visuals"]),
        meta_policy=build_meta(cfg["meta"]),
        reward_config=RewardConfig(**{k: type(getattr(RewardConfig, k))(v) for k, v in cfg["reward"].items()
                                      if k in RewardConfig.__dataclass_fields__}),
        encoder_config=EncoderConfig(**{k: type(enc_fields[k].default)(v) for k, v in cfg["encoder"].items()
                                        if k in enc_fields}),
        max_episode_steps=int(run["max_episode_steps"]) if run.get("max_episode_steps") else None,
        reply_timeout=float(env["reply_timeout"]),
        connect_timeout=float(env["connect_timeout"]),
        keep_files=bool(env["keep_files"]),
        game_options=game_options(g),
        resolution=(int(w), int(h)),
        window_position=position,
        instance_name=instance_name,
    )


# -- machine info and window layout (Windows) ----------------------------------------------

# Measured per game process (working set): ~410 MB without a window, ~460 MB with one on
# wave 1, up to ~530 MB in late waves. The estimate uses the late-wave figure.
RAM_PER_INSTANCE_MB = {"headless": 530, "window": 580}


def system_info() -> dict:
    info = {"cpus": os.cpu_count() or 1, "ram_total_mb": None, "ram_available_mb": None, "screen": None,
            "ram_per_instance_mb": RAM_PER_INSTANCE_MB}
    if os.name != "nt":
        return info

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        info["ram_total_mb"] = status.ullTotalPhys // 2**20
        info["ram_available_mb"] = status.ullAvailPhys // 2**20
    info["screen"] = work_area()
    return info


def work_area():
    """(left, top, width, height) of the primary screen without the taskbar."""
    if os.name != "nt":
        return (0, 0, 1920, 1080)
    rect = wintypes.RECT()
    if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):  # SPI_GETWORKAREA
        return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
    return (0, 0, ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1))


# Window frame around the client area (measured: 640x360 client -> 656x399 window).
_FRAME_W, _FRAME_H = 16, 39


def tile_windows(n: int, max_resolution=(1280, 720)):
    """Grid of n 16:9 windows filling the work area (the grid with the largest windows).
    Returns [(position, resolution)]."""
    left, top, width, height = work_area()
    best = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w, cell_h = width // cols, height // rows
        client_w = min(cell_w - _FRAME_W, int((cell_h - _FRAME_H) * 16 / 9), max_resolution[0])
        if best is None or client_w > best[0]:
            best = (client_w, cols, cell_w, cell_h)
    client_w, cols, cell_w, cell_h = best
    client_w = max(320, client_w - client_w % 16)
    client_h = client_w * 9 // 16
    return [((left + (i % cols) * cell_w, top + (i // cols) * cell_h), (client_w, client_h)) for i in range(n)]


class Agent:
    """Wraps any player into act(obs) -> env action, for both action types."""

    def __init__(self, a: dict, env: BrotatoEnv, shared: dict = None):
        self.kind = a.get("type", "heuristic")
        self.continuous = env.action_type == "continuous"
        self.obs_space = env.observation_space
        self.flatten = bool(a.get("flatten"))
        self.deterministic = bool(a.get("deterministic", True))
        self.model_lock = (shared or {}).get("model_lock") or threading.Lock()
        self.impl = None
        if self.kind == "heuristic":
            names = inspect.signature(HeuristicBot.__init__).parameters
            self.impl = HeuristicBot(**{k: float(v) for k, v in a.items() if k in names and k != "self"})
        elif self.kind == "random":
            self.impl = RandomBot(hold_steps=int(a["hold_steps"]))
        elif self.kind == "sb3":
            self.impl = (shared or {}).get("model") or self._load_sb3(a)
        elif self.kind == "custom":
            self.impl = self._load_custom(a.get("custom_path", ""), env)
        elif self.kind != "still":
            raise ValueError(f"unknown agent {self.kind!r}")

    @staticmethod
    def _load_sb3(a):
        if not (a.get("sb3_path") or "").strip():
            raise ValueError("the model path (.zip) is required")
        path = _resolve(a["sb3_path"])
        if not path.exists() and not path.with_suffix(".zip").exists():
            raise FileNotFoundError(f"model file not found: {path}")
        algo, cls = a.get("sb3_algo", "PPO"), None
        for module in ("stable_baselines3", "sb3_contrib"):
            try:
                cls = getattr(importlib.import_module(module), algo, None)
            except ImportError:
                continue
            if cls is not None:
                break
        if cls is None:
            raise ImportError(f"class {algo} not found (pip install stable-baselines3 / sb3-contrib)")
        return cls.load(str(path), device=a.get("device", "cpu"))

    @staticmethod
    def _load_custom(spec, env):
        if ":" not in spec:
            raise ValueError("format: module:Class (e.g. my_agents:MyAgent)")
        module_name, attr = spec.strip().split(":", 1)
        for extra in (str(paths.ROOT), str(paths.ROOT / "python")):
            if extra not in sys.path:
                sys.path.insert(0, extra)
        factory = getattr(importlib.import_module(module_name), attr)
        required = [p for p in inspect.signature(factory).parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        obj = factory(env) if required else factory()
        if not hasattr(obj, "act"):
            raise TypeError(f"{spec} has no act(obs) method")
        return obj

    def act(self, obs):
        if self.kind == "still":
            return np.zeros(2, np.float32) if self.continuous else 0
        if self.kind == "sb3":
            x = spaces.flatten(self.obs_space, obs) if self.flatten else obs
            with self.model_lock:  # one network shared by every instance's thread
                action, _ = self.impl.predict(x, deterministic=self.deterministic)
            return action
        if self.kind == "heuristic" and self.continuous:
            return np.asarray(self.impl.direction(obs), np.float32)
        action = self.impl.act(obs)
        if self.continuous and np.ndim(action) == 0:
            return DIRECTIONS[int(action)].astype(np.float32)
        return action


# -- catalog (characters, starting weapons, dangers, zones from the game's hello) --------

def load_catalog() -> dict:
    if CATALOG_FILE.is_file():
        return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    return {}


def save_catalog(hello: dict) -> dict:
    data = {k: hello[k] for k in ("characters", "dangers", "zones", "game_version", "mod_version") if k in hello}
    CATALOG_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def refresh_catalog() -> dict:
    """Starts one hidden game instance only to read its hello message."""
    inst = GameInstance(name="catalog")
    try:
        hello = inst.start()
    finally:
        inst.close()
    return save_catalog(hello)


# -- background runner -------------------------------------------------------------------

class Runner:
    """Plays episodes on `instances` game processes at once (one worker thread each).

    Workers take episode numbers from a shared counter, so N episodes are spread over the
    instances and episode i always gets seed + i whichever instance plays it.
    """

    MAX_INSTANCES = 16

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = threading.Event()
        self._reset_status()

    def _reset_status(self):
        self.status = {"state": "idle", "episode": 0, "episodes": 0, "instances": 0, "workers": {}, "current": {},
                       "results": [], "summary": None, "error": None, "started": None}
        self.log = deque(maxlen=400)

    def _log(self, text):
        self.log.append(f"{time.strftime('%H:%M:%S')} {text}")

    @property
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, config: dict):
        with self.lock:
            if self.busy:
                raise RuntimeError("a run is already in progress, stop it first")
            self._reset_status()
            cfg = merge_defaults(config)
            self._apply_model_spec(cfg)
            episodes = int(cfg["run"]["episodes"])
            instances = max(1, min(int(cfg["env"].get("instances", 1)), self.MAX_INSTANCES, episodes))
            self.status.update(state="starting", episodes=episodes, instances=instances, started=time.time())
            self.stop_flag.clear()
            self.thread = threading.Thread(target=self._run, args=(cfg, instances), daemon=True)
            self.thread.start()

    def _apply_model_spec(self, cfg):
        """A trained model only works with the observation layout, action type and frame_skip it
        was trained with: take them from spec.json next to the model (the rest stays as set)."""
        if cfg["agent"].get("type") != "sb3" or not (cfg["agent"].get("sb3_path") or "").strip():
            return
        from .. import spec as specs
        found = specs.find_for_model(_resolve(cfg["agent"]["sb3_path"]))
        if found is None:
            self._log("warning: no spec.json next to the model, using the panel settings")
            return
        s = specs.load(found)
        cfg["encoder"].update(s["encoder"])
        cfg["env"]["action_type"] = s["action_type"]
        cfg["env"]["frame_skip"] = s["frame_skip"]
        self._log(f"model settings from {found}: free_rays={s['encoder']['free_rays']}, "
                  f"sectors={s['encoder']['sectors']}, {s['action_type']}, frame_skip={s['frame_skip']}")

    def stop(self):
        self.stop_flag.set()
        if self.busy:
            self.status["state"] = "stopping"

    def snapshot(self):
        return {**self.status, "workers": {k: dict(v) for k, v in self.status["workers"].items()}, "log": list(self.log)}

    def _run(self, cfg, instances):
        run = cfg["run"]
        self._log(f"start: {instances} instance(s), agent={cfg['agent']['type']}, meta={cfg['meta']['type']}, "
                  f"character={run['character']}, weapon={run['weapon'] or '(first)'}, danger={run['danger']}, "
                  f"speed={cfg['env'].get('speed')}, windows={'yes' if cfg['env']['watch'] else 'no'}")
        layout = tile_windows(instances) if cfg["env"]["watch"] and cfg["env"].get("tile_windows") and instances > 1 else None
        shared = {"model": None, "model_lock": threading.Lock()}
        if cfg["agent"].get("type") == "sb3":
            try:
                shared["model"] = Agent._load_sb3(cfg["agent"])  # one copy of the network for all instances
            except Exception as exc:  # noqa: BLE001 - shown in the panel
                self.status.update(state="error", error=f"{type(exc).__name__}: {exc}")
                self._log(traceback.format_exc(limit=3))
                return
        counter = {"next": 0}
        counter_lock = threading.Lock()

        def next_episode():
            with counter_lock:
                if self.stop_flag.is_set() or counter["next"] >= int(run["episodes"]):
                    return None
                counter["next"] += 1
                return counter["next"] - 1

        workers = []
        for w in range(instances):
            position, resolution = layout[w] if layout else (None, None)
            t = threading.Thread(target=self._worker, args=(w, cfg, shared, next_episode, position, resolution),
                                 daemon=True)
            workers.append(t)
            t.start()
            time.sleep(0.3)  # stagger process starts (each loads the game's .pck files)
        for t in workers:
            t.join()
        self._summarize()
        errors = [v.get("error") for v in self.status["workers"].values() if v.get("error")]
        if errors and not self.status["results"]:
            self.status.update(state="error", error=errors[0])
        else:
            if errors:
                self.status["error"] = f"{len(errors)} of {instances} instances failed: {errors[0]}"
            self.status["state"] = "stopped" if self.stop_flag.is_set() else "done"

    def _worker(self, w, cfg, shared, next_episode, position, resolution):
        ws = self.status["workers"].setdefault(str(w), {"state": "starting", "episode": None, "done": 0})
        env = None
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                env = build_env(cfg, position=position, resolution=resolution, instance_name=f"panel-{os.getpid()}-{w}")
                agent = Agent(cfg["agent"], env, shared)
                seed0 = cfg["run"].get("seed")
                while True:
                    i = next_episode()
                    if i is None:
                        break
                    ws.update(state="running", episode=i + 1)
                    self.status["state"] = "running"
                    self.status["episode"] = max(self.status["episode"], i + 1)
                    seed = None if seed0 in (None, "") else int(seed0) + i
                    result = self._episode(env, agent, seed, ws)
                    if env.hello and not CATALOG_FILE.is_file():
                        save_catalog(env.hello)
                    for warning in caught:
                        self._log(f"[{w}] warning: {warning.message}")
                    caught.clear()
                    if result is None:
                        break
                    result.update(episode=i + 1, instance=w)
                    self.status["results"].append(result)
                    ws["done"] += 1
                    self._log(f"[{w}] episode {i + 1}: {result['outcome']} on wave {result['wave']}, "
                              f"return {result['return']:.2f}, {result['steps']} steps ({result['steps_per_s']}/s)")
            ws["state"] = "stopped" if self.stop_flag.is_set() else "done"
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            ws.update(state="error", error=f"{type(exc).__name__}: {exc}")
            self._log(f"[{w}] " + traceback.format_exc(limit=4))
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass

    def _summarize(self):
        results = sorted(self.status["results"], key=lambda r: r.get("episode", 0))
        self.status["results"] = results
        if results:
            waves = [r["wave"] or 0 for r in results]
            self.status["summary"] = {
                "episodes": len(results), "mean_wave": round(float(np.mean(waves)), 2), "best_wave": int(max(waves)),
                "wins": sum(r["outcome"] == "won" for r in results),
                "mean_return": round(float(np.mean([r["return"] for r in results])), 3),
            }

    def _episode(self, env, agent, seed, ws):
        obs, info = env.reset(seed=seed)
        game_seed = info.get("seed")
        total, steps, start = 0.0, 0, time.perf_counter()
        decisions, parts_sum = {}, {}
        while True:
            if self.stop_flag.is_set():
                return None
            obs, reward, terminated, truncated, info = env.step(agent.act(obs))
            total += reward
            steps += 1
            for d in info["decisions"]:
                key = f"{d['kind']}:{d.get('action')}"
                decisions[key] = decisions.get(key, 0) + 1
            for k, v in (info.get("reward_parts") or {}).items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            if steps % 15 == 0 or terminated or truncated:
                current = {
                    "seed": seed, "game_seed": game_seed, "wave": info["wave"], "hp": info["hp"], "max_hp": info["max_hp"],
                    "level": info["level"], "materials": info["materials"], "kills": info["kills"],
                    "steps": steps, "return": round(total, 3),
                    "steps_per_s": round(steps / max(1e-6, time.perf_counter() - start), 1),
                }
                ws["current"] = current
                self.status["current"] = current
            if terminated or truncated:
                break
        if "error" in info:
            self._log(f"instance error: {info['error']}")
        return {
            "seed": seed, "game_seed": game_seed,
            "outcome": info.get("outcome") or ("error" if "error" in info else "truncated"),
            "wave": info["wave"], "return": round(total, 3), "steps": steps, "kills": info["kills"],
            "materials": info["materials_collected"], "damage_taken": info["damage_taken"],
            "decisions": decisions, "reward_parts": {k: round(v, 3) for k, v in parts_sum.items()},
            "steps_per_s": round(steps / max(1e-6, time.perf_counter() - start), 1),
        }


def run_config(config: dict, print_fn=print) -> dict:
    """Blocking run without the browser (CLI). Returns the final status."""
    runner = Runner()
    runner.start(config)
    seen = 0
    try:
        while runner.busy:
            time.sleep(0.5)
            log = list(runner.log)
            for line in log[seen:]:
                print_fn(line)
            seen = len(log)
    except KeyboardInterrupt:
        runner.stop()
        runner.thread.join()
    for line in list(runner.log)[seen:]:
        print_fn(line)
    return runner.snapshot()
