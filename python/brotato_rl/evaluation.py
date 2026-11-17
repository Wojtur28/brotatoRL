"""Episode metrics shared by scripts/evaluate.py and the evaluation inside scripts/train.py.

Per episode: outcome, wave, return, kills, materials, damage, position metrics
  corner_frac  share of combat steps closer than `margin` px to a vertical AND a horizontal wall
  wall_frac    share of combat steps closer than `margin` px to any wall (corners included)
  freedom      mean escape freedom (space.py: 1 open, ~0.56 at a wall, ~0.31 empty corner)
  death_place  where the last combat observation put the character: corner / wall / open
and, for won runs, win_by:
  "boss_kill"  the bosses of wave 20 died and the game cut the wave short (main.gd sets the wave
               timer to 0.1 s), seen as wave_time_left dropping > 0.5 s between two observations
               (normal drop: frame_skip/60 s)
  "timer"      the character survived until the wave-20 timer ran out
"""
import threading
import time

import numpy as np

from .space import freedom_from_raw

TIMER_JUMP_S = 0.5


def place(state, margin=200.0):
    player, zone = (state or {}).get("player") or {}, (state or {}).get("zone")
    if not zone or not player.get("pos"):
        return None
    px, py = player["pos"]
    x0, y0, x1, y1 = zone
    near_x = min(px - x0, x1 - px) < margin
    near_y = min(py - y0, y1 - py) < margin
    return "corner" if near_x and near_y else "wall" if near_x or near_y else "open"


def run_episode(env, act, seed=None, margin=200.0, stop=None, recorder=None):
    """Plays one episode on a BrotatoEnv with act(obs) -> action. Returns the metrics dict,
    or None if `stop` (threading.Event) was set. recorder: a replay.ReplayRecorder (optional)."""
    obs, info = env.reset(seed=seed)
    if recorder is not None:
        recorder.add(env.raw_state, None, info)
    places, freedoms = [], []
    total, steps, start = 0.0, 0, time.perf_counter()
    last_combat = env.raw_state
    timer_jump = False
    while True:
        if stop is not None and stop.is_set():
            return None
        action = act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if recorder is not None:
            recorder.add(env.raw_state, action, info)
        total += reward
        steps += 1
        state = env.raw_state
        where = place(state, margin)
        if where is not None:
            if (state.get("wave") == state.get("nb_waves") == last_combat.get("wave")
                    and last_combat.get("wave_time_left") is not None and state.get("wave_time_left") is not None
                    and last_combat["wave_time_left"] - state["wave_time_left"] > TIMER_JUMP_S):
                timer_jump = True
            places.append(where)
            freedoms.append(freedom_from_raw(state))
            last_combat = state
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - start
    outcome = info.get("outcome") or ("error" if "error" in info else "truncated")
    death = place(last_combat, margin) if outcome == "died" else None
    n = max(1, len(places))
    if recorder is not None:
        recorder.finish(info, outcome=outcome, seed=seed, episode_return=round(total, 3))
    return {
        "seed": seed,
        "outcome": outcome,
        "win_by": ("boss_kill" if timer_jump else "timer") if outcome == "won" else None,
        "error": info.get("error"),
        "wave": info["wave"],
        "steps": steps,
        "return": round(total, 3),
        "kills": info["kills"],
        "materials_collected": info["materials_collected"],
        "damage_taken": info["damage_taken"],
        "corner_frac": round(places.count("corner") / n, 4),
        "wall_frac": round((places.count("corner") + places.count("wall")) / n, 4),
        "freedom": round(float(np.mean(freedoms)), 4) if freedoms else None,
        "death_place": death,
        "death_freedom": round(freedom_from_raw(last_combat), 4) if death else None,
        "seconds": round(elapsed, 1),
        "steps_per_s": round(steps / max(elapsed, 1e-6), 1),
    }


def summarize(episodes):
    waves = [e["wave"] or 0 for e in episodes]
    deaths = [e["death_place"] for e in episodes if e["death_place"]]
    freedoms = [e["freedom"] for e in episodes if e["freedom"] is not None]
    return {
        "episodes": len(episodes),
        "mean_wave": round(float(np.mean(waves)), 2),
        "sem_wave": round(float(np.std(waves, ddof=1) / np.sqrt(len(waves))), 2) if len(waves) > 1 else None,
        "median_wave": float(np.median(waves)),
        "best_wave": int(max(waves)),
        "wins": sum(e["outcome"] == "won" for e in episodes),
        "wins_by": {k: sum(e.get("win_by") == k for e in episodes) for k in ("timer", "boss_kill")},
        "errors": sum(e["outcome"] == "error" for e in episodes),
        "mean_return": round(float(np.mean([e["return"] for e in episodes])), 2),
        "corner_frac": round(float(np.mean([e["corner_frac"] for e in episodes])), 4),
        "wall_frac": round(float(np.mean([e["wall_frac"] for e in episodes])), 4),
        "freedom": round(float(np.mean(freedoms)), 4) if freedoms else None,
        "deaths_by_place": {p: deaths.count(p) for p in ("corner", "wall", "open")},
        "total_steps": int(sum(e["steps"] for e in episodes)),
    }


def evaluate_model(model, env_kwargs, seeds, n_envs=4, deterministic=True, margin=200.0, stop=None, log=None):
    """Plays `seeds` (one episode each) on n_envs fresh games in threads with one shared model.

    env_kwargs: a callable returning BrotatoEnv kwargs (called once per game, so every game gets
    its own meta policy object; RuleMetaPolicy keeps per-shop-visit state) or a spec dict.
    Games are started here and closed at the end, so they only take RAM during the evaluation.
    Returns (summary, episodes). Episodes are sorted by seed.
    """
    from . import spec as specs
    from .env import BrotatoEnv

    make_kwargs = env_kwargs if callable(env_kwargs) else (lambda: specs.env_kwargs(env_kwargs))
    lock = threading.Lock()
    queue, episodes, errors = list(seeds), [], []

    def act(obs):
        with lock:  # the network is shared between threads
            return model.predict(obs, deterministic=deterministic)[0]

    def worker(w):
        env = None
        try:
            env = BrotatoEnv(**make_kwargs(), instance_name=f"eval-{id(queue)}-{w}")
            while True:
                with lock:
                    if not queue or (stop is not None and stop.is_set()):
                        return
                    seed = queue.pop(0)
                result = run_episode(env, act, seed, margin, stop)
                if result is not None:
                    with lock:
                        episodes.append(result)
                    if log:
                        log(f"eval seed {seed}: {result['outcome']} at wave {result['wave']}")
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            errors.append(f"{type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                env.close()

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(min(n_envs, len(queue)))]
    for t in threads:
        t.start()
        time.sleep(0.2)
    for t in threads:
        t.join()
    if errors and not episodes:
        raise RuntimeError("evaluation failed: " + errors[0])
    episodes.sort(key=lambda e: e["seed"])
    summary = summarize(episodes)
    summary["worker_errors"] = errors
    return summary, episodes
