"""Where does the damage come from in one wave? Reads replays (play.py --replay-dir).

    python scripts/wave_damage.py --wave 8 "showcase/replays/wave8/*.json.gz"

Replays are grouped by the file name before "_seed" (e.g. ppo3_8M, heuristic). Every HP loss in the
wave is attributed to its likely source: a dashing charger within 140 px ("dash"), else the
nearest enemy within 90 px ("contact"), else an enemy projectile within 70 px ("projectile").
A heuristic: the replay keeps positions every step (4 frames), not the game's hit events.
"""
import argparse
import collections
import glob
import math
from pathlib import Path

from brotato_rl.replay import load

CLS = {0: "contact", 1: "shooter", 2: "charger", 3: "spawner"}


def enemies(frame):
    e = frame.get("e") or []
    return [e[i:i + 6] for i in range(0, len(e), 6)]


def projectiles(frame):
    p = frame.get("pr") or []
    return [p[i:i + 2] for i in range(0, len(p), 2)]


def cause(prev, cur):
    px, py = prev["p"][0], prev["p"][1]
    near = []
    for fr in (prev, cur):
        for x, y, kind, cls, st, hp in enemies(fr):
            near.append((math.hypot(x - px, y - py), cls, st, kind))
    near.sort()
    proj = min((math.hypot(x - px, y - py) for fr in (prev, cur) for x, y in projectiles(fr)), default=1e9)
    dash = [n for n in near if n[1] == 2 and n[2] == 2 and n[0] < 140]
    if dash:
        return "dash"
    if near and near[0][0] < 90:
        d, cls, st, kind = near[0]
        return f"contact: {CLS.get(cls, cls)}" + (" (boss/elite)" if kind else "")
    if proj < 70:
        return "projectile"
    return "other (no enemy or projectile nearby)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, default=8)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    files = sorted(f for pattern in args.files for f in glob.glob(pattern))
    groups = collections.defaultdict(list)
    for f in files:
        r = load(f)
        meta, frames = r["meta"], r["frames"]
        who = Path(f).name.split("_seed")[0]
        wf = [fr for fr in frames if fr["w"] == args.wave]
        if not wf:
            groups[who].append({"seed": meta.get("seed"), "reached": False, "end_wave": meta.get("wave")})
            continue
        dmg = collections.Counter()
        for prev, cur in zip(wf, wf[1:]):
            loss = prev["p"][2] - cur["p"][2]
            if loss > 0:
                dmg[cause(prev, cur)] += loss
        first, last = wf[0], wf[-1]
        n_charge = max(sum(1 for e in enemies(fr) if e[3] == 2) for fr in wf)
        n_enem = max(len(enemies(fr)) for fr in wf)
        died = meta.get("wave") == args.wave and meta.get("outcome") != "win"
        groups[who].append({
            "seed": meta.get("seed"), "reached": True, "died": died, "end_wave": meta.get("wave"),
            "start_hp": f"{first['p'][2]}/{first['p'][3]}", "lv": first.get("lv"), "mat": first.get("m"),
            "time_left_at_end": last.get("tl"), "max_chargers": n_charge, "max_enemies": n_enem,
            "dmg": dict(dmg.most_common()), "dmg_total": sum(dmg.values()),
        })
    for who, rows in groups.items():
        print(f"=== {who} (wave {args.wave}) ===")
        total = collections.Counter()
        for r in sorted(rows, key=lambda r: r["seed"] or 0):
            if not r["reached"]:
                print(f"  seed {r['seed']}: did not reach it (ended on wave {r['end_wave']})")
                continue
            total.update(r["dmg"])
            print(f"  seed {r['seed']}: {'DIED' if r['died'] else 'survived'} (ended on wave {r['end_wave']}), "
                  f"start HP {r['start_hp']}, level {r['lv']}, materials {r['mat']}, "
                  f"time left {r['time_left_at_end']} s, max chargers {r['max_chargers']}, max enemies {r['max_enemies']}, "
                  f"damage {r['dmg_total']}: {r['dmg']}")
        s = sum(total.values()) or 1
        print("  TOTAL:", {k: f"{v} ({100 * v / s:.0f}%)" for k, v in total.most_common()})


if __name__ == "__main__":
    main()
