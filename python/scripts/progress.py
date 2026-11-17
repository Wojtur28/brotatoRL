"""Show how a model improved during training: a chart over checkpoints, plus replays.

    python scripts/progress.py --runs ppo1 ppo2_noshape                        # from the evaluations logged in training
    python scripts/progress.py --runs ppo3 --evaluate --every 2 --episodes 8   # re-evaluate checkpoints on common seeds
    python scripts/progress.py --runs ppo3 --replays --every 4                 # one replay per checkpoint, same seed

Output in showcase/:
  progress.html                      chart (mean wave vs training steps, +-1 standard error) + table
  checkpoint_evals/<run>.json        --evaluate results, cached per checkpoint (re-runs skip done ones)
  replays/<run>/<run>_<steps>_seed<s>.json.gz   --replays, open several in viewer/replay.html side by side

Without --evaluate the chart uses evals.jsonl from training. Those are VALIDATION runs (the
seeds best/ was chosen on); --evaluate plays every chosen checkpoint on the same --seed range,
which also makes runs with different validation seeds comparable.
"""
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

from brotato_rl import BrotatoEnv, paths, runs, spec as specs
from brotato_rl.evaluation import evaluate_model, run_episode
from brotato_rl.policy import check_spaces, load_sb3
from brotato_rl.replay import ReplayRecorder

SHOWCASE = paths.ROOT / "showcase"


def pick(checkpoints, every):
    chosen = checkpoints[::max(1, every)]
    if checkpoints and checkpoints[-1] not in chosen:
        chosen.append(checkpoints[-1])
    return chosen


def training_evals(run):
    path = runs.run_path(run) / "evals.jsonl"
    if not path.is_file():
        return []
    points = {}  # by steps: --resume from a checkpoint just below an eval step repeats that evaluation
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            s = json.loads(line)
            points[s["timesteps"]] = {"steps": s["timesteps"], "mean": s["mean_wave"], "sem": s.get("sem_wave") or 0,
                                      "wins": s.get("wins", 0), "episodes": s.get("episodes")}
    return [points[k] for k in sorted(points)]


def evaluate_checkpoints(run, args):
    out = SHOWCASE / "checkpoint_evals" / f"{runs.run_path(run).name}.json"
    cache = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    seeds = list(range(args.seed, args.seed + args.episodes))
    spec = specs.load(runs.run_path(run) / specs.SPEC_FILE)
    for steps, path in pick(runs.checkpoints(run), args.every):
        key = f"{steps}:{seeds[0]}-{seeds[-1]}"
        if key in cache:
            continue
        print(f"{runs.label(run, steps)}: {len(seeds)} episodes on {args.workers} games", flush=True)
        model = load_sb3(path, threads=2)
        summary, episodes = evaluate_model(model, lambda: specs.env_kwargs(spec), seeds, n_envs=args.workers)
        cache[key] = {"steps": steps, "mean": summary["mean_wave"], "sem": summary["sem_wave"] or 0,
                      "wins": summary["wins"], "episodes": len(episodes), "seeds": [seeds[0], seeds[-1]],
                      "waves": [e["wave"] for e in episodes]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"  mean wave {summary['mean_wave']} +- {summary['sem_wave']}", flush=True)
    wanted = f":{seeds[0]}-{seeds[-1]}"
    return sorted((v for k, v in cache.items() if k.endswith(wanted)), key=lambda p: p["steps"])


def record_replays(run, args):
    spec = specs.load(runs.run_path(run) / specs.SPEC_FILE)
    env = BrotatoEnv(**specs.env_kwargs(spec))
    folder = SHOWCASE / "replays" / runs.run_path(run).name
    saved = []
    try:
        for steps, path in pick(runs.checkpoints(run), args.every):
            name = runs.label(run, steps)
            target = folder / f"{name.replace(' @ ', '_')}_seed{args.replay_seed}.json.gz"
            if target.is_file():
                saved.append(target)
                continue
            model = load_sb3(path, threads=2)
            check_spaces(model, env)
            rec = ReplayRecorder({"title": name, "model": str(path), "steps_trained": steps, "seed": args.replay_seed})
            result = run_episode(env, lambda obs: model.predict(obs, deterministic=True)[0], args.replay_seed,
                                 recorder=rec)
            saved.append(rec.save(target))
            print(f"{name}: {result['outcome']} at wave {result['wave']} -> {target}", flush=True)
    finally:
        env.close()
    return saved


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>BrotatoRL – training progress</title>
<style>
.viz { color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink-2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; } }
body { margin:0; }
.viz { background:var(--page); color:var(--ink); font:14px/1.45 system-ui,"Segoe UI",sans-serif; min-height:100vh; padding:16px; box-sizing:border-box; }
.card { background:var(--surface); border:1px solid var(--grid); border-radius:8px; padding:16px; max-width:1100px; margin:0 auto 16px; }
h1 { font-size:18px; margin:0 0 4px; } .sub { color:var(--ink-2); margin:0 0 12px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; color:var(--ink-2); font-size:13px; margin-bottom:8px; }
.legend i { display:inline-block; width:14px; height:2px; vertical-align:middle; margin-right:6px; }
svg { width:100%; height:auto; display:block; overflow:visible; }
.tip { position:fixed; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--axis);
  border-radius:6px; padding:6px 8px; font-size:12px; display:none; box-shadow:0 2px 8px rgba(0,0,0,.15); }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:13px; }
th, td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--grid); } th:first-child, td:first-child { text-align:left; }
th { color:var(--ink-2); font-weight:600; } details summary { cursor:pointer; color:var(--ink-2); }
.note { color:var(--muted); font-size:12px; } a { color:var(--s1); }
</style></head>
<body><div class="viz">
<div class="card">
  <h1>Training progress: mean wave</h1>
  <p class="sub">Mean wave on which a run ends (horizontal axis: training steps). Band around the line = ±1 standard error.
  Horizontal lines = baselines passed with --baseline (evaluate.py results, e.g. of the heuristic bot).</p>
  <div class="legend" id="legend"></div>
  <svg id="chart" viewBox="0 0 960 420" role="img" aria-label="Mean wave vs training steps"></svg>
  <p class="note" id="source"></p>
</div>
<div class="card"><details><summary>Data table</summary><div id="table"></div></details></div>
<div class="card" id="replays"></div>
<div class="tip" id="tip"></div>
</div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const svg = document.getElementById("chart"), tip = document.getElementById("tip");
const L = 44, T = 16, B = 44;
let W = 960, H = 420, R = 150, cross = null;
const allPts = DATA.series.flatMap(s => s.points);
const maxX = Math.max(1, ...allPts.map(p => p.steps)) * 1.02;
const BASES = DATA.baselines;
const maxY = Math.min(20, Math.max(5, Math.ceil(Math.max(...allPts.map(p => p.mean + p.sem), ...BASES.map(b => b.mean), 0) + 1)));
const x = v => L + v / maxX * (W - L - R), y = v => T + (1 - v / maxY) * (H - T - B);
const el = (n, a, p = svg) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
const fmtM = v => v === 0 ? "0" : (v / 1e6).toFixed(v < 1e6 ? 2 : 1) + "M";
const legend = document.getElementById("legend");
DATA.series.forEach((s, i) => legend.insertAdjacentHTML("beforeend", `<span><i style="background:var(--s${i + 1})"></i>${s.name}</span>`));
const metaName = b => b.weapon && b.weapon !== "any" ? b.weapon : "general";
BASES.forEach((b, i) => legend.insertAdjacentHTML("beforeend",
  `<span><i style="background:var(--ink-2)"></i>heuristic bot, ${metaName(b)} meta${BASES.length > 1 ? (i ? " (dashed line)" : " (solid line)") : ""}</span>`));

function drawChart() {
  // Drawn in real pixels of the current width, so text stays 12 px on phones and wide screens.
  W = Math.max(300, Math.round(svg.parentElement.clientWidth - 32));
  H = Math.round(Math.min(440, Math.max(260, W * 0.45)));
  R = W < 560 ? 84 : 140;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  svg.replaceChildren();
  for (let v = 0; v <= maxY; v += maxY > 12 ? 2 : 1) {
    el("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: "var(--grid)", "stroke-width": 1});
    el("text", {x: L - 8, y: y(v) + 4, "text-anchor": "end", fill: "var(--muted)", "font-size": 12}).textContent = v;
  }
  const ticks = Math.max(2, Math.floor((W - L - R) / 90));
  const raw = maxX / ticks, mag = 10 ** Math.floor(Math.log10(raw));
  const stepX = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw);
  for (let v = 0; v <= maxX; v += stepX) {
    el("text", {x: x(v), y: H - B + 18, "text-anchor": "middle", fill: "var(--muted)", "font-size": 12}).textContent = fmtM(v);
  }
  el("line", {x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: "var(--axis)", "stroke-width": 1});
  el("text", {x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", fill: "var(--ink-2)", "font-size": 12}).textContent = "training steps";
  el("text", {x: 12, y: (T + H - B) / 2, "text-anchor": "middle", fill: "var(--ink-2)", "font-size": 12,
    transform: `rotate(-90 12 ${(T + H - B) / 2})`}).textContent = "mean wave";
  const labelYs = [];
  const place = (want) => { let ly = want; while (labelYs.some(v => Math.abs(v - ly) < 15)) ly += 15; labelYs.push(ly); return ly; };
  BASES.forEach((b, i) => {
    el("line", {x1: L, x2: W - R, y1: y(b.mean), y2: y(b.mean), stroke: "var(--ink-2)", "stroke-width": 1.5,
      "stroke-dasharray": i ? "6 4" : "none"});
    el("text", {x: W - R + 8, y: place(y(b.mean) + 4), fill: "var(--ink-2)", "font-size": 12})
      .textContent = `bot${metaName(b) === "general" ? "" : " " + metaName(b)} ${b.mean.toFixed(2)}`;
  });
  DATA.series.forEach((s, i) => {
    const color = `var(--s${i + 1})`, pts = s.points;
    if (!pts.length) return;
    const band = pts.map(p => `${x(p.steps)},${y(p.mean + p.sem)}`).concat(pts.slice().reverse().map(p => `${x(p.steps)},${y(Math.max(0, p.mean - p.sem))}`));
    el("polygon", {points: band.join(" "), fill: color, "fill-opacity": 0.14});
    el("polyline", {points: pts.map(p => `${x(p.steps)},${y(p.mean)}`).join(" "), fill: "none", stroke: color,
      "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"});
    pts.forEach(p => el("circle", {cx: x(p.steps), cy: y(p.mean), r: 4, fill: color, stroke: "var(--surface)", "stroke-width": 2}));
    const last = pts[pts.length - 1];
    el("text", {x: W - R + 8, y: place(y(last.mean) + 4), fill: "var(--ink)", "font-size": 12})
      .textContent = `${s.short} ${last.mean.toFixed(2)}`;
  });
  cross = el("line", {y1: T, y2: H - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden"});
}
drawChart();
new ResizeObserver(() => drawChart()).observe(svg.parentElement);
svg.addEventListener("pointermove", ev => {
  const box = svg.getBoundingClientRect(), sx = (ev.clientX - box.left) / box.width * W;
  if (sx < L || sx > W - R) { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); return; }
  const steps = (sx - L) / (W - L - R) * maxX;
  const rows = DATA.series.map((s, i) => {
    if (!s.points.length) return "";
    const p = s.points.reduce((a, b) => Math.abs(b.steps - steps) < Math.abs(a.steps - steps) ? b : a);
    return `<div><b style="color:var(--s${i + 1})">●</b> ${s.short}: ${fmtM(p.steps)} → wave <b>${p.mean.toFixed(2)}</b> ±${p.sem.toFixed(2)}` +
      `${p.wins ? `, wins ${p.wins}/${p.episodes}` : ""}</div>`;
  }).join("");
  cross.setAttribute("x1", sx); cross.setAttribute("x2", sx); cross.setAttribute("visibility", "visible");
  tip.innerHTML = rows; tip.style.display = "block";
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = ev.clientY + 14 + "px";
});
svg.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
document.getElementById("source").textContent = DATA.source;
document.getElementById("table").innerHTML = "<table><tr><th>series</th><th>steps</th><th>mean wave</th><th>± std. error</th><th>wins</th><th>episodes</th></tr>" +
  DATA.series.flatMap(s => s.points.map(p => `<tr><td>${s.name}</td><td>${p.steps.toLocaleString("en")}</td><td>${p.mean.toFixed(2)}</td><td>${p.sem.toFixed(2)}</td><td>${p.wins}</td><td>${p.episodes ?? ""}</td></tr>`)).join("") + "</table>";
const rep = document.getElementById("replays");
rep.innerHTML = DATA.replays.length ? `<h1>Replays</h1><p class="sub">Open <code>viewer/replay.html</code> and drag the files in (several at once = side-by-side comparison):</p><ul>` +
  DATA.replays.map(r => `<li><code>${r}</code></li>`).join("") + "</ul>" : `<p class="note">No replays (python scripts/progress.py --runs ... --replays).</p>`;
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run names under runs/ (up to 4)")
    parser.add_argument("--evaluate", action="store_true", help="play the checkpoints on --seed.. (else training evals)")
    parser.add_argument("--replays", action="store_true", help="record one replay per chosen checkpoint")
    parser.add_argument("--every", type=int, default=2, help="use every N-th checkpoint (the last one always)")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30000, help="first seed of --evaluate (30000+ = showcase seeds)")
    parser.add_argument("--replay-seed", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=2, help="games in parallel for --evaluate (~0.5 GB RAM each)")
    parser.add_argument("--baseline", nargs="*", default=[],
                        help="results/*.json files from evaluate.py (e.g. of the bot), one horizontal line each")
    parser.add_argument("--out", default=str(SHOWCASE / "progress.html"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore", message="Steam is running")

    series, replays = [], []
    for run in args.runs[:4]:
        name = runs.run_path(run).name
        if args.evaluate:
            points = evaluate_checkpoints(run, args)
            label = f"{name} (checkpoints, seeds {args.seed}–{args.seed + args.episodes - 1})"
        else:
            points = training_evals(run)
            label = f"{name} (evaluations during training)"
        series.append({"name": label, "short": name, "points": points})
        if args.replays:
            replays += [str(p.relative_to(paths.ROOT)) for p in record_replays(run, args)]
    baselines = []
    for path in map(Path, args.baseline):
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            b = result["summary"]
            meta = ((result.get("config") or {}).get("spec") or {}).get("meta") or {}
            baselines.append({"mean": b["mean_wave"], "sem": b.get("sem_wave") or 0, "weapon": meta.get("weapon_type"),
                              "file": path.name})
    source = ("Data: " + ("checkpoints re-evaluated (--evaluate)" if args.evaluate else "evals.jsonl from training (validation seeds)")
              + (f"; bot: {', '.join(b['file'] for b in baselines)}" if baselines else "")
              + f"; generated {datetime.now():%Y-%m-%d %H:%M}.")
    data = {"series": series, "baselines": baselines, "source": source, "replays": replays}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    print(f"saved {out}")
    for s in series:
        if s["points"]:
            print(f"  {s['name']}: {len(s['points'])} points, last mean wave {s['points'][-1]['mean']}")


if __name__ == "__main__":
    main()
