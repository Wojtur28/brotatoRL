"""Models inside a training run (runs/<name>/): checkpoints by step count, best, final, bc.

    path, steps = resolve("ppo1", "2.5M")   # nearest checkpoint to 2.5 million steps
    path, steps = resolve("ppo1", "best")   # best/model.zip and the step count it was saved at
"""
import json
import zipfile
from pathlib import Path

from . import paths

RUNS_DIR = paths.ROOT / "runs"


def run_path(run) -> Path:
    p = Path(run)
    return p if p.is_dir() else RUNS_DIR / str(run)


def model_steps(model_zip) -> int:
    """num_timesteps stored inside an SB3 model zip (0 if unreadable)."""
    try:
        with zipfile.ZipFile(model_zip) as zf:
            return int(json.loads(zf.read("data")).get("num_timesteps", 0))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return 0


def checkpoints(run) -> list:
    """[(steps, path)] of runs/<run>/checkpoints/model_<steps>_steps.zip, sorted."""
    found = []
    for p in (run_path(run) / "checkpoints").glob("model_*_steps.zip"):
        try:
            found.append((int(p.stem.split("_")[1]), p))
        except (IndexError, ValueError):
            continue
    return sorted(found)


def parse_steps(text) -> int:
    t = str(text).strip().lower().replace("_", "").replace(",", ".")
    for suffix, mult in (("m", 1_000_000), ("k", 1_000)):
        if t.endswith(suffix):
            return int(float(t[:-1]) * mult)
    return int(float(t))


def resolve(run, at="best"):
    """(model path, steps) for at = "best" | "final" | "interrupted" | "bc" | a step count ("2.5M", "750k",
    "1250000"; the nearest checkpoint is used)."""
    run_dir = run_path(run)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run folder {run_dir}")
    if str(at) in ("best", "final", "interrupted", "bc"):
        path = run_dir / str(at) / "model.zip"
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        return path, model_steps(path)
    target = parse_steps(at)
    cps = checkpoints(run_dir)
    if not cps:
        raise FileNotFoundError(f"no checkpoints in {run_dir / 'checkpoints'}")
    steps, path = min(cps, key=lambda c: abs(c[0] - target))
    return path, steps


def label(run, steps) -> str:
    name = run_path(run).name
    if steps >= 1_000_000:
        return f"{name} @ {steps / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    return f"{name} @ {round(steps / 1e3)}k"
