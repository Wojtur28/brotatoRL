"""Save a trained model as a self-contained version in models/<name>/ (safe from new training runs).

    python scripts/export_model.py --run ppo1 --name ppo1_v1                  # runs/ppo1/best/model.zip
    python scripts/export_model.py --run ppo1 --name ppo1_5M --which final
    python scripts/export_model.py --run ppo1 --name ppo1_3M --which checkpoints/model_2999976_steps.zip

models/<name>/ gets:
  model.zip     the SB3 model
  spec.json     its environment (play.py / evaluate.py read it automatically)
  config.json   the full training config
  card.json     where it came from + the evaluation of the run (and of the full results/ file, if given)
To share it, zip the folder; the receiver needs this project and their own copy of the game.
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from brotato_rl import paths

RUNS, MODELS = paths.ROOT / "runs", paths.ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder under runs/")
    parser.add_argument("--name", required=True, help="folder name under models/")
    parser.add_argument("--which", default="best", help="best | final | interrupted | a path inside the run")
    parser.add_argument("--results", help="results/*.json of an evaluate.py run of this model (copied into card.json)")
    parser.add_argument("--note", default="", help="free text for card.json")
    args = parser.parse_args()

    run = RUNS / args.run
    source = run / args.which / "model.zip" if (run / args.which).is_dir() else run / args.which
    if not source.is_file():
        raise SystemExit(f"model not found: {source}")
    out = MODELS / args.name
    if out.exists():
        raise SystemExit(f"{out} already exists; pick another --name")
    out.mkdir(parents=True)
    shutil.copy2(source, out / "model.zip")
    for name in ("spec.json", "config.json"):
        if (run / name).is_file():
            shutil.copy2(run / name, out / name)

    card = {"name": args.name, "exported": datetime.now().isoformat(timespec="seconds"), "note": args.note,
            "source": str(source.relative_to(paths.ROOT))}
    best_eval = run / "best" / "eval.json"
    if args.which == "best" and best_eval.is_file():
        card["training_eval"] = json.loads(best_eval.read_text(encoding="utf-8"))["summary"]
    if args.results:
        data = json.loads(Path(args.results).read_text(encoding="utf-8"))
        card["evaluation"] = {"file": Path(args.results).name, "summary": data["summary"]}
    (out / "card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    print(f"saved {out}:", ", ".join(p.name for p in out.iterdir()))
    print(f"watch it:  python python\\scripts\\play.py --model {out.relative_to(paths.ROOT)}\\model.zip --watch")


if __name__ == "__main__":
    main()
