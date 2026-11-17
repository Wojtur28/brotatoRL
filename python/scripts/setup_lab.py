"""Prepare (or refresh) the lab copy of the game and build the RLBridge mod.

    python scripts/setup_lab.py                 # first setup, or after editing the mod
    python scripts/setup_lab.py --update-game   # re-copy game files after a Steam update
"""
import argparse
from pathlib import Path

from brotato_rl import lab, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game-dir", type=Path, default=paths.STEAM_GAME_DIR, help="Steam install (read only)")
    parser.add_argument("--update-game", action="store_true", help="re-copy game files that differ from Steam")
    args = parser.parse_args()

    print(f"game: {args.game_dir}")
    result = lab.setup(update_game=args.update_game, source_dir=args.game_dir)
    print(f"lab: {paths.GAME_DIR}")
    for name, status in result["game_files"]:
        print(f"  {name:28s} {status}")
    print(f"mod: {result['mod_zip']}")
    print(f"note: the mod is tested with Brotato {paths.TESTED_GAME_VERSION}; other versions may not work.")


if __name__ == "__main__":
    main()
