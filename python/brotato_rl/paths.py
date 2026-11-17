"""Filesystem layout of the BrotatoRL project."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "lab"
GAME_DIR = LAB / "game"
INSTANCES_DIR = LAB / "instances"
MOD_SRC = ROOT / "mod" / "rl_bridge"
MOD_ID = "BrotatoRL-RLBridge"
LAB_FILES = Path(__file__).resolve().parent / "lab_files"

STEAM_APP_ID = "1942280"
# Files copied from the Steam install into the lab.
GAME_FILES = ("Brotato.exe", "Brotato.pck", "steam_api64.dll", "steam_data.json")
# Up to 1.1.15.x the Abyssal Terrors DLC ships as a separate pack; later versions merge it into Brotato.pck.
OPTIONAL_GAME_FILES = ("BrotatoAbyssalTerrors.pck",)
# The game version the RLBridge mod was written and tested against.
TESTED_GAME_VERSION = "1.1.15.4"
GAME_EXE = GAME_DIR / "Brotato.exe"
# The lab copy uses its own user:// folder name, so it can never open the real saves.
LAB_USER_DIR_NAME = "BrotatoRL"


def _steam_roots() -> list:
    """Steam install folders, from the registry when available, else the default location."""
    roots = []
    try:
        import winreg

        for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
            try:
                with winreg.OpenKey(hive, key) as k:
                    for name in ("SteamPath", "InstallPath"):
                        try:
                            roots.append(Path(winreg.QueryValueEx(k, name)[0]))
                        except OSError:
                            pass
            except OSError:
                pass
    except ImportError:  # not Windows
        pass
    roots.append(Path(r"C:\Program Files (x86)\Steam"))
    return roots


def find_steam_game_dir() -> Path:
    """Locate the Brotato install in any Steam library. Only ever read, never written."""
    candidates = []
    for root in _steam_roots():
        candidates.append(root / "steamapps" / "common" / "Brotato")
        vdf = root / "steamapps" / "libraryfolders.vdf"
        try:
            text = vdf.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lib in re.findall(r'"path"\s+"([^"]+)"', text):
            candidates.append(Path(lib.replace("\\\\", "\\")) / "steamapps" / "common" / "Brotato")
    for path in candidates:
        if (path / "Brotato.exe").is_file():
            return path
    return candidates[0]


# Override with the BROTATO_GAME_DIR environment variable or setup_lab.py --game-dir.
STEAM_GAME_DIR = Path(os.environ["BROTATO_GAME_DIR"]) if os.environ.get("BROTATO_GAME_DIR") else find_steam_game_dir()
