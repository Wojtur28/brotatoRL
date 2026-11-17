"""Builds the lab: a private copy of the game that loads the RLBridge mod.

The Steam build of Brotato only loads Workshop mods. The lab copy gets an override.cfg that
swaps the ModLoaderStore autoload for a small shim, so it loads local mod zips like the
GOG/Epic builds do. The Steam install itself is only read.
"""
import hashlib
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from . import paths


class LabError(RuntimeError):
    pass


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_game(source_dir: Path = paths.STEAM_GAME_DIR, update: bool = False) -> list:
    """Copy the game files into the lab. Existing copies are kept unless update=True.

    Keeping the copy pinned means a Steam update cannot silently change the training
    environment (or break the mod) in the middle of an experiment.
    """
    source_dir = Path(source_dir)
    if not (source_dir / "Brotato.exe").is_file():
        raise LabError(f"Brotato.exe not found in {source_dir} (set BROTATO_GAME_DIR)")
    paths.GAME_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for name in paths.GAME_FILES + paths.OPTIONAL_GAME_FILES:
        src, dst = source_dir / name, paths.GAME_DIR / name
        if not src.is_file():
            if name not in paths.OPTIONAL_GAME_FILES:
                raise LabError(f"missing game file {src}")
            if not dst.is_file():
                status = "not in this game version"
            elif update:
                # A pack left over from an older version must not load on top of the new one.
                dst.unlink()
                status = "removed (not in this game version)"
            else:
                status = "kept (not in Steam copy)"
            report.append((name, status))
            continue
        if dst.is_file() and not update:
            status = "kept" if dst.stat().st_size == src.stat().st_size else "kept (differs from Steam copy)"
        elif dst.is_file() and _file_digest(dst) == _file_digest(src):
            status = "up to date"
        else:
            shutil.copy2(src, dst)
            status = "copied"
        report.append((name, status))
    return report


def write_lab_files() -> None:
    game = paths.GAME_DIR
    (game / "steam_appid.txt").write_text(paths.STEAM_APP_ID + "\n", encoding="ascii")
    shim_dir = game / "rl_shim"
    shim_dir.mkdir(exist_ok=True)
    shim = shim_dir / "mod_loader_store_shim.gd"
    shutil.copy2(paths.LAB_FILES / "mod_loader_store_shim.gd", shim)
    override = (
        "[application]\n\n"
        f'config/custom_user_dir_name="{paths.LAB_USER_DIR_NAME}"\n\n'
        "[autoload]\n\n"
        f'ModLoaderStore="*{shim.as_posix()}"\n'
    )
    (game / "override.cfg").write_text(override, encoding="utf-8")


def build_mod(src_root: Path = paths.MOD_SRC, mod_id: str = paths.MOD_ID) -> Path:
    src = Path(src_root) / "mods-unpacked" / mod_id
    if not (src / "manifest.json").is_file():
        raise LabError(f"no manifest.json in {src}")
    mods_dir = paths.GAME_DIR / "mods"
    mods_dir.mkdir(parents=True, exist_ok=True)
    out = mods_dir / f"{mod_id}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, PurePosixPath("mods-unpacked", mod_id, *path.relative_to(src).parts))
    return out


def check_lab() -> None:
    """Raise LabError with a fix-it hint when the lab is incomplete."""
    missing = [n for n in paths.GAME_FILES if not (paths.GAME_DIR / n).is_file()]
    for extra in ("override.cfg", "steam_appid.txt", "rl_shim/mod_loader_store_shim.gd", f"mods/{paths.MOD_ID}.zip"):
        if not (paths.GAME_DIR / extra).is_file():
            missing.append(extra)
    if missing:
        raise LabError(f"lab is incomplete ({', '.join(missing)}); run: python scripts/setup_lab.py")


def setup(update_game: bool = False, source_dir: Path = paths.STEAM_GAME_DIR) -> dict:
    report = sync_game(source_dir, update=update_game)
    write_lab_files()
    mod_zip = build_mod()
    # Only one mod belongs in the lab; earlier experiments (probe) must not load.
    for other in (paths.GAME_DIR / "mods").glob("*.zip"):
        if other != mod_zip:
            other.unlink()
    check_lab()
    return {"game_files": report, "mod_zip": str(mod_zip)}
