"""Pack a Mod Loader mod into the zip layout the loader expects.

Usage: python build_mod.py <src_root> <mod_id> <out_zip>
<src_root> contains mods-unpacked/<mod_id>/ (manifest.json, mod_main.gd, ...).
"""
import pathlib
import sys
import zipfile


def build(src_root, mod_id, out_zip):
    src = pathlib.Path(src_root) / "mods-unpacked" / mod_id
    if not (src / "manifest.json").is_file():
        raise SystemExit(f"no manifest.json in {src}")
    out = pathlib.Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                zf.write(path, pathlib.PurePosixPath("mods-unpacked", mod_id, *path.relative_to(src).parts))
                count += 1
    print(f"built {out} ({count} files)")


if __name__ == "__main__":
    build(*sys.argv[1:4])
