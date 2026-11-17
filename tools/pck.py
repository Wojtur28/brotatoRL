"""Minimal Godot 3 PCK (format v1) lister/extractor. Read-only on the source pck."""
import os
import struct
import sys


def read_index(path):
    f = open(path, "rb")
    magic = f.read(4)
    if magic != b"GDPC":
        raise SystemExit(f"bad magic {magic!r}")
    fmt, major, minor, patch = struct.unpack("<4I", f.read(16))
    f.read(16 * 4)  # reserved
    (count,) = struct.unpack("<I", f.read(4))
    entries = []
    for _ in range(count):
        (plen,) = struct.unpack("<I", f.read(4))
        name = f.read(plen).rstrip(b"\0").decode("utf-8")
        off, size = struct.unpack("<QQ", f.read(16))
        f.read(16)  # md5
        entries.append((name, off, size))
    return f, (fmt, major, minor, patch), entries


def main():
    cmd, pck = sys.argv[1], sys.argv[2]
    f, ver, entries = read_index(pck)
    if cmd == "info":
        print("format/version:", ver, "files:", len(entries))
        exts = {}
        for n, _, s in entries:
            e = os.path.splitext(n)[1]
            exts[e] = exts.get(e, 0) + 1
        for e, c in sorted(exts.items(), key=lambda x: -x[1]):
            print(f"  {e or '<none>'}: {c}")
    elif cmd == "list":
        for n, o, s in entries:
            print(s, n)
    elif cmd == "extract":
        out = sys.argv[3]
        exts = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
        n_done = 0
        for n, o, s in entries:
            if exts and not n.endswith(exts):
                continue
            rel = n.replace("res://", "")
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f.seek(o)
            with open(dst, "wb") as w:
                w.write(f.read(s))
            n_done += 1
        print("extracted", n_done)


if __name__ == "__main__":
    main()
