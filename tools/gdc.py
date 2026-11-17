"""Godot 3.x GDScript bytecode (GDSC v13) -> approximate source text.

Good enough to read game logic; spacing is heuristic, comments are gone.
"""
import os
import struct
import sys

TOKENS = [
    "EMPTY", "IDENTIFIER", "CONSTANT", "self", "BUILT_IN_TYPE", "BUILT_IN_FUNC",
    "in", "==", "!=", "<", "<=", ">", ">=", "and", "or", "not",
    "+", "-", "*", "/", "%", "<<", ">>",
    "=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=",
    "&", "|", "^", "~",
    "if", "elif", "else", "for", "while", "break", "continue", "pass", "return", "match",
    "func", "class", "class_name", "extends", "is", "onready", "tool", "static", "export",
    "setget", "const", "var", "as", "void", "enum", "preload", "assert", "yield", "signal",
    "breakpoint", "remote", "sync", "master", "slave", "puppet", "remotesync", "mastersync",
    "puppetsync",
    "[", "]", "{", "}", "(", ")", ",", ";", ".", "?", ":", "$", "->", "NEWLINE",
    "PI", "TAU", "_", "INF", "NAN", "ERROR", "EOF", "CURSOR",
]

TYPES = [
    "null", "bool", "int", "float", "String", "Vector2", "Rect2", "Vector3", "Transform2D",
    "Plane", "Quat", "AABB", "Basis", "Transform", "Color", "NodePath", "RID", "Object",
    "Dictionary", "Array", "PoolByteArray", "PoolIntArray", "PoolRealArray",
    "PoolStringArray", "PoolVector2Array", "PoolVector3Array", "PoolColorArray",
]

FUNCS = [
    "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "atan2", "sqrt",
    "fmod", "fposmod", "posmod", "floor", "ceil", "round", "abs", "sign", "pow", "log", "exp",
    "is_nan", "is_inf", "is_equal_approx", "is_zero_approx", "ease", "decimals",
    "step_decimals", "stepify", "lerp", "lerp_angle", "inverse_lerp", "range_lerp",
    "smoothstep", "move_toward", "dectime", "randomize", "randi", "randf", "rand_range",
    "seed", "rand_seed", "deg2rad", "rad2deg", "linear2db", "db2linear", "polar2cartesian",
    "cartesian2polar", "wrapi", "wrapf", "max", "min", "clamp", "nearest_po2", "weakref",
    "funcref", "convert", "typeof", "type_exists", "char", "ord", "str", "print", "printt",
    "prints", "printerr", "printraw", "print_debug", "push_error", "push_warning", "var2str",
    "str2var", "var2bytes", "bytes2var", "range", "load", "inst2dict", "dict2inst",
    "validate_json", "parse_json", "to_json", "hash", "Color8", "ColorN", "print_stack",
    "get_stack", "instance_from_id", "len", "is_instance_valid", "deep_equal",
]

FLAG_64 = 1 << 16


class R:
    def __init__(self, b, p=0):
        self.b, self.p = b, p

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.p)[0]
        self.p += 4
        return v

    def i32(self):
        v = struct.unpack_from("<i", self.b, self.p)[0]
        self.p += 4
        return v

    def f(self, n, dbl=False):
        fmt = "<%d%s" % (n, "d" if dbl else "f")
        v = struct.unpack_from(fmt, self.b, self.p)
        self.p += n * (8 if dbl else 4)
        return v

    def string(self):
        n = self.u32()
        s = self.b[self.p:self.p + n].decode("utf-8", "replace")
        self.p += n + ((4 - n % 4) % 4)
        return s


def fnum(x):
    r = repr(float(x))
    return r


def variant(r):
    h = r.u32()
    t = h & 0xFF
    dbl = bool(h & FLAG_64)
    if t == 0:
        return "null"
    if t == 1:
        return "true" if r.u32() else "false"
    if t == 2:
        if dbl:
            v = struct.unpack_from("<q", r.b, r.p)[0]
            r.p += 8
            return str(v)
        return str(r.i32())
    if t == 3:
        return fnum(r.f(1, dbl)[0])
    if t == 4:
        s = r.string()
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'
    if t == 5:
        return "Vector2(%s)" % ", ".join(fnum(x) for x in r.f(2))
    if t == 6:
        return "Rect2(%s)" % ", ".join(fnum(x) for x in r.f(4))
    if t == 7:
        return "Vector3(%s)" % ", ".join(fnum(x) for x in r.f(3))
    if t == 8:
        return "Transform2D(%s)" % ", ".join(fnum(x) for x in r.f(6))
    if t == 9:
        return "Plane(%s)" % ", ".join(fnum(x) for x in r.f(4))
    if t == 10:
        return "Quat(%s)" % ", ".join(fnum(x) for x in r.f(4))
    if t == 11:
        return "AABB(%s)" % ", ".join(fnum(x) for x in r.f(6))
    if t == 12:
        return "Basis(%s)" % ", ".join(fnum(x) for x in r.f(9))
    if t == 13:
        return "Transform(%s)" % ", ".join(fnum(x) for x in r.f(12))
    if t == 14:
        return "Color(%s)" % ", ".join(fnum(x) for x in r.f(4))
    if t == 15:
        n = r.u32()
        if n & 0x80000000:
            names = n & 0x7FFFFFFF
            subs = r.u32()
            flags = r.u32()
            parts = [r.string() for _ in range(names + subs)]
            path = "/".join(parts[:names])
            if subs:
                path += ":" + ":".join(parts[names:])
            if flags & 1:
                path = "/" + path
            return 'NodePath("%s")' % path
        s = r.b[r.p:r.p + n].decode()
        r.p += n + ((4 - n % 4) % 4)
        return 'NodePath("%s")' % s
    if t == 18:
        n = r.u32() & 0x7FFFFFFF
        items = []
        for _ in range(n):
            k = variant(r)
            v = variant(r)
            items.append("%s: %s" % (k, v))
        return "{" + ", ".join(items) + "}"
    if t == 19:
        n = r.u32() & 0x7FFFFFFF
        return "[" + ", ".join(variant(r) for _ in range(n)) + "]"
    if t in (21, 22):
        n = r.u32()
        vals = struct.unpack_from("<%d%s" % (n, "i" if t == 21 else "f"), r.b, r.p)
        r.p += 4 * n
        return "%s([%s])" % (TYPES[t], ", ".join(str(v) for v in vals))
    if t == 23:
        n = r.u32()
        return "PoolStringArray([%s])" % ", ".join('"%s"' % r.string() for _ in range(n))
    if t == 24:
        n = r.u32()
        vals = r.f(2 * n)
        return "PoolVector2Array([%s])" % ", ".join(
            "Vector2(%s, %s)" % (fnum(vals[i]), fnum(vals[i + 1])) for i in range(0, 2 * n, 2))
    if t == 26:
        n = r.u32()
        vals = r.f(4 * n)
        return "PoolColorArray(%d colors)" % n
    raise ValueError("unsupported variant type %d at %d" % (t, r.p))


NO_SPACE_BEFORE = {")", "]", ",", ".", ":", "(", "["}
NO_SPACE_AFTER = {"(", "[", ".", "$", "~"}
WORDS = set(TOKENS[38:83]) | {"in", "and", "or", "not", "self", "PI", "TAU", "INF", "NAN"}


def decompile(data):
    if data[:4] != b"GDSC":
        raise ValueError("not GDSC")
    r = R(data, 4)
    ver = r.u32()
    nid, nconst, nlines, ntok = r.u32(), r.u32(), r.u32(), r.u32()
    ids = []
    for _ in range(nid):
        n = r.u32()
        raw = bytes(c ^ 0xB6 for c in data[r.p:r.p + n])
        r.p += n
        ids.append(raw.split(b"\0")[0].decode("utf-8", "replace"))
    consts = [variant(r) for _ in range(nconst)]
    r.p += nlines * 8
    toks = []
    for _ in range(ntok):
        b0 = data[r.p]
        if b0 & 0x80:
            toks.append(struct.unpack_from("<I", data, r.p)[0] & ~0x80)
            r.p += 4
        else:
            toks.append(b0)
            r.p += 1

    out = []
    line = []
    prev = None
    indent = 0

    def flush():
        if line:
            out.append("\t" * indent + "".join(line).rstrip())
        else:
            out.append("")

    for t in toks:
        kind = t & 0xFF
        val = t >> 8
        name = TOKENS[kind] if kind < len(TOKENS) else "?%d" % kind
        if name == "NEWLINE":
            flush()
            line = []
            indent = val
            prev = None
            continue
        if name == "EOF":
            break
        if kind == 1:
            s = ids[val]
        elif kind == 2:
            s = consts[val]
        elif kind == 4:
            s = TYPES[val] if val < len(TYPES) else "TYPE%d" % val
        elif kind == 5:
            s = FUNCS[val] if val < len(FUNCS) else "FUNC%d" % val
        else:
            s = name
        if line:
            if s in NO_SPACE_BEFORE and not (s in ("(", "[") and prev in WORDS and prev not in ("self",)):
                if s == ":" and prev not in (")", "]") and False:
                    line.append(" ")
            elif prev in NO_SPACE_AFTER:
                pass
            elif s in ("(", "[") and prev in WORDS and prev != "self":
                line.append(" ")
            elif prev == "-" and len(line) >= 2 and line[-2] in ("(", "[", ",", " = ", "return ") :
                pass
            else:
                line.append(" ")
        line.append(s)
        prev = s
    flush()
    return ver, "\n".join(out) + "\n"


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ok = bad = 0
    vers = set()
    for root, _, files in os.walk(src):
        for fn in files:
            if not fn.endswith(".gdc"):
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, src)
            outp = os.path.join(dst, rel[:-1])
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            try:
                ver, text = decompile(open(p, "rb").read())
                vers.add(ver)
                open(outp, "w", encoding="utf-8").write(text)
                ok += 1
            except Exception as e:  # noqa
                bad += 1
                print("FAIL", rel, e)
        for fn in files:
            if fn.endswith(".gd"):
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, src)
                outp = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(outp), exist_ok=True)
                open(outp, "wb").write(open(p, "rb").read())
    print("ok", ok, "fail", bad, "versions", vers)


if __name__ == "__main__":
    main()
