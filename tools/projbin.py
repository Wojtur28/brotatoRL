import sys, struct
sys.path.insert(0, sys.argv[2])
from gdc import R, variant
d = open(sys.argv[1], "rb").read()
assert d[:4] == b"ECFG"
r = R(d, 4)
n = r.u32()
for _ in range(n):
    kl = r.u32(); k = d[r.p:r.p+kl].decode(); r.p += kl
    vl = r.u32(); start = r.p
    try:
        v = variant(R(d, r.p))
    except Exception as e:
        v = "<%s>" % e
    r.p = start + vl
    print(k, "=", v[:300])
