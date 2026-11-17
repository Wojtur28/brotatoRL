"""Launch the lab copy of the game once with the probe mod and print its report.

Usage: python run_probe.py <name> [game args...]
Each run gets a fresh instance folder (lab/instances/<name>) used as APPDATA.
"""
import ctypes
import ctypes.wintypes as wt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

LAB = pathlib.Path(__file__).resolve().parents[1] / "lab"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_mb(handle):
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(int(handle)), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**20 if ok else -1.0


def main():
    name, game_args = sys.argv[1], sys.argv[2:]
    inst = LAB / "instances" / name
    shutil.rmtree(inst, ignore_errors=True)
    inst.mkdir(parents=True)
    env = dict(os.environ, APPDATA=str(inst))
    cmd = [str(LAB / "game" / "Brotato.exe"), *game_args]
    print("cmd:", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=LAB / "game", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    peak = -1.0
    try:
        while proc.poll() is None and time.time() - t0 < 180:
            peak = max(peak, peak_working_set_mb(proc._handle))
            time.sleep(0.2)
    finally:
        if proc.poll() is None:
            proc.kill()
            print("TIMEOUT: killed")
    out = proc.stdout.read().decode("utf-8", "replace")
    print(f"exit={proc.returncode} wall={time.time() - t0:.1f}s peak_ws={peak:.0f}MB")
    (inst / "stdout.txt").write_text(out, encoding="utf-8")
    for line in out.splitlines():
        if any(k in line for k in ("shim", "Steam", "steam", "ERROR", "SCRIPT ERROR", "BrotatoRL", "ModLoader:Loader", "Installing script")):
            print("  |", line[:300])
    reports = list(inst.rglob("rl_probe_report.json"))
    if reports:
        print(json.dumps(json.loads(reports[0].read_text(encoding="utf-8")), indent=1))
    else:
        print("NO REPORT; instance files:", [str(p.relative_to(inst)) for p in inst.rglob("*")][:30])


if __name__ == "__main__":
    main()
