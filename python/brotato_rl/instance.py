"""Launching and talking to one game process."""
import ctypes
import itertools
import os
import shutil
import socket
import subprocess
import time
import warnings
from ctypes import wintypes
from pathlib import Path

from . import lab, paths
from .protocol import BridgeError, Connection

PROTOCOL_VERSION = 1
_instance_counter = itertools.count()
_steam_warning_shown = False


class _KillOnCloseJob:
    """Windows job object: every game started here dies when this Python process exits."""

    def __init__(self):
        self.handle = None
        if os.name != "nt":
            return
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32 = kernel32

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        handle = kernel32.CreateJobObjectW(None, None)
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if handle and kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            self.handle = handle

    def add(self, proc: subprocess.Popen) -> None:
        if self.handle:
            self._kernel32.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(proc._handle)))


_job = None


def steam_is_running() -> bool:
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq steam.exe", "/NH"], capture_output=True, text=True).stdout
    return "steam.exe" in out.lower()


class GameInstance:
    """One Brotato process in the lab, driven over a localhost socket.

    Each instance gets its own user data folder (APPDATA is redirected for the child
    process only), its own port and its own window (or none).
    """

    def __init__(self, name=None, render=False, realtime=False, window=None, sound=False, lite_visuals=True,
                 resolution=(1280, 720), connect_timeout=120.0, reply_timeout=60.0, keep_files=False,
                 extra_args=(), position=None):
        self.name = name or f"{os.getpid()}-{next(_instance_counter)}"
        self.render = render
        self.realtime = realtime
        self.window = render if window is None else window
        self.sound = sound
        self.lite_visuals = lite_visuals
        self.resolution = resolution
        self.connect_timeout = connect_timeout
        self.reply_timeout = reply_timeout
        self.keep_files = keep_files
        self.extra_args = list(extra_args)
        self.position = position  # (x, y) of the window's top-left corner on screen, None = centred
        self.dir = paths.INSTANCES_DIR / self.name
        self.proc = None
        self.conn = None
        self.hello = None

    # -- lifecycle --------------------------------------------------------------------

    def start(self) -> dict:
        global _job, _steam_warning_shown
        lab.check_lab()
        if _job is None:
            _job = _KillOnCloseJob()
        if not _steam_warning_shown and steam_is_running():
            _steam_warning_shown = True
            warnings.warn("Steam is running: game instances will connect to the Steam client "
                          "(achievements are blocked by the mod). Close Steam for fully offline runs.")

        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        args = [str(paths.GAME_EXE), "--fixed-fps", "60"]
        if not self.sound:
            args += ["--audio-driver", "Dummy"]
        if self.window:
            args += ["--windowed", "--resolution", f"{self.resolution[0]}x{self.resolution[1]}"]
            if self.position is not None:
                args += ["--position", f"{int(self.position[0])},{int(self.position[1])}"]
        else:
            args += ["--no-window"]
        args += [
            f"--rl-port={port}",
            f"--rl-render={int(self.render)}",
            f"--rl-realtime={int(self.realtime)}",
            f"--rl-lite={int(self.lite_visuals)}",
            f"--rl-sound={int(self.sound)}",
            *self.extra_args,
        ]
        env = dict(os.environ, APPDATA=str(self.dir))
        # Console output holds what the log file may miss, e.g. native crash backtraces.
        with open(self.dir / "console.txt", "wb") as console:
            self.proc = subprocess.Popen(args, cwd=paths.GAME_DIR, env=env, stdout=console, stderr=subprocess.STDOUT)
        _job.add(self.proc)

        try:
            sock = self._accept(server)
        finally:
            server.close()
        self.conn = Connection(sock)
        self.hello = self.recv(timeout=self.connect_timeout)
        if self.hello.get("type") != "hello":
            raise BridgeError(f"expected hello, got {self.hello.get('type')}")
        if self.hello.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(f"mod speaks protocol {self.hello.get('protocol')}, expected {PROTOCOL_VERSION}; "
                              "rebuild the mod with scripts/setup_lab.py")
        user_dir = Path(self.hello.get("user_dir", ""))
        if user_dir.name != paths.LAB_USER_DIR_NAME or self.dir.resolve() not in user_dir.resolve().parents:
            self.close()
            raise BridgeError(f"instance user dir {user_dir} is outside the lab; refusing to continue")
        return self.hello

    def _accept(self, server: socket.socket) -> socket.socket:
        deadline = time.monotonic() + self.connect_timeout
        server.settimeout(0.5)
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BridgeError(f"game exited with code {self.proc.returncode} before connecting\n{self.log_tail()}")
            try:
                sock, _ = server.accept()
                return sock
            except socket.timeout:
                continue
        self.kill()
        raise BridgeError(f"game did not connect within {self.connect_timeout:.0f}s\n{self.log_tail()}")

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.send({"type": "quit"})
            except BridgeError:
                pass
            self.conn.close()
            self.conn = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()
            self.proc = None
        if not self.keep_files:
            # Windows keeps the log files locked for a moment after the game exits.
            for _ in range(20):
                shutil.rmtree(self.dir, ignore_errors=True)
                if not self.dir.exists():
                    break
                time.sleep(0.25)

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.conn is not None

    # -- messages ---------------------------------------------------------------------

    def send(self, msg: dict) -> None:
        if self.conn is None:
            raise BridgeError("instance is not connected")
        self.conn.send(msg)

    def recv(self, timeout=None) -> dict:
        if self.conn is None:
            raise BridgeError("instance is not connected")
        return self.conn.recv(self.reply_timeout if timeout is None else timeout)

    def log_tail(self, lines=40) -> str:
        parts = []
        for label, log in (("godot.log", self.dir / paths.LAB_USER_DIR_NAME / "logs" / "godot.log"),
                           ("console", self.dir / "console.txt")):
            try:
                text = log.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            parts.append(f"--- {label} ---\n" + "\n".join(text[-lines:]))
        return "\n".join(parts) or "(no game log)"
