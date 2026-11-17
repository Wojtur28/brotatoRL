"""Recording the game window to an MP4 with ffmpeg (gdigrab), for showing a model play.

The window is found by the game's process id and its client area (without the title bar) is
captured from the screen, so it must stay visible: start the game with "--always-on-top"
(BrotatoEnv(game_args=["--always-on-top"])) and do not minimise it. Audio is not recorded.
"""
import ctypes
import shutil
import subprocess
import time
from ctypes import wintypes
from pathlib import Path


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FileNotFoundError("ffmpeg not found on PATH (winget install Gyan.FFmpeg)")
    return exe


def _make_dpi_aware():
    # Window coordinates must be physical pixels, the same ones gdigrab captures.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def window_rect(pid: int):
    """(x, y, width, height) of the client area of the visible top-level window of `pid`."""
    _make_dpi_aware()
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    best = None
    for hwnd in found:  # the process also owns small helper windows; take the largest client area
        rect, origin = wintypes.RECT(), wintypes.POINT(0, 0)
        user32.GetClientRect(hwnd, ctypes.byref(rect))
        user32.ClientToScreen(hwnd, ctypes.byref(origin))
        w, h = rect.right - rect.left, rect.bottom - rect.top
        if w > 0 and h > 0 and (best is None or w * h > best[2] * best[3]):
            best = (origin.x, origin.y, w, h)
    if best is None:
        raise RuntimeError(f"no visible window with a size for process {pid}")
    return best


def wait_for_window(pid: int, timeout=10.0):
    """window_rect, retried: right after start-up the game window may not have its size yet."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return window_rect(pid)
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.25)


def _escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


class WindowRecorder:
    """ffmpeg capturing a screen rectangle until stop()."""

    def __init__(self, rect, out, fps=30, label=None, crf=20):
        x, y, w, h = rect
        w, h = w - w % 2, h - h % 2  # yuv420p needs even sizes
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "gdigrab", "-framerate", str(fps),
               "-draw_mouse", "0", "-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}", "-i", "desktop"]
        if label:
            font = "C\\:/Windows/Fonts/segoeui.ttf"
            cmd += ["-vf", f"drawtext=fontfile='{font}':text='{_escape_text(label)}':x=16:y=h-th-16:fontsize={max(16, h // 30)}"
                           ":fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", str(self.out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def stop(self, timeout=30) -> Path:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"q")
                self.proc.stdin.flush()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        err = self.proc.stderr.read().decode("utf-8", "replace").strip()
        if self.proc.returncode not in (0, None) or not self.out.is_file():
            raise RuntimeError(f"ffmpeg failed ({self.proc.returncode}): {err[-500:]}")
        return self.out
