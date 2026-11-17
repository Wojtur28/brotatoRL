"""Put 2-4 recordings side by side in one video (2 = one row, 3-4 = a 2x2 grid), for comparing iterations.

    python scripts/record_video.py --run untrained --at final --stochastic --seed 30000 --out showcase/videos/a.mp4
    python scripts/record_video.py --run ppo1 --at 2.5M --seed 30000 --out showcase/videos/b.mp4
    ...
    python scripts/grid_video.py showcase/videos/a.mp4 showcase/videos/b.mp4 showcase/videos/c.mp4 showcase/videos/d.mp4
        --out showcase/videos/grid.mp4

Each tile is scaled to --tile (default 960x540). A recording that ends earlier (the run died sooner)
holds its last frame until the longest one ends, so the tiles stay comparable to the end. The
captions burnt in by record_video.py stay; --labels adds another one per tile. Audio is dropped.
The recordings do not have to be made at the same time: same seed = the same run start.
"""
import argparse
import subprocess
from pathlib import Path

from brotato_rl.video import _escape_text, ffmpeg_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", help="2-4 .mp4 files, in reading order (left to right, top to bottom)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", default="960x540", help="size of one tile")
    parser.add_argument("--labels", nargs="*", help="one caption per tile (optional)")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if not 2 <= len(args.videos) <= 4:
        raise SystemExit("give 2 to 4 videos")
    if args.labels and len(args.labels) != len(args.videos):
        raise SystemExit("--labels needs one caption per video")
    w, h = (int(v) for v in args.tile.lower().split("x"))
    filters = []
    for i in range(len(args.videos)):
        chain = (f"[{i}:v]fps={args.fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
                 f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,tpad=stop_mode=clone:stop_duration=36000")
        if args.labels:
            font = "C\\:/Windows/Fonts/segoeui.ttf"  # top left: record_video.py puts its caption bottom left
            chain += (f",drawtext=fontfile='{font}':text='{_escape_text(args.labels[i])}':x=12:y=12:"
                      "fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8")
        filters.append(chain + f"[v{i}]")
    n = len(args.videos)
    if n == 2:
        layout, size = "0_0|w0_0", None
    else:
        if n == 3:  # a black fourth tile keeps the grid square
            filters.append(f"color=c=black:s={w}x{h}:r={args.fps}[v3]")
        layout = "0_0|w0_0|0_h0|w0_h0"
    inputs = "".join(f"[v{i}]" for i in range(max(n, 4) if n > 2 else n))
    filters.append(f"{inputs}xstack=inputs={max(n, 4) if n > 2 else n}:layout={layout}:shortest=0[grid]")
    # tpad holds the last frame "forever"; stop at the longest input instead.
    durations = [_duration(v) for v in args.videos]
    cmd = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y"]
    for v in args.videos:
        cmd += ["-i", v]
    cmd += ["-filter_complex", ";".join(filters), "-map", "[grid]", "-t", f"{max(durations):.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", args.out]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"saved {args.out} ({max(durations):.0f} s, {n} tiles)")


def _duration(path) -> float:
    probe = Path(ffmpeg_path()).with_name("ffprobe.exe")
    out = subprocess.run([str(probe) if probe.exists() else "ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


if __name__ == "__main__":
    main()
