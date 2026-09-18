"""Turn an Isaac capture (frames/*.jpg + meta.jsonl from coverage_scene.py --record) into an mp4.

    python scripts/build_footage.py <capture_dir> <out.mp4> [--fps 30]

Uses ffmpeg's concat demuxer with the real per-frame durations, then motion-interpolates to a
constant frame rate, so a 15 fps capture plays smoothly at 30.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    meta_path = args.capture / "meta.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found: pass the --record directory coverage_scene.py wrote")
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    frames = args.capture / "frames"
    lines = []
    for a, b in zip(meta, meta[1:] + [None], strict=False):
        dur = (b["t"] - a["t"]) if b else 1.0 / args.fps
        name = f"f_{a['frame']:05d}.jpg"
        path = (frames / name).as_posix()
        lines.append(f"file '{path}'\nduration {dur:.5f}")
    last_name = f"f_{meta[-1]['frame']:05d}.jpg"
    last = (frames / last_name).as_posix()
    lines.append(f"file '{last}'")
    concat = args.capture / "concat.txt"
    concat.write_text("\n".join(lines) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-vf", f"minterpolate=fps={args.fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,format=yuv420p",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-movflags", "+faststart", str(args.out),
    ], check=True)
    span = meta[-1]["t"] - meta[0]["t"]
    print(f"{args.out}: {len(meta)} frames over {span:.1f}s ({(len(meta) - 1) / max(span, 1e-9):.1f} fps captured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
