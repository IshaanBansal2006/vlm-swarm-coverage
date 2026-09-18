#!/usr/bin/env bash
# Run the demo config headless and render it: one command from a clean checkout to a video.
#   bash scripts/demo.sh [out_dir]      (default: runs/)
# Needs the [dev] or [analysis] extra for matplotlib, and ffmpeg for the mp4.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HERE/runs}"
RUN="$(python -m vlm_swarm_coverage "$HERE/configs/demo.toml" --out "$OUT" 2>/dev/null | tail -1)"
python -m vlm_swarm_coverage.viz "$RUN" --video "$RUN/belief.mp4" --summary "$RUN/summary.png"
echo "run:     $RUN"
echo "video:   $RUN/belief.mp4"
echo "summary: $RUN/summary.png"
