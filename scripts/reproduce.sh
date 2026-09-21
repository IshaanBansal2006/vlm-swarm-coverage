#!/usr/bin/env bash
# Reproduce the study from config + seed, stage by stage. Every stage is idempotent: it skips
# what already exists, so a partial run resumes.
#
#   bash scripts/reproduce.sh demo                 # the three-drone headless demo + video
#   bash scripts/reproduce.sh render               # Isaac (Windows, via interop): calibration + cache frames
#   bash scripts/reproduce.sh score  [MODEL...]    # score rendered frames into stores (GPU)
#   bash scripts/reproduce.sh cache  [MODEL]       # store -> pose-bucket cache for the sweeps
#   bash scripts/reproduce.sh calibrate [MODEL]    # gate report + error-model fit (private layer)
#   bash scripts/reproduce.sh sweeps SPEC...       # run study specs through the harness (private layer)
#   bash scripts/reproduce.sh analyse STAGES       # fits and figures from a study's stage rows (private layer)
#   bash scripts/reproduce.sh all                  # everything above in order (Isaac steps need Windows)
#
# Environment: VSC_OUT (default: ./runs/study, ignored by git), VSC_WIN_TMP (Windows temp dir seen from WSL),
# VSC_TEXTURES_WIN (Windows-side texture dir), VSC_MODELS (default: clipseg).
# The private layer (`vlm_swarm_coverage.held`) holds the study's protocol, faults, metrics and
# analysis; stages that need it say so and exit 2 when it is absent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${VSC_OUT:-$HERE/runs/study}"
WIN_TMP="${VSC_WIN_TMP:-/mnt/c/Users/ishaa/AppData/Local/Temp/vsc}"
WIN_TMP_W="$(wslpath -w "$WIN_TMP" 2>/dev/null || echo "$WIN_TMP")"
TEX_W="${VSC_TEXTURES_WIN:-C:\\Users\\ishaa\\AppData\\Local\\vsc\\textures}"
MODELS="${VSC_MODELS:-clipseg}"
STUDY="$HERE/configs/study.toml"
PY="${VSC_PYTHON:-python}"
UNC="$(wslpath -w "$HERE")"
mkdir -p "$OUT"

need_held() {
  if ! $PY -c "import vlm_swarm_coverage.held" 2>/dev/null; then
    echo "stage '$1' needs the private layer (vlm_swarm_coverage.held), which this checkout does not carry" >&2
    exit 2
  fi
}

stage_demo() {
  bash "$HERE/scripts/demo.sh" "$OUT/demo"
}

stage_render() {
  mkdir -p "$WIN_TMP"
  need_held render
  $PY - <<PYEOF
from pathlib import Path
from vlm_swarm_coverage.calibration import views_to_json, cache_views
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.field import Grid
from vlm_swarm_coverage.held.calibration_analysis import protocol
from vlm_swarm_coverage.simulation import build_scene
cfg = RunConfig.load("$STUDY"); scene = build_scene(cfg)
grid = Grid(cfg.area.width, cfg.area.height, cfg.area.cell_size)
views_to_json(protocol(scene, grid, cfg.swarm.camera_fov_deg, cfg.swarm.camera_aspect, z_ref=cfg.swarm.altitude), Path("$WIN_TMP/views-calibration.json"))
views_to_json(cache_views(grid, [cfg.swarm.altitude], cfg.swarm.camera_fov_deg, cfg.swarm.camera_aspect, scene.mission.text), Path("$WIN_TMP/views-cache.json"))
PYEOF
  for job in calibration cache; do
    if [ -f "$WIN_TMP/$job/manifest.jsonl" ]; then echo "have   frames: $job"; continue; fi
    echo "render $job (Isaac headless on Windows)"
    (cd /mnt/c && cmd.exe /c "cd /d C:\\ && C:\\IsaacSim\\run_scene.bat $UNC\\sim\\scenes\\coverage_scene.py --repo $UNC --record $WIN_TMP_W\\$job --sweep $WIN_TMP_W\\views-$job.json --textures $TEX_W --headless > $WIN_TMP_W\\$job-isaac.log 2>&1")
    echo "        $(ls "$WIN_TMP/$job/sweep" | wc -l) frames"
  done
}

stage_score() {
  local models="${*:-$MODELS}"
  mkdir -p "$OUT/stores"
  for m in oracle $models; do
    for job in calibration cache; do
      local store="$OUT/stores/$job-$m.jsonl"
      if [ -s "$store" ]; then echo "have   $store"; continue; fi
      $PY -m vlm_swarm_coverage.calibration score "$WIN_TMP/$job" "$STUDY" --model "$m" --out "$store"
    done
  done
}

stage_cache() {
  local m="${1:-${MODELS%% *}}"
  mkdir -p "$OUT/cache"
  $PY -m vlm_swarm_coverage.calibration cache "$OUT/stores/cache-$m.jsonl" "$STUDY" --out "$OUT/cache/$m.json" --model-id "$m:mission"
}

stage_calibrate() {
  need_held calibrate
  local m="${1:-${MODELS%% *}}"
  $PY -m vlm_swarm_coverage.held.calibration_analysis "$OUT/stores/calibration-$m.jsonl" "$STUDY" --out "$OUT/calibration/$m" || true
}

stage_sweeps() {
  need_held sweeps
  for spec in "$@"; do
    $PY -m vlm_swarm_coverage.held.sweeps "$spec" --out "$OUT/sweeps"
  done
}

stage_analyse() {
  need_held analyse
  $PY -m vlm_swarm_coverage.held.analysis "$1" --out "$OUT/analysis/$(basename "$1" -stages.json)"
}

case "${1:-}" in
  demo) stage_demo ;;
  render) stage_render ;;
  score) shift; stage_score "$@" ;;
  cache) shift; stage_cache "$@" ;;
  calibrate) shift; stage_calibrate "$@" ;;
  sweeps) shift; stage_sweeps "$@" ;;
  analyse) shift; stage_analyse "$@" ;;
  all) stage_demo; stage_render; stage_score; stage_cache; stage_calibrate; echo "sweeps and analysis: pass the study specs explicitly (bash scripts/reproduce.sh sweeps SPEC...; analyse STAGES)" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
