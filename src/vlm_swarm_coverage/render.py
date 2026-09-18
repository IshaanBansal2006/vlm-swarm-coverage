"""Command line for a rendered run: the loop on WSL driving Isaac Sim over the bridge.

    source scripts/ros-env.sh
    python3 -m vlm_swarm_coverage.render configs/demo.toml --out runs [--remote-scorer]

Start the Isaac scene first (sim/run_scene.bat coverage_scene.py). Without `--remote-scorer` the
loop scores locally with the configured scorer and Isaac only renders; with it, the loop takes
its observations from whatever scorer the scene is running.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vlm_swarm_coverage.artifacts import RunDir
from vlm_swarm_coverage.bridge import IsaacBridge, RemoteScorer
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.simulation import build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsc-render", description="Run the loop against Isaac Sim over ROS 2.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--remote-scorer", action="store_true", help="Take observations from the scene's scorer.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace to wall clock.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = RunConfig.load(args.config)
    bridge = IsaacBridge(realtime=not args.no_realtime)
    try:
        sim = build(cfg, scorer=RemoteScorer(bridge) if args.remote_scorer else None, on_step=bridge)
        run_dir = sim.run(RunDir.create(args.out, cfg))
    finally:
        bridge.close()
    print(run_dir.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
