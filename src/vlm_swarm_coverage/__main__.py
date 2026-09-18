"""Command line: `python -m vlm_swarm_coverage configs/demo.toml --out runs`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.simulation import run_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vlm-swarm-coverage", description="Run one configured simulation.")
    parser.add_argument("config", type=Path, help="Run config (.toml or .json)")
    parser.add_argument("--out", type=Path, default=Path("runs"), help="Root for run directories")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_dir = run_from_config(RunConfig.load(args.config), args.out)
    print(run_dir.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
