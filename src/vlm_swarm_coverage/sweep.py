"""Sweeps: a design over config axes, expanded into runs, executed in parallel, indexed (062).

A sweep spec names a base config, a set of axes (dotted config keys with the values to try), how
to combine them (`full` factorial or `one_at_a_time`), the seeds, and optional explicit cells.
`expand` turns that into cells; `run_sweep` runs every cell that is not already in the archive
index and appends one line per run to `index.jsonl`, so a killed sweep resumes where it stopped
and a re-run with the same spec does nothing.

Two hooks keep this module ignorant of things the config does not describe. `hook` is a
`module:function` that receives the cell's config and its non-config parameters (axis keys that
start with `_`) and returns keyword arguments for `simulation.build`; `post_run` receives the
finished run directory and returns a dict merged into the index row. Both are imported by name in
the worker, so they must be importable there.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vlm_swarm_coverage.config import RunConfig, _toml_loader
from vlm_swarm_coverage.simulation import run_from_config

log = logging.getLogger(__name__)

INDEX_FILE = "index.jsonl"


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    base: Path
    axes: dict[str, list[Any]] = Field(default_factory=dict, description="Dotted config key (or _param) -> values; a dict value is a composite level whose keys nest under the axis key.")
    design: Literal["full", "one_at_a_time"] = "full"
    seeds: list[int] = Field(default_factory=lambda: [0])
    cells: list[dict[str, Any]] = Field(default_factory=list, description="Explicit extra cells, each a dict of overrides.")
    hook: str | None = Field(None, description="module:function(cfg, params) -> build kwargs.")
    post_run: str | None = Field(None, description="module:function(run_dir) -> dict merged into the index row.")
    workers: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1), ge=1)

    @classmethod
    def load(cls, path: Path | str) -> SweepSpec:
        path = Path(path)
        with path.open("rb") as fh:
            raw = _toml_loader().load(fh)
        spec = cls.model_validate(raw)
        if not spec.base.is_absolute():
            spec = spec.model_copy(update={"base": (path.parent / spec.base).resolve()})
        return spec


@dataclass(frozen=True)
class Cell:
    cell_id: str
    overrides: dict[str, Any]
    params: dict[str, Any]


def _flatten(assignments: dict[str, Any]) -> dict[str, Any]:
    """A level that is a dict is a composite: `"message": {"kind": "topk", "k": 5}` becomes
    `message.kind` and `message.k`, so one axis can move several keys of one factor together.
    Under the axis key `"*"` the level's keys are taken as they are, already dotted, for factors
    that span sections (a scene seed and the cache rendered for it). Hook parameters (`_name`)
    keep their dict values whole."""
    out: dict[str, Any] = {}
    for key, value in assignments.items():
        if isinstance(value, dict) and key == "*":
            out.update(value)  # a level that sets several dotted keys as they are (a layout, say)
        elif isinstance(value, dict) and not key.startswith("_"):
            for sub, sub_value in value.items():
                out[f"{key}.{sub}"] = sub_value
        else:
            out[key] = value
    return out


def _split(assignments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assignments = _flatten(assignments)
    overrides = {k: v for k, v in assignments.items() if not k.startswith("_")}
    params = {k[1:]: v for k, v in assignments.items() if k.startswith("_")}
    return overrides, params


def _cell_id(assignments: dict[str, Any]) -> str:
    text = json.dumps(assignments, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def expand(spec: SweepSpec) -> list[Cell]:
    """Every cell of the design, crossed with the seeds, plus the explicit cells. Deterministic order."""
    keys = list(spec.axes)
    if not keys:
        combos: list[dict[str, Any]] = []
    elif spec.design == "full":
        combos = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(spec.axes[k] for k in keys))]
    else:
        # one factor at a time, crossed with the `*` axis (a layout is not a factor to screen)
        factors = [k for k in keys if k != "*"]
        single = [{}] + [{k: v} for k in factors for v in spec.axes[k]]
        layouts = spec.axes.get("*", [None])
        combos = [({"*": lay} if lay is not None else {}) | c for lay in layouts for c in single]
    combos += [dict(c) for c in spec.cells]
    cells: list[Cell] = []
    seen: set[str] = set()
    for combo in combos:
        combo = _flatten(combo)
        for seed in ([None] if "sim.seed" in combo else spec.seeds):
            assignments = dict(combo)
            if seed is not None:
                assignments["sim.seed"] = seed
            cid = _cell_id(assignments)
            if cid in seen:
                continue
            seen.add(cid)
            overrides, params = _split(assignments)
            cells.append(Cell(cid, overrides, params))
    return cells


def apply_overrides(cfg: RunConfig, overrides: dict[str, Any], name: str | None = None) -> RunConfig:
    """A new config with dotted keys set. Unknown keys fail in validation, as they should."""
    data = cfg.model_dump(mode="json")
    for key, value in overrides.items():
        node = data
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    if name is not None:
        data["name"] = name
    return RunConfig.model_validate(data)


def config_hash(cfg: RunConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:12]


def _import(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"hook {spec!r} must be 'module:function'")
    return getattr(importlib.import_module(module), attr)


def run_cell(cfg_json: str, out_root: str, cell: Cell, hook: str | None, post_run: str | None) -> dict[str, Any]:
    """One run, in a worker process. Never raises: a failure is a row with status 'error'."""
    cfg = RunConfig.model_validate_json(cfg_json)
    row: dict[str, Any] = {"cell_id": cell.cell_id, "overrides": cell.overrides, "params": cell.params,
                           "config_hash": config_hash(cfg), "seed": cfg.sim.seed}
    t0 = time.perf_counter()
    try:
        kwargs = _import(hook)(cfg, cell.params) if hook else {}
        run_dir = run_from_config(cfg, Path(out_root), **kwargs)
        row.update(run_dir=str(run_dir.path), status="ok")
        if post_run:
            row.update(_import(post_run)(run_dir))
    except Exception as exc:
        log.exception("cell %s failed", cell.cell_id)
        row.update(status="error", error=repr(exc))
    row["wall_s"] = round(time.perf_counter() - t0, 3)
    return row


def read_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_sweep(spec: SweepSpec, out_root: Path | str, dry_run: bool = False) -> list[dict[str, Any]]:
    """Run every cell not already in the index; return all index rows (old and new)."""
    base = RunConfig.load(spec.base)
    cells = expand(spec)
    sweep_dir = Path(out_root) / spec.name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    index_path = sweep_dir / INDEX_FILE
    rows = read_index(index_path)
    done = {r["config_hash"] for r in rows if r.get("status") == "ok"}
    todo: list[tuple[str, Cell]] = []
    for cell in cells:
        cfg = apply_overrides(base, cell.overrides, name=f"{spec.name}-{cell.cell_id}")
        if config_hash(cfg) in done:
            continue
        todo.append((cfg.model_dump_json(), cell))
    log.info("sweep %s: %d cells, %d already done, %d to run, %d workers", spec.name, len(cells), len(cells) - len(todo), len(todo), spec.workers)
    if dry_run:
        for _, cell in todo:
            print(cell.cell_id, json.dumps(cell.overrides, sort_keys=True), json.dumps(cell.params, sort_keys=True))
        return rows
    (sweep_dir / "spec.json").write_text(spec.model_dump_json(indent=2) + "\n")
    failures = 0
    with index_path.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(
        max_workers=spec.workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        futures = [pool.submit(run_cell, cfg_json, str(sweep_dir), cell, spec.hook, spec.post_run) for cfg_json, cell in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            failures += row["status"] != "ok"
            rows.append(row)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(futures):
                log.info("sweep %s: %d/%d done, %d failed", spec.name, n, len(futures), failures)
    if failures:
        log.error("sweep %s: %d of %d runs failed; see status='error' rows in %s", spec.name, failures, len(todo), index_path)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsc-sweep", description="Expand and run a sweep spec.")
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("sweeps"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="List the cells that would run.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    spec = SweepSpec.load(args.spec)
    if args.workers:
        spec = spec.model_copy(update={"workers": args.workers})
    rows = run_sweep(spec, args.out, dry_run=args.dry_run)
    failed = sum(r.get("status") == "error" for r in rows)
    print(f"{len(rows)} rows in {Path(args.out) / spec.name / INDEX_FILE}; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
