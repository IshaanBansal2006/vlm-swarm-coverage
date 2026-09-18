"""Importance scoring: turning what a drone sees into values on the cells it can see.

A scorer is a function from a `View` (pose, optional RGB frame, mission text) to an `Observation`
(the cells in the camera footprint and a value for each). The interface is host-agnostic on
purpose: the oracle runs anywhere, the model scorer runs wherever the GPU is, and the rig does
not care which. Three scorers:

- `OracleScorer` reads the ground-truth field. What a perfect model would say. Runs without a
  frame, so the loop runs without a renderer.
- `VLMScorer` is the slot for the model. It needs the `[vlm]` extra and a GPU.
- `CachedScorer` wraps any scorer and memoises by pose bucket (decision 012), so a scene is
  scored once per position x altitude and every later run reads the cache.

The local belief update is `apply_observation`: overwrite (decision 011). The scorer is the only
place noise handling may live; the belief carries no memory of its own.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from vlm_swarm_coverage.field import Grid, ImportanceField

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class View:
    """One look: where the camera is and, when a renderer is attached, what it saw."""

    drone_id: int
    t: float
    x: float
    y: float
    z: float
    yaw: float
    fov_deg: float
    aspect: float = 4 / 3
    frame: NDArray[np.uint8] | None = None
    mission_text: str = ""


@dataclass(frozen=True)
class Observation:
    """Scores for the cells in view. `cells` is (k, 2) of (row, col); `values` is (k,)."""

    drone_id: int
    t: float
    cells: NDArray[np.int64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.cells.ndim != 2 or self.cells.shape[1] != 2 or self.values.shape != (len(self.cells),):
            raise ValueError(
                f"observation needs cells (k, 2) and values (k,), got {self.cells.shape} and "
                f"{self.values.shape}"
            )


def footprint_cells(grid: Grid, x: float, y: float, z: float, yaw: float, fov_deg: float, aspect: float) -> NDArray[np.int64]:
    """(k, 2) rows/cols whose centre lies in a nadir camera's ground footprint.

    A camera pointing straight down at height z with full horizontal field of view f sees a
    ground rectangle of half-width z·tan(f/2) along its own x axis and half-height that over the
    aspect ratio, rotated by the drone's yaw. Cells outside the grid are simply not in the list.
    """
    half_w = z * math.tan(math.radians(fov_deg) / 2)
    half_h = half_w / aspect
    centres = grid.cell_centers()
    dx, dy = centres[..., 0] - x, centres[..., 1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    inside = (np.abs(u) <= half_w) & (np.abs(v) <= half_h)
    return np.argwhere(inside).astype(np.int64)


class ImportanceScorer(Protocol):
    def score(self, view: View) -> Observation: ...


class OracleScorer:
    """Reads the answer key for the cells in view. The upper bound on any real scorer."""

    def __init__(self, ground_truth: ImportanceField) -> None:
        self.ground_truth = ground_truth

    def score(self, view: View) -> Observation:
        g = self.ground_truth.grid
        cells = footprint_cells(g, view.x, view.y, view.z, view.yaw, view.fov_deg, view.aspect)
        values = self.ground_truth.values[cells[:, 0], cells[:, 1]]
        return Observation(view.drone_id, view.t, cells, values)


class VLMScorer:
    """The model slot. Construction checks the `[vlm]` extra is installed so a missing stack fails
    at startup rather than at the first frame. Scoring is not implemented yet: it needs a chosen
    model, a prompt, and the GPU host, none of which exist in this repository at this tag."""

    def __init__(self, model: str, grid: Grid) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VLMScorer needs the [vlm] extra: uv pip install -e '.[vlm]' on the GPU host"
            ) from exc
        self.model = model
        self.grid = grid

    def score(self, view: View) -> Observation:
        raise NotImplementedError(
            f"VLMScorer({self.model!r}) has no scoring implementation yet; use scorer.kind='oracle' "
            f"or 'cached' until the model path lands"
        )


class CachedScorer:
    """Memoise a scorer by pose bucket: the cell the drone is over and its altitude rounded to
    `altitude_step`. Yaw is ignored, which assumes the wrapped scorer's output is view-invariant
    in yaw; state that assumption wherever cached results are reported."""

    def __init__(self, inner: ImportanceScorer, grid: Grid, path: Path | None = None, altitude_step: float = 2.0) -> None:
        self.inner = inner
        self.grid = grid
        self.path = path
        self.altitude_step = altitude_step
        self._store: dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]] = {}
        self.hits = 0
        self.misses = 0
        if path is not None and path.exists():
            self._load(path)

    def key(self, view: View) -> str:
        row, col = self.grid.cell_of(view.x, view.y)
        z_bucket = int(round(view.z / self.altitude_step))
        return f"{row},{col},{z_bucket}"

    def score(self, view: View) -> Observation:
        k = self.key(view)
        if k in self._store:
            self.hits += 1
            cells, values = self._store[k]
            return Observation(view.drone_id, view.t, cells, values)
        self.misses += 1
        obs = self.inner.score(view)
        self._store[k] = (obs.cells, obs.values)
        return obs

    def save(self) -> None:
        if self.path is None:
            raise ValueError("CachedScorer has no path; construct it with path=... to persist")
        payload = {k: {"cells": c.tolist(), "values": v.tolist()} for k, (c, v) in self._store.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload))
        log.info("importance cache saved: %d entries -> %s", len(payload), self.path)

    def _load(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        for k, entry in raw.items():
            self._store[k] = (
                np.asarray(entry["cells"], dtype=np.int64).reshape(-1, 2),
                np.asarray(entry["values"], dtype=np.float64),
            )
        log.info("importance cache loaded: %d entries <- %s", len(self._store), path)


def apply_observation(field: ImportanceField, obs: Observation) -> None:
    """Overwrite the observed cells with the new values (decision 011). No memory here."""
    field.values[obs.cells[:, 0], obs.cells[:, 1]] = obs.values
