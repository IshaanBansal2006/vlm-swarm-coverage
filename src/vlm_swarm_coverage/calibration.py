"""Raw score store and pose-sweep recorder: the data side of a model-stability study (decision 014).

A stability study scores the same scene from many poses and asks how the importance map moves
with altitude, position and viewpoint. This module records what any scorer says at each pose,
raw, so the protocol reruns unchanged on a different scorer and two runs can be compared offline
without re-collecting. It does not analyse; analysis is a separate, later step.

File format: JSONL, one record per view, each with the pose, the scorer id and the observation.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.scoring import Observation, View

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from pathlib import Path

    from vlm_swarm_coverage.scoring import ImportanceScorer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreRecord:
    model_id: str
    view: View
    obs: Observation

    def to_json(self) -> str:
        v, o = self.view, self.obs
        return json.dumps({
            "model_id": self.model_id, "drone": v.drone_id, "t": v.t, "x": v.x, "y": v.y, "z": v.z, "yaw": v.yaw,
            "fov_deg": v.fov_deg, "aspect": v.aspect, "mission": v.mission_text,
            "cells": o.cells.tolist(), "values": [round(float(x), 6) for x in o.values],
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> ScoreRecord:
        d = json.loads(line)
        view = View(d["drone"], d["t"], d["x"], d["y"], d["z"], d["yaw"], d["fov_deg"], d["aspect"], None, d["mission"])
        obs = Observation(d["drone"], d["t"], np.asarray(d["cells"], dtype=np.int64).reshape(-1, 2), np.asarray(d["values"], dtype=np.float64))
        return cls(d["model_id"], view, obs)


class RecordingScorer:
    """Wrap any scorer and append every observation it produces to a JSONL store.

    Frames are not stored, only poses and outputs: the pose plus the scene regenerate the frame.
    """

    def __init__(self, inner: ImportanceScorer, path: Path, model_id: str) -> None:
        self.inner = inner
        self.model_id = model_id
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self.count = 0

    def score(self, view: View) -> Observation:
        obs = self.inner.score(view)
        self._fh.write(ScoreRecord(self.model_id, view, obs).to_json() + "\n")
        self.count += 1
        return obs

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            log.info("score store closed: %d records for %s -> %s", self.count, self.model_id, self.path)

    def __enter__(self) -> RecordingScorer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def pose_sweep(
    xs: Sequence[float], ys: Sequence[float], zs: Sequence[float], yaws: Sequence[float],
    fov_deg: float, aspect: float, mission_text: str,
) -> Iterator[View]:
    """The Cartesian product of positions, altitudes and yaws as views, in a fixed order, so two
    scorers see exactly the same protocol."""
    for i, (z, yaw, x, y) in enumerate(itertools.product(zs, yaws, xs, ys)):
        yield View(0, float(i), float(x), float(y), float(z), float(yaw), fov_deg, aspect, None, mission_text)


def record_sweep(scorer: ImportanceScorer, views: Iterable[View], path: Path, model_id: str) -> int:
    with RecordingScorer(scorer, path, model_id) as rec:
        for v in views:
            rec.score(v)
        return rec.count


def load_records(path: Path) -> Iterator[ScoreRecord]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield ScoreRecord.from_json(line)
