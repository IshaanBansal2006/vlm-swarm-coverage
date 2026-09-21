"""Raw score store and pose-sweep recorder: the data side of a model-stability study (decision 014).

A stability study scores the same scene from many poses and asks how the importance map moves
with altitude, position and viewpoint. This module records what any scorer says at each pose,
raw, so the protocol reruns unchanged on a different scorer and two runs can be compared offline
without re-collecting. It does not analyse; analysis is a separate, later step.

File format: JSONL, one record per view, each with the pose, the scorer id and the observation.

The frame side: `views_to_json` writes the poses a renderer should visit; the renderer writes a
frame per pose and a manifest; `score_frames` walks the manifest, opens each frame, and scores it
with any scorer into the same store. So the frames a model sees are produced once, on the
renderer's host, and scored as many times as there are models, on whichever host has the GPU.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.scoring import Observation, View

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from vlm_swarm_coverage.scoring import ImportanceScorer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoreRecord:
    """One scored view. `tag` names the protocol phase the view belongs to (a repeat, an altitude
    ladder, a track), so an analysis groups records without re-deriving the protocol."""

    model_id: str
    view: View
    obs: Observation
    tag: str = ""

    def to_json(self) -> str:
        v, o = self.view, self.obs
        return json.dumps({
            "model_id": self.model_id, "tag": self.tag, "drone": v.drone_id, "t": v.t, "x": v.x, "y": v.y, "z": v.z, "yaw": v.yaw,
            "fov_deg": v.fov_deg, "aspect": v.aspect, "mission": v.mission_text,
            "cells": o.cells.tolist(), "values": [round(float(x), 6) for x in o.values],
        }, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> ScoreRecord:
        d = json.loads(line)
        view = View(d["drone"], d["t"], d["x"], d["y"], d["z"], d["yaw"], d["fov_deg"], d["aspect"], None, d["mission"])
        obs = Observation(d["drone"], d["t"], np.asarray(d["cells"], dtype=np.int64).reshape(-1, 2), np.asarray(d["values"], dtype=np.float64))
        return cls(d["model_id"], view, obs, d.get("tag", ""))


class RecordingScorer:
    """Wrap any scorer and append every observation it produces to a JSONL store.

    Frames are not stored, only poses and outputs: the pose plus the scene regenerate the frame.
    """

    def __init__(self, inner: ImportanceScorer, path: Path, model_id: str, tag: str = "") -> None:
        self.inner = inner
        self.model_id = model_id
        self.path = path
        self.tag = tag
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self.count = 0

    def score(self, view: View) -> Observation:
        obs = self.inner.score(view)
        self._fh.write(ScoreRecord(self.model_id, view, obs, self.tag).to_json() + "\n")
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


def record_sweep(scorer: ImportanceScorer, views: Iterable[View], path: Path, model_id: str, tag: str = "") -> int:
    with RecordingScorer(scorer, path, model_id, tag) as rec:
        for v in views:
            rec.score(v)
        return rec.count


def load_records(path: Path) -> Iterator[ScoreRecord]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield ScoreRecord.from_json(line)


# --- Frames on disk -----------------------------------------------------------------------------


def views_to_json(views: Iterable[tuple[str, View]], path: Path) -> int:
    """The renderer's input: a list of poses with their protocol tags and group ids."""
    rows = [{"tag": tag, "drone": v.drone_id, "t": v.t, "x": v.x, "y": v.y, "z": v.z, "yaw": v.yaw} for tag, v in views]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return len(rows)


@dataclass(frozen=True)
class FrameRecord:
    """One rendered frame and the pose it was rendered from, as the renderer's manifest lists it."""

    tag: str
    drone: int
    t: float
    x: float
    y: float
    z: float
    yaw: float
    fov_deg: float
    aspect: float
    file: str

    @classmethod
    def from_json(cls, line: str) -> FrameRecord:
        d = json.loads(line)
        return cls(str(d.get("tag", "")), int(d.get("drone", 0)), float(d.get("t", 0.0)), float(d["x"]), float(d["y"]),
                   float(d["z"]), float(d["yaw"]), float(d["fov_deg"]), float(d["aspect"]), str(d["file"]))


def load_manifest(path: Path) -> list[FrameRecord]:
    """`path` is the manifest file, or a directory holding `manifest.jsonl` or `nadir_manifest.jsonl`."""
    if path.is_dir():
        candidates = [path / "manifest.jsonl", path / "nadir_manifest.jsonl"]
        found = [c for c in candidates if c.exists()]
        if not found:
            raise FileNotFoundError(f"{path} holds neither manifest.jsonl nor nadir_manifest.jsonl")
        path = found[0]
    with path.open(encoding="utf-8") as fh:
        return [FrameRecord.from_json(line) for line in fh if line.strip()]


def frame_view(rec: FrameRecord, root: Path, mission_text: str) -> View:
    from PIL import Image

    with Image.open(root / rec.file) as im:
        frame = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return View(rec.drone, rec.t, rec.x, rec.y, rec.z, rec.yaw, rec.fov_deg, rec.aspect, frame, mission_text)


def score_frames(scorer: ImportanceScorer, frames: Path, store: Path, model_id: str, mission_text: str) -> int:
    """Score every frame in a manifest into the store, tagged as the manifest tags it."""
    root = frames if frames.is_dir() else frames.parent
    records = load_manifest(frames)
    with RecordingScorer(scorer, store, model_id) as rec:
        for r in records:
            rec.tag = r.tag
            rec.score(frame_view(r, root, mission_text))
        log.info("scored %d frames from %s with %s", rec.count, frames, model_id)
        return rec.count


def main(argv: list[str] | None = None) -> int:
    import argparse

    from vlm_swarm_coverage.config import RunConfig
    from vlm_swarm_coverage.field import Grid, rasterize_ground_truth
    from vlm_swarm_coverage.scoring import OracleScorer, VLMScorer
    from vlm_swarm_coverage.simulation import build_scene

    parser = argparse.ArgumentParser(prog="vsc-score-frames", description="Score rendered frames into a score store.")
    parser.add_argument("frames", type=Path, help="Manifest file or the directory holding it")
    parser.add_argument("config", type=Path, help="Run config defining the scene and grid the frames were rendered on")
    parser.add_argument("--model", required=True, help="'oracle' or a model id from models.MODEL_IDS")
    parser.add_argument("--out", type=Path, required=True, help="Score store (JSONL) to append to")
    parser.add_argument("--prompt-set", default="mission", choices=["mission", "null"])
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = RunConfig.load(args.config)
    scene = build_scene(cfg)
    grid = Grid(cfg.area.width, cfg.area.height, cfg.area.cell_size)
    if args.model == "oracle":
        scorer: ImportanceScorer = OracleScorer(rasterize_ground_truth(scene, grid))
        model_id = "oracle"
    else:
        scorer = VLMScorer(args.model, grid, scene.mission, args.prompt_set, args.calibration, args.device)
        model_id = f"{args.model}:{args.prompt_set}"
    n = score_frames(scorer, args.frames, args.out, model_id, scene.mission.text)
    print(f"{n} records -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
