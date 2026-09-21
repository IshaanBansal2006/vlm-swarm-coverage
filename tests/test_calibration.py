from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vlm_swarm_coverage.calibration import RecordingScorer, load_records, pose_sweep, record_sweep
from vlm_swarm_coverage.field import Grid, ImportanceField, rasterize_ground_truth
from vlm_swarm_coverage.scene import default_scene
from vlm_swarm_coverage.scoring import CachedScorer, OracleScorer, View

if TYPE_CHECKING:
    from pathlib import Path


def _oracle() -> tuple[OracleScorer, Grid]:
    scene = default_scene()
    g = Grid(scene.width, scene.height, 2.0)
    return OracleScorer(rasterize_ground_truth(scene, g)), g


def test_pose_sweep_is_deterministic_and_complete() -> None:
    a = list(pose_sweep([1.0, 2.0], [3.0], [10.0, 20.0], [0.0, 1.0], 70.0, 4 / 3, "m"))
    b = list(pose_sweep([1.0, 2.0], [3.0], [10.0, 20.0], [0.0, 1.0], 70.0, 4 / 3, "m"))
    assert a == b and len(a) == 8
    assert [v.t for v in a] == list(range(8))


def test_record_and_load_round_trip(tmp_path: Path) -> None:
    scorer, _ = _oracle()
    views = list(pose_sweep([18.0, 33.0], [20.0], [12.0], [0.0, 0.5], 70.0, 4 / 3, "storm"))
    n = record_sweep(scorer, views, tmp_path / "scores.jsonl", model_id="oracle")
    recs = list(load_records(tmp_path / "scores.jsonl"))
    assert n == len(recs) == 4
    assert recs[0].model_id == "oracle" and recs[0].view.mission_text == "storm"
    direct = scorer.score(views[0])
    np.testing.assert_array_equal(recs[0].obs.cells, direct.cells)
    np.testing.assert_allclose(recs[0].obs.values, direct.values, atol=1e-6)


def test_recording_scorer_is_transparent(tmp_path: Path) -> None:
    scorer, _ = _oracle()
    v = View(0, 0.0, 18.0, 20.0, 12.0, 0.0, 70.0)
    with RecordingScorer(scorer, tmp_path / "s.jsonl", "oracle") as rec:
        out = rec.score(v)
    np.testing.assert_array_equal(out.cells, scorer.score(v).cells)
    assert rec.count == 1


def test_cache_refuses_another_models_file(tmp_path: Path) -> None:
    scorer, g = _oracle()
    path = tmp_path / "cache.json"
    a = CachedScorer(scorer, g, path=path, model_id="oracle")
    a.score(View(0, 0.0, 18.0, 20.0, 12.0, 0.0, 70.0))
    a.save()
    with pytest.raises(ValueError, match="model_id"):
        CachedScorer(scorer, g, path=path, model_id="siglip2")
    with pytest.raises(ValueError, match="altitude_step"):
        CachedScorer(scorer, g, path=path, model_id="oracle", altitude_step=4.0)
    with pytest.raises(ValueError, match="cell_size"):
        CachedScorer(scorer, Grid(g.width, g.height, 4.0), path=path, model_id="oracle")
    ok = CachedScorer(scorer, g, path=path, model_id="oracle")
    ok.score(View(0, 1.0, 18.0, 20.0, 12.0, 0.0, 70.0))
    assert (ok.hits, ok.misses) == (1, 0)


def test_pre_014_cache_is_rejected_with_advice(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    path.write_text('{"0,0,6": {"cells": [[0, 0]], "values": [0.1]}}')
    g = Grid(4.0, 4.0, 2.0)
    with pytest.raises(ValueError, match="re-score"):
        CachedScorer(OracleScorer(ImportanceField.uniform(g, 0.1)), g, path=path)


def test_frames_round_trip_through_manifest_and_score_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    from PIL import Image

    from vlm_swarm_coverage.calibration import (
        FrameRecord,
        frame_view,
        load_manifest,
        load_records,
        score_frames,
        views_to_json,
    )
    from vlm_swarm_coverage.field import Grid, ImportanceField
    from vlm_swarm_coverage.scoring import Observation, OracleScorer, View, footprint_cells

    views = [("repeat", View(3, 1.0, 10.0, 8.0, 12.0, 0.3, 70.0, 4 / 3, None, "m")), ("yaw", View(0, 2.0, 5.0, 5.0, 8.0, 1.0, 70.0, 4 / 3, None, "m"))]
    assert views_to_json(views, tmp_path / "views.json") == 2
    rows = json.loads((tmp_path / "views.json").read_text())
    assert rows[0] == {"tag": "repeat", "drone": 3, "t": 1.0, "x": 10.0, "y": 8.0, "z": 12.0, "yaw": 0.3}
    (tmp_path / "sweep").mkdir()
    lines = []
    for k, row in enumerate(rows):
        Image.fromarray(np.full((30, 40, 3), (k * 100, 20, 20), dtype=np.uint8)).save(tmp_path / "sweep" / f"{k}.png")
        lines.append(json.dumps({**row, "file": f"sweep/{k}.png", "fov_deg": 70.0, "aspect": 4 / 3, "width": 40, "height": 30}))
    (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n")
    manifest = load_manifest(tmp_path)
    assert len(manifest) == 2 and manifest[0] == FrameRecord("repeat", 3, 1.0, 10.0, 8.0, 12.0, 0.3, 70.0, 4 / 3, "sweep/0.png")
    v = frame_view(manifest[1], tmp_path, "mission")
    assert v.frame is not None and v.frame.shape == (30, 40, 3) and v.frame[0, 0, 0] == 100 and v.mission_text == "mission"

    class MeanRed:
        def score(self, view: View) -> Observation:
            cells = footprint_cells(g, view.x, view.y, view.z, view.yaw, view.fov_deg, view.aspect)
            return Observation(view.drone_id, view.t, cells, np.full(len(cells), float(view.frame[..., 0].mean()) / 255.0))

    g = Grid(20.0, 20.0, 2.0)
    n = score_frames(MeanRed(), tmp_path, tmp_path / "store.jsonl", "meanred", "mission")
    records = list(load_records(tmp_path / "store.jsonl"))
    assert n == 2 and [r.tag for r in records] == ["repeat", "yaw"] and records[1].obs.values[0] == pytest.approx(100 / 255, abs=1e-5)
    n_oracle = score_frames(OracleScorer(ImportanceField.uniform(g, 0.2)), tmp_path / "manifest.jsonl", tmp_path / "store.jsonl", "oracle", "m")
    assert n_oracle == 2 and len(list(load_records(tmp_path / "store.jsonl"))) == 4
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_manifest(tmp_path / "sweep")


def test_cache_views_and_store_to_cache_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from vlm_swarm_coverage.calibration import (
        cache_views,
        load_records,
        main,
        store_to_cache,
        views_to_json,
    )
    from vlm_swarm_coverage.field import Grid
    from vlm_swarm_coverage.scoring import CachedScorer, OracleScorer, View

    g = Grid(8.0, 4.0, 2.0)  # 2 x 4 cells
    scene = default_scene()
    gt = rasterize_ground_truth(scene, Grid(60.0, 40.0, 2.0))
    views = cache_views(g, [12.0, 16.0], 70.0, 4 / 3, "m")
    assert len(views) == 16 and views[0][0] == "cache" and views[8][1].z == 16.0
    assert views_to_json(views, tmp_path / "v.json") == 16
    small_gt = ImportanceField(g, gt.values[:2, :4])
    record_sweep(OracleScorer(small_gt), (v for _, v in views), tmp_path / "s.jsonl", "oracle", tag="cache")
    record_sweep(OracleScorer(small_gt), [View(0, 99.0, 1.0, 1.0, 12.0, 0.0, 70.0)], tmp_path / "s.jsonl", "oracle", tag="other")
    assert sum(r.tag == "cache" for r in load_records(tmp_path / "s.jsonl")) == 16
    n = store_to_cache(tmp_path / "s.jsonl", g, tmp_path / "cache.json", "oracle", altitude_step=2.0)
    assert n == 16
    cached = CachedScorer(OracleScorer(small_gt), g, path=tmp_path / "cache.json", model_id="oracle")
    obs = cached.score(View(0, 0.0, 1.0, 1.0, 12.0, 0.7, 70.0))
    assert cached.hits == 1 and cached.misses == 0 and len(obs.cells) > 0
    with pytest.raises(ValueError, match="cache views"):
        store_to_cache(tmp_path / "s.jsonl", g, tmp_path / "c2.json", "oracle", tag="nope")
    cfg = tmp_path / "c.toml"
    cfg.write_text('name = "x"\n[area]\nwidth = 8.0\nheight = 4.0\ncell_size = 2.0\n')
    assert main(["cache-views", str(cfg), "--out", str(tmp_path / "cv.json"), "--altitudes", "12", "16"]) == 0
    assert main(["cache", str(tmp_path / "s.jsonl"), str(cfg), "--out", str(tmp_path / "c3.json"), "--model-id", "oracle"]) == 0
    from vlm_swarm_coverage.models import ScoreCalibration

    ScoreCalibration(lo=0.0, hi=0.5, floor=0.1).save(tmp_path / "cal.json")
    store_to_cache(tmp_path / "s.jsonl", g, tmp_path / "c4.json", "oracle", calibration=tmp_path / "cal.json")
    calibrated = CachedScorer(OracleScorer(small_gt), g, path=tmp_path / "c4.json", model_id="oracle")
    raw = CachedScorer(OracleScorer(small_gt), g, path=tmp_path / "cache.json", model_id="oracle")
    v = View(0, 0.0, 1.0, 1.0, 12.0, 0.0, 70.0)
    np.testing.assert_allclose(calibrated.score(v).values, 0.1 + 0.9 * np.clip(raw.score(v).values / 0.5, 0, 1))
