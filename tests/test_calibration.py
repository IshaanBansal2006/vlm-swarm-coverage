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
