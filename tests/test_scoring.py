from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vlm_swarm_coverage.field import Grid, ImportanceField, rasterize_ground_truth
from vlm_swarm_coverage.scene import default_scene
from vlm_swarm_coverage.scoring import (
    CachedScorer,
    Observation,
    OracleScorer,
    View,
    apply_observation,
    footprint_cells,
)

if TYPE_CHECKING:
    from pathlib import Path


def _view(x: float, y: float, z: float = 12.0, yaw: float = 0.0, t: float = 0.0) -> View:
    return View(drone_id=0, t=t, x=x, y=y, z=z, yaw=yaw, fov_deg=70.0, aspect=4 / 3)


def test_footprint_size_follows_altitude_and_fov() -> None:
    g = Grid(60.0, 40.0, 1.0)
    half_w = 12.0 * math.tan(math.radians(35.0))
    cells = footprint_cells(g, 30.0, 20.0, 12.0, 0.0, 70.0, 4 / 3)
    assert len(cells) == pytest.approx(4 * half_w * half_w / (4 / 3), rel=0.1)
    assert len(footprint_cells(g, 30.0, 20.0, 6.0, 0.0, 70.0, 4 / 3)) < len(cells)
    assert len(footprint_cells(g, 30.0, 20.0, 12.0, 0.0, 40.0, 4 / 3)) < len(cells)


def test_footprint_is_clipped_at_the_edge_and_rotates() -> None:
    g = Grid(60.0, 40.0, 1.0)
    corner = footprint_cells(g, 0.0, 0.0, 12.0, 0.0, 70.0, 4 / 3)
    centre = footprint_cells(g, 30.0, 20.0, 12.0, 0.0, 70.0, 4 / 3)
    assert 0 < len(corner) < len(centre)
    turned = footprint_cells(g, 30.0, 20.0, 12.0, math.pi / 2, 70.0, 4 / 3)
    assert np.ptp(turned[:, 0]) > np.ptp(centre[:, 0])


def test_observation_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="cells"):
        Observation(0, 0.0, np.zeros((3, 3), dtype=np.int64), np.zeros(3))


def test_oracle_reads_ground_truth_and_overwrite_applies_it() -> None:
    scene = default_scene()
    g = Grid(scene.width, scene.height, 2.0)
    gt = rasterize_ground_truth(scene, g)
    target = scene.targets()[0]
    obs = OracleScorer(gt).score(_view(*target.position[:2]))
    r, c = g.cell_of(*target.position[:2])
    idx = np.flatnonzero((obs.cells[:, 0] == r) & (obs.cells[:, 1] == c))
    assert len(idx) == 1 and obs.values[idx[0]] > scene.mission.floor
    belief = ImportanceField.uniform(g, 0.0)
    apply_observation(belief, obs)
    assert belief.values[r, c] == gt.values[r, c]
    assert belief.values.sum() == pytest.approx(obs.values.sum())


def test_cache_hits_by_pose_bucket_and_persists(tmp_path: Path) -> None:
    scene = default_scene()
    g = Grid(scene.width, scene.height, 2.0)
    gt = rasterize_ground_truth(scene, g)
    path = tmp_path / "cache.json"
    cached = CachedScorer(OracleScorer(gt), g, path=path, altitude_step=2.0)
    a = cached.score(_view(18.0, 20.0, t=0.0))
    b = cached.score(_view(18.5, 20.5, z=12.4, t=1.0, yaw=0.7))
    assert (cached.hits, cached.misses) == (1, 1)
    np.testing.assert_array_equal(a.cells, b.cells)
    assert b.t == 1.0
    cached.score(_view(18.0, 20.0, z=16.0))
    assert cached.misses == 2
    cached.save()
    reloaded = CachedScorer(OracleScorer(gt), g, path=path)
    reloaded.score(_view(18.0, 20.0))
    assert (reloaded.hits, reloaded.misses) == (1, 0)


def test_cache_save_without_path_is_actionable() -> None:
    g = Grid(4.0, 4.0, 2.0)
    with pytest.raises(ValueError, match="path="):
        CachedScorer(OracleScorer(ImportanceField.uniform(g, 0.1)), g).save()
