from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.control import (
    HoldController,
    LloydController,
    voronoi_mask,
    weighted_centroid,
)
from vlm_swarm_coverage.field import Grid, ImportanceField


def test_voronoi_partitions_without_overlap_or_gap() -> None:
    g = Grid(20.0, 10.0, 1.0)
    c = g.cell_centers()
    peers = {0: np.array([5.0, 5.0]), 1: np.array([15.0, 5.0]), 2: np.array([10.0, 5.0])}
    masks = [voronoi_mask(c, peers[i], {j: p for j, p in peers.items() if j != i}, i) for i in peers]
    total = sum(m.astype(int) for m in masks)
    assert (total == 1).all()


def test_centroid_of_uniform_density_is_geometric_centre() -> None:
    g = Grid(10.0, 10.0, 1.0)
    c = g.cell_centers()
    centroid, mass = weighted_centroid(c, np.ones(g.shape), np.ones(g.shape, dtype=bool))
    np.testing.assert_allclose(centroid, [5.0, 5.0])
    assert mass == 100.0
    nan_c, zero = weighted_centroid(c, np.zeros(g.shape), np.ones(g.shape, dtype=bool))
    assert zero == 0.0 and np.isnan(nan_c).all()


def test_lloyd_moves_toward_mass_and_stops_at_centroid() -> None:
    g = Grid(20.0, 20.0, 1.0)
    belief = ImportanceField.uniform(g, 0.01)
    belief.values[15:18, 15:18] = 1.0
    ctrl = LloydController(gain=1.0)
    pos = np.array([2.0, 2.0])
    v = ctrl.command(0, pos, {}, belief)
    assert v[0] > 0 and v[1] > 0
    for _ in range(200):
        pos = pos + 0.1 * ctrl.command(0, pos, {}, belief)
    assert np.linalg.norm(ctrl.command(0, pos, {}, belief)) < 1e-3
    assert 14.0 < pos[0] < 18.0 and 14.0 < pos[1] < 18.0


def test_lloyd_with_peer_shares_the_area() -> None:
    g = Grid(20.0, 10.0, 1.0)
    belief = ImportanceField.uniform(g, 1.0)
    ctrl = LloydController()
    v = ctrl.command(0, np.array([8.0, 5.0]), {1: np.array([12.0, 5.0])}, belief)
    assert v[0] < 0
    assert v[1] == pytest.approx(0.0)


def test_hold_and_gain_validation() -> None:
    g = Grid(4.0, 4.0, 2.0)
    assert (HoldController().command(0, np.zeros(2), {}, ImportanceField.uniform(g, 1.0)) == 0).all()
    with pytest.raises(ValueError, match="gain"):
        LloydController(gain=0.0)
