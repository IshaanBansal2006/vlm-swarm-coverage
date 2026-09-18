from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.field import Grid, ImportanceField, rasterize_ground_truth
from vlm_swarm_coverage.scene import Feature, Mission, Road, Scene, default_scene


def test_grid_shape_and_centres() -> None:
    g = Grid(60.0, 40.0, 2.0)
    assert g.shape == (20, 30)
    c = g.cell_centers()
    assert c.shape == (20, 30, 2)
    np.testing.assert_allclose(c[0, 0], [1.0, 1.0])
    np.testing.assert_allclose(c[19, 29], [59.0, 39.0])


def test_cell_of_far_edge_and_outside() -> None:
    g = Grid(60.0, 40.0, 2.0)
    assert g.cell_of(0.0, 0.0) == (0, 0)
    assert g.cell_of(60.0, 40.0) == (19, 29)
    assert g.cell_of(3.9, 2.0) == (1, 1)
    with pytest.raises(ValueError, match="outside"):
        g.cell_of(-1.0, 0.0)


def test_field_shape_is_checked_and_normalise_needs_mass() -> None:
    g = Grid(4.0, 4.0, 2.0)
    with pytest.raises(ValueError, match="does not match"):
        ImportanceField(g, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="zero total mass"):
        ImportanceField.uniform(g, 0.0).normalized()
    assert ImportanceField.uniform(g, 0.5).normalized().sum() == pytest.approx(1.0)


def test_ground_truth_paints_targets_not_distractors() -> None:
    scene = default_scene()
    g = Grid(scene.width, scene.height, 2.0)
    gt = rasterize_ground_truth(scene, g)
    floor = scene.mission.floor
    for f in scene.targets():
        r, c = g.cell_of(*f.position[:2])
        assert gt.values[r, c] > floor
    for f in scene.distractors():
        r, c = g.cell_of(*f.position[:2])
        assert gt.values[r, c] == floor
    assert gt.values.min() == floor
    assert gt.values.max() <= floor + max(scene.mission.weights.values())


def test_ground_truth_coverage_fraction_is_partial_at_edges() -> None:
    road = Road(5.0, 6.0)
    m = Mission("x", {"damaged_road": 1.0}, floor=0.0)
    scene = Scene(10.0, 10.0, road, m, features=(Feature("d", "damaged_road", (4.0, 4.0, 0.0)),))
    gt = rasterize_ground_truth(scene, Grid(10.0, 10.0, 2.0), subsamples=4)
    # 6 x 6 m square centred at (4, 4) spans [1, 7]: cells 1-2 full, cells 0 and 3 half, cell 4 empty
    assert gt.values[1, 1] == pytest.approx(1.0)
    assert gt.values[0, 1] == pytest.approx(0.5)
    assert gt.values[0, 0] == pytest.approx(0.25)
    assert gt.values[4, 4] == 0.0
    assert gt.total_mass == pytest.approx(36.0 / 4.0)


def test_overlapping_features_take_the_strongest() -> None:
    road = Road(5.0, 6.0)
    m = Mission("x", {"debris": 0.8, "stalled_vehicle": 0.9}, floor=0.1)
    scene = Scene(
        10.0, 10.0, road, m,
        features=(Feature("a", "debris", (5.0, 5.0, 0.0)), Feature("b", "stalled_vehicle", (5.0, 5.0, 0.0))),
    )
    gt = rasterize_ground_truth(scene, Grid(10.0, 10.0, 1.0))
    assert gt.values[5, 5] == pytest.approx(0.1 + 0.9)
