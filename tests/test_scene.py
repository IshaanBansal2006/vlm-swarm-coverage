"""The scene description is the one thing every other component trusts, so it is pinned here."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.scene import (
    DISTRACTOR_LABELS,
    FEATURE_EXTENTS,
    TARGET_LABELS,
    Feature,
    Mission,
    Road,
    Scene,
    default_scene,
    random_scene,
)


def test_labels_partition_the_extent_table() -> None:
    assert set(FEATURE_EXTENTS) == TARGET_LABELS | DISTRACTOR_LABELS
    assert not TARGET_LABELS & DISTRACTOR_LABELS


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown feature label"):
        Feature("x", "pothole", (1.0, 1.0, 0.0))


def test_footprint_rotates_about_centre() -> None:
    f = Feature("v", "stalled_vehicle", (10.0, 5.0, 0.0), yaw=np.pi / 2)
    fp = f.footprint
    assert fp.shape == (4, 2)
    np.testing.assert_allclose(fp.mean(axis=0), [10.0, 5.0])
    # rotated 90 degrees: the long axis now runs along y
    assert np.ptp(fp[:, 1]) == pytest.approx(4.6)
    assert np.ptp(fp[:, 0]) == pytest.approx(1.8)


def test_mission_weights_default_to_zero_for_unlisted_labels() -> None:
    m = Mission("find debris", {"debris": 1.0})
    assert m.weight("debris") == 1.0
    assert m.weight("tree") == 0.0


def test_mission_rejects_unknown_label_and_negative_floor() -> None:
    with pytest.raises(ValueError, match="unknown labels"):
        Mission("x", {"lava": 1.0})
    with pytest.raises(ValueError, match="floor"):
        Mission("x", {}, floor=-0.1)


def test_scene_rejects_out_of_bounds_and_duplicate_ids() -> None:
    road = Road(20.0, 6.0)
    m = Mission("x", {})
    with pytest.raises(ValueError, match="outside"):
        Scene(60.0, 40.0, road, m, features=(Feature("a", "tree", (70.0, 1.0, 0.0)),))
    with pytest.raises(ValueError, match="duplicate"):
        Scene(
            60.0, 40.0, road, m,
            features=(Feature("a", "tree", (1.0, 1.0, 0.0)), Feature("a", "shed", (2.0, 2.0, 0.0))),
        )


def test_default_scene_has_targets_and_distractors_and_starts() -> None:
    s = default_scene(n_drones=3, altitude=12.0)
    assert {f.label for f in s.targets()} == TARGET_LABELS
    assert len(s.distractors()) == 4
    assert len(s.drone_starts) == 3
    assert all(z == 12.0 for _, _, z in s.drone_starts)
    assert s.mission.weight("damaged_road") > 0


def test_random_scene_is_reproducible_and_keeps_distractors_off_the_road() -> None:
    a, b = random_scene(7), random_scene(7)
    assert a == b
    assert random_scene(8) != a
    half = a.road.width / 2
    for d in a.distractors():
        assert abs(d.position[1] - a.road.y_center) > half
    for t in a.targets():
        assert abs(t.position[1] - a.road.y_center) <= half
