from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.world import KinematicWorld


def test_speed_is_saturated_and_yaw_follows_motion() -> None:
    w = KinematicWorld(((10.0, 10.0, 12.0),), 60.0, 40.0, max_speed=3.0, dt=0.5)
    w.step({0: np.array([30.0, 0.0])})
    d = w.drones[0]
    np.testing.assert_allclose(d.position, [11.5, 10.0, 12.0])
    assert d.yaw == 0.0
    w.step({0: np.array([0.0, -1.0])})
    assert d.yaw == pytest.approx(-np.pi / 2)
    w.step({})
    assert d.yaw == pytest.approx(-np.pi / 2)
    assert w.t == pytest.approx(1.5)


def test_position_is_clamped_to_area() -> None:
    w = KinematicWorld(((59.0, 1.0, 12.0),), 60.0, 40.0, max_speed=10.0, dt=1.0)
    w.step({0: np.array([5.0, -5.0])})
    np.testing.assert_allclose(w.drones[0].position[:2], [60.0, 0.0])


def test_needs_a_start() -> None:
    with pytest.raises(ValueError, match="start"):
        KinematicWorld((), 1.0, 1.0, 1.0, 1.0)
