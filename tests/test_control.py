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


def test_replay_controller_retraces_a_logged_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from vlm_swarm_coverage.config import RunConfig
    from vlm_swarm_coverage.control import ReplayController
    from vlm_swarm_coverage.simulation import run_from_config

    base = {"name": "src", "sim": {"dt": 0.1, "duration_s": 15.0, "seed": 3}}
    source = run_from_config(RunConfig.model_validate(base), tmp_path / "a")
    twin_cfg = RunConfig.model_validate({**base, "name": "twin", "channel": {"drop_rate": 0.95},
                                         "fusion": {"kind": "none"},
                                         "controller": {"kind": "replay", "replay_run": str(source.path)}})
    twin = run_from_config(twin_cfg, tmp_path / "b")
    a = [(e["drone"], e["t"], e["x"], e["y"]) for e in source.iter_events("pose")]
    b = [(e["drone"], e["t"], e["x"], e["y"]) for e in twin.iter_events("pose")]
    assert len(a) == len(b) > 0
    np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-4)
    assert list(twin.iter_events("drop"))
    ctrl = ReplayController.from_run(source)
    assert ctrl.n_drones == 3 and ctrl.n_steps == 150
    with pytest.raises(KeyError, match="no replay trajectory"):
        ctrl.command(9, np.zeros(2), {}, ImportanceField.uniform(Grid(4.0, 4.0, 2.0), 1.0), 0.0)
    with pytest.raises(ValueError, match="Replay needs both to match"):
        RunConfig.model_validate({**base, "swarm": {"n_drones": 2},
                                  "controller": {"kind": "replay", "replay_run": str(source.path)}})
        from vlm_swarm_coverage.simulation import build
        build(RunConfig.model_validate({**base, "swarm": {"n_drones": 2},
                                        "controller": {"kind": "replay", "replay_run": str(source.path)}}))


def test_replay_holds_past_the_end_and_validates() -> None:
    from vlm_swarm_coverage.control import ReplayController

    ctrl = ReplayController({0: np.array([[1.0, 0.0], [2.0, 0.0]])}, dt=0.5)
    np.testing.assert_allclose(ctrl.command(0, np.zeros(2), {}, None, t=0.0), [2.0, 0.0])  # type: ignore[arg-type]
    np.testing.assert_allclose(ctrl.command(0, np.array([1.0, 0.0]), {}, None, t=0.5), [2.0, 0.0])  # type: ignore[arg-type]
    np.testing.assert_allclose(ctrl.command(0, np.array([2.0, 0.0]), {}, None, t=5.0), [0.0, 0.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt"):
        ReplayController({0: np.zeros((1, 2))}, dt=0.0)
    with pytest.raises(ValueError, match="at least one"):
        ReplayController({}, dt=0.1)
