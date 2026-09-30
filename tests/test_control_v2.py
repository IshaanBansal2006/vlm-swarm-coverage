"""The v2 controllers (uniform Lloyd, lawnmower, ergodic) and the strict cache mode."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.config import ControllerConfig, RunConfig, ScorerConfig
from vlm_swarm_coverage.control import (
    ErgodicController,
    LawnmowerController,
    LloydController,
    ergodic_basis,
    ergodic_coefficients,
    ergodic_eval,
    ergodic_metric,
)
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.scoring import CachedScorer, OracleScorer, View
from vlm_swarm_coverage.simulation import build, build_controller
from vlm_swarm_coverage.sweep import apply_overrides, config_hash

GRID = Grid(60.0, 40.0, 2.0)


def two_gaussians() -> ImportanceField:
    c = GRID.cell_centers()
    d = (np.exp(-((c[..., 0] - 15) ** 2 + (c[..., 1] - 10) ** 2) / (2 * 4.0**2))
         + np.exp(-((c[..., 0] - 45) ** 2 + (c[..., 1] - 30) ** 2) / (2 * 4.0**2)))
    return ImportanceField(GRID, 0.005 + d)


def fly(controller, belief: ImportanceField, starts: np.ndarray, steps: int, dt: float = 0.1, speed: float = 3.0) -> np.ndarray:  # type: ignore[no-untyped-def]
    pos = starts.astype(float).copy()
    out = np.zeros((steps, len(pos), 2))
    for s in range(steps):
        peers = {j: pos[j].copy() for j in range(len(pos))}
        cmds = []
        for i in range(len(pos)):
            u = controller.command(i, pos[i].copy(), {j: p for j, p in peers.items() if j != i}, belief, s * dt)
            n = np.linalg.norm(u)
            cmds.append(u if n <= speed else u / n * speed)
        pos = np.clip(pos + np.array(cmds) * dt, [0, 0], [GRID.width, GRID.height])
        out[s] = pos
    return out


def phi(traj: np.ndarray, target: ImportanceField) -> float:
    size = (GRID.width, GRID.height)
    ks, h, lam = ergodic_basis(size, 10)
    c = ergodic_eval(traj.reshape(-1, 2), ks, h, size).mean(axis=0)
    return ergodic_metric(c, ergodic_coefficients(target.values, GRID.cell_centers(), ks, h, size), lam)


def test_basis_is_orthonormal_on_the_rectangle() -> None:
    fine = Grid(60.0, 40.0, 0.5)
    ks, h, _ = ergodic_basis((60.0, 40.0), 4)
    f = ergodic_eval(fine.cell_centers().reshape(-1, 2), ks, h, (60.0, 40.0))
    gram = f.T @ f * fine.cell_area
    assert np.allclose(gram, np.eye(len(ks)), atol=1e-2)


def test_ergodic_metric_falls_over_time_on_a_known_density() -> None:
    belief = two_gaussians()
    starts = np.array([[2.0, 8.0 + 6 * i] for i in range(5)])
    traj = fly(ErgodicController((60.0, 40.0), 3.0, 0.1), belief, starts, 1200)
    series = [phi(traj[:s], belief) for s in (100, 300, 600, 1200)]
    assert all(b < a for a, b in zip(series, series[1:], strict=False)), series
    assert series[-1] < 0.2 * series[0]
    hold = np.repeat(starts[None], 1200, axis=0)
    assert series[-1] < 0.1 * phi(hold, belief)


def test_ergodic_spends_more_time_where_the_density_is() -> None:
    belief = two_gaussians()
    starts = np.array([[2.0, 8.0 + 6 * i] for i in range(5)])
    traj = fly(ErgodicController((60.0, 40.0), 3.0, 0.1), belief, starts, 1200)[300:].reshape(-1, 2)
    near = (np.hypot(traj[:, 0] - 15, traj[:, 1] - 10) < 8) | (np.hypot(traj[:, 0] - 45, traj[:, 1] - 30) < 8)
    area_share = 2 * np.pi * 8**2 / (60 * 40)
    assert near.mean() > 2 * area_share


def test_ergodic_controller_moves_at_full_speed() -> None:
    u = ErgodicController((60.0, 40.0), 3.0, 0.1).command(0, np.array([10.0, 10.0]), {}, two_gaussians())
    assert np.linalg.norm(u) == pytest.approx(3.0)


def test_ergodic_drone_in_a_corner_is_not_stuck() -> None:
    ctl = ErgodicController((60.0, 40.0), 3.0, 0.1)
    for corner in ([0.0, 0.0], [60.0, 40.0], [0.0, 40.0]):
        assert np.linalg.norm(ctl.command(0, np.array(corner), {}, two_gaussians())) == pytest.approx(3.0)
    starts = np.array([[0.0, 0.0], [0.0, 40.0], [60.0, 0.0], [60.0, 40.0], [0.0, 20.0]])
    traj = fly(ErgodicController((60.0, 40.0), 3.0, 0.1), two_gaussians(), starts, 600)
    assert np.all(np.linalg.norm(traj[-1] - starts, axis=1) > 1.0)


def test_uniform_lloyd_ignores_the_belief() -> None:
    pos = np.array([30.0, 20.0])
    peers = {1: np.array([10.0, 10.0])}
    a = LloydController(uniform=True).command(0, pos, peers, two_gaussians())
    b = LloydController(uniform=True).command(0, pos, peers, ImportanceField.uniform(GRID, 1.0))
    c = LloydController().command(0, pos, peers, two_gaussians())
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_lawnmower_sweeps_its_own_strip_and_reads_nothing() -> None:
    ctl = LawnmowerController(60.0, 40.0, 5, 12.0, 70.0, 4 / 3, 3.0, 0.1)
    starts = np.array([[2.0, 8.0 + 6 * i] for i in range(5)])
    traj = fly(ctl, ImportanceField.uniform(GRID, 1.0), starts, 1200)
    ctl2 = LawnmowerController(60.0, 40.0, 5, 12.0, 70.0, 4 / 3, 3.0, 0.1)
    assert np.allclose(traj, fly(ctl2, two_gaussians(), starts, 1200))
    half = 12.0 * np.tan(np.radians(35.0))
    for i in range(5):
        late = traj[200:, i]
        assert np.allclose(late[:, 1], 4.0 + 8.0 * i, atol=1e-6)      # one pass on the strip centre
        assert late[:, 0].min() == pytest.approx(half, abs=0.05)
        assert late[:, 0].max() == pytest.approx(60 - half, abs=0.05)
    speeds = np.linalg.norm(np.diff(traj[200:, 0], axis=0), axis=1) / 0.1
    assert speeds.max() <= 3.0 + 1e-9 and np.median(speeds) == pytest.approx(3.0)


def test_lawnmower_wide_strip_gets_passes_one_footprint_apart() -> None:
    ctl = LawnmowerController(60.0, 40.0, 1, 12.0, 70.0, 4 / 3, 3.0, 0.1)
    ys = sorted(set(np.round(ctl.routes[0][:, 1], 6)))
    assert len(ys) == 4
    assert np.allclose(np.diff(ys), ctl.spacing)


def test_every_controller_kind_builds_and_an_unknown_kind_is_rejected() -> None:
    for kind in ("lloyd", "uniform", "lawnmower", "ergodic", "hold"):
        cfg = RunConfig(controller=ControllerConfig(kind=kind))
        assert build_controller(cfg) is not None
    with pytest.raises(ValidationError):
        ControllerConfig(kind="spiral")  # type: ignore[arg-type]


def test_new_fields_leave_old_config_hashes_unchanged() -> None:
    cfg = RunConfig()
    dumped = cfg.model_dump(mode="json")
    assert "strict_cache" not in dumped["scorer"] and "ergodic_modes" not in dumped["controller"]
    strict = apply_overrides(cfg, {"scorer.strict_cache": True})
    assert strict.model_dump(mode="json")["scorer"]["strict_cache"] is True
    assert config_hash(strict) != config_hash(cfg)
    assert RunConfig.model_validate_json(strict.model_dump_json()) == strict


def test_strict_cache_raises_on_a_miss_and_lenient_falls_back(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gt = two_gaussians()
    path = tmp_path / "c.json"
    full = CachedScorer(OracleScorer(gt), GRID, path=path)
    full.score(View(0, 0.0, 11.0, 11.0, 12.0, 0.0, 70.0))
    full.save()
    lenient = CachedScorer(OracleScorer(gt), GRID, path=path)
    lenient.score(View(0, 0.0, 31.0, 21.0, 12.0, 0.0, 70.0))
    assert lenient.misses == 1
    strict = CachedScorer(OracleScorer(gt), GRID, path=path, strict=True)
    strict.score(View(0, 0.0, 11.0, 11.0, 12.0, 0.0, 70.0))
    with pytest.raises(KeyError, match="strict cache"):
        strict.score(View(0, 0.0, 31.0, 21.0, 12.0, 0.0, 70.0))


def test_strict_cache_in_a_run_raises_and_needs_the_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"model_id": "oracle", "altitude_step": 2.0, "cell_size": 2.0, "grid": [20, 30], "entries": {}}))
    cfg = RunConfig(scorer=ScorerConfig(kind="cached", cache_path=path, strict_cache=True))
    sim = build(cfg)
    with pytest.raises(KeyError):
        sim._observe(0.0, _NullLog())
    missing = RunConfig(scorer=ScorerConfig(kind="cached", cache_path=tmp_path / "none.json", strict_cache=True))
    with pytest.raises(FileNotFoundError):
        build(missing)


class _NullLog:
    def write(self, *args: object, **kwargs: object) -> None:
        return None
