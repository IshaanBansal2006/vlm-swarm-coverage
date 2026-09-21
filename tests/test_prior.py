from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.config import PriorConfig, RunConfig
from vlm_swarm_coverage.field import rasterize_ground_truth
from vlm_swarm_coverage.scene import random_scene
from vlm_swarm_coverage.simulation import build


def _sim(**prior: object):  # type: ignore[no-untyped-def]
    return build(RunConfig.model_validate({"name": "p", "prior": prior}))


def test_floor_prior_is_unobserved_floor() -> None:
    sim = _sim(kind="floor")
    b = sim.agents[0].belief
    assert (b.values == sim.scene.mission.floor).all() and np.isnan(b.stamps).all()


def test_truth_prior_matches_answer_key_and_is_copied_per_drone() -> None:
    sim = _sim(kind="truth")
    gt = rasterize_ground_truth(sim.scene, sim.grid).values
    np.testing.assert_array_equal(sim.agents[0].belief.values, gt)
    assert sim.agents[0].belief.values is not sim.agents[1].belief.values
    assert np.isnan(sim.agents[0].belief.stamps).all()


def test_shifted_truth_moves_the_map_and_fills_with_floor() -> None:
    sim = _sim(kind="shifted_truth", shift_m=(4.0, -2.0))
    gt = rasterize_ground_truth(sim.scene, sim.grid).values
    b = sim.agents[0].belief.values
    np.testing.assert_array_equal(b[:-1, 2:], gt[1:, :-2])
    assert (b[:, :2] == sim.scene.mission.floor).all() and (b[-1, :] == sim.scene.mission.floor).all()


def test_other_scene_prior_is_a_different_seeded_answer_key() -> None:
    sim = _sim(kind="other_scene", scene_seed=7)
    cfg = sim.config
    other = random_scene(7, cfg.scene.n_targets, cfg.scene.n_distractors, cfg.area.width, cfg.area.height,
                         cfg.swarm.n_drones, cfg.swarm.altitude)
    np.testing.assert_array_equal(sim.agents[0].belief.values, rasterize_ground_truth(other, sim.grid).values)
    assert not np.array_equal(sim.agents[0].belief.values, rasterize_ground_truth(sim.scene, sim.grid).values)


def test_observed_at_stamps_the_prior_so_it_is_shared() -> None:
    sim = _sim(kind="truth", observed_at=0.0)
    assert (sim.agents[0].belief.stamps == 0.0).all() and sim.agents[0].belief.observed.all()


def test_prior_config_validation() -> None:
    with pytest.raises(ValidationError, match="scene_seed"):
        PriorConfig(kind="other_scene")
    with pytest.raises(ValidationError, match="shift_m"):
        PriorConfig(kind="truth", shift_m=(1.0, 0.0))
    with pytest.raises(ValidationError, match="scene_seed"):
        PriorConfig(kind="floor", scene_seed=1)


def test_scene_floor_override_changes_the_answer_key_and_the_prior() -> None:
    sim = build(RunConfig.model_validate({"name": "f", "scene": {"floor": 0.005}}))
    gt = rasterize_ground_truth(sim.scene, sim.grid).values
    assert sim.scene.mission.floor == 0.005 and gt.min() == 0.005 and gt.max() > 0.9
    assert (sim.agents[0].belief.values == 0.005).all()
    default = build(RunConfig(name="d"))
    assert default.scene.mission.floor == 0.05
