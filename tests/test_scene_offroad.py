from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.field import Grid, rasterize_ground_truth
from vlm_swarm_coverage.scene import (
    OFFROAD_DISTRACTOR_CLEARANCE_M,
    OFFROAD_MIN_OFFSET_M,
    OFFROAD_TARGET_LABELS,
    Scene,
    random_scene,
)
from vlm_swarm_coverage.simulation import build_scene
from vlm_swarm_coverage.sweep import apply_overrides, config_hash

# Recorded before target_mode existed: every road seed must keep producing exactly these scenes.
ROAD_SEEDS_1_24_SHA256 = "dc1505f4738b1b4ee4793fd20cc0bb91739d58191fdf22bb7c3cd17c2e9127b2"
DEFAULT_CONFIG_HASH = "ec498de8131f"
STUDY_CONFIG_HASHES = {1: "cf29640202a2", 5: "e58225122cd4"}
STUDY_BASE = {
    "area": {"width": 60.0, "height": 40.0, "cell_size": 2.0},
    "swarm": {"n_drones": 5, "altitude": 12.0, "max_speed": 3.0, "camera_fov_deg": 70.0, "camera_aspect": 1.3333},
    "sim": {"dt": 0.1, "duration_s": 120.0, "seed": 0},
    "channel": {"drop_rate": 0.0, "latency_s": 0.0, "sync_interval_s": 1.0},
    "scorer": {"kind": "cached", "period_s": 1.0, "model": "clipseg:mission", "cache_path": "runs/study/cache/clipseg-mission-s1.json"},
    "message": {"kind": "dense"},
    "fusion": {"kind": "age_weighted", "tau_s": 10.0},
    "controller": {"kind": "lloyd", "gain": 1.0},
    "prior": {"kind": "floor"},
    "scene": {"kind": "random", "seed": 1, "floor": 0.005},
    "name": "study",
}


def _digest(s: Scene) -> str:
    return json.dumps({
        "road": [s.road.y_center, s.road.width],
        "f": [[f.feature_id, f.label, list(f.position), f.yaw] for f in s.features],
        "st": [list(x) for x in s.drone_starts],
        "w": [s.width, s.height, s.seed],
    })


def test_road_mode_reproduces_seeds_1_to_24_bit_for_bit() -> None:
    h = hashlib.sha256()
    for k in range(1, 25):
        h.update(_digest(random_scene(k)).encode())
        h.update(_digest(random_scene(k, 3, 4, 60.0, 40.0, 5, 12.0, "road")).encode())
    assert h.hexdigest() == ROAD_SEEDS_1_24_SHA256


def test_road_configs_dump_and_hash_as_before() -> None:
    assert config_hash(RunConfig()) == DEFAULT_CONFIG_HASH
    base = RunConfig.model_validate(STUDY_BASE)
    for seed, expected in STUDY_CONFIG_HASHES.items():
        cfg = apply_overrides(base, {"scene.seed": seed})
        assert "target_mode" not in cfg.model_dump(mode="json")["scene"]
        assert config_hash(cfg) == expected


def test_offroad_config_round_trips_and_differs() -> None:
    base = RunConfig.model_validate(STUDY_BASE)
    off = apply_overrides(base, {"scene.seed": 205, "scene.target_mode": "offroad"})
    again = RunConfig.model_validate_json(off.model_dump_json())
    assert again.scene.target_mode == "offroad"
    built, direct = build_scene(again), random_scene(205, 3, 4, 60.0, 40.0, 5, 12.0, "offroad")
    assert (built.road, built.features) == (direct.road, direct.features)
    assert config_hash(off) != config_hash(apply_overrides(base, {"scene.seed": 205}))


def test_target_mode_needs_a_random_scene() -> None:
    with pytest.raises(ValueError, match="target_mode"):
        RunConfig.model_validate({"scene": {"kind": "default", "target_mode": "offroad"}})
    with pytest.raises(ValueError, match="target_mode"):
        random_scene(1, target_mode="field")


@pytest.mark.parametrize("seed", range(205, 225))
def test_offroad_targets_are_far_from_the_road_and_confusers_far_from_targets(seed: int) -> None:
    s = random_scene(seed, 3, 4, 60.0, 40.0, 5, 12.0, "offroad")
    assert s.road == random_scene(seed).road
    targets, distractors = s.targets(), s.distractors()
    assert len(targets) == 3 and len(distractors) == 4
    assert {t.label for t in targets} <= OFFROAD_TARGET_LABELS
    for t in targets:
        assert abs(t.position[1] - s.road.y_center) >= max(10.0, OFFROAD_MIN_OFFSET_M) - 1e-9
    for d in distractors:
        dist = min(np.hypot(d.position[0] - t.position[0], d.position[1] - t.position[1]) for t in targets)
        assert dist >= OFFROAD_DISTRACTOR_CLEARANCE_M - 1e-9
        assert abs(d.position[1] - s.road.y_center) >= s.road.width / 2 + 2.0 - 1e-9


@pytest.mark.parametrize("seed", range(205, 225))
def test_offroad_road_band_lies_outside_the_near_mask(seed: int) -> None:
    """The analysis' masks: road band |y - road| <= width/2 + 2, near = within 8 m of a target cell."""
    s = random_scene(seed, 3, 4, 60.0, 40.0, 5, 12.0, "offroad")
    grid = Grid(60.0, 40.0, 2.0)
    gt = rasterize_ground_truth(s, grid)
    centres = grid.cell_centers()
    target = gt.values > gt.values.min() + 1e-9
    on_road = np.abs(centres[..., 1] - s.road.y_center) <= s.road.width / 2 + 2.0
    tgt = centres[target]
    dist = np.min(np.linalg.norm(centres[:, :, None, :] - tgt[None, None], axis=-1), axis=-1)
    assert target.any()
    assert not (on_road & (dist <= 8.0)).any()
