from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from vlm_swarm_coverage.bridge import RemoteScorer, pose_payload, yaw_quat
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.schemas import ObservationMessage
from vlm_swarm_coverage.scoring import Observation, View
from vlm_swarm_coverage.simulation import build
from vlm_swarm_coverage.world import KinematicWorld


def test_yaw_quat_is_unit_and_matches_half_angle() -> None:
    q = yaw_quat(np.pi / 2)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert q[0] == pytest.approx(np.cos(np.pi / 4)) and q[3] == pytest.approx(np.sin(np.pi / 4))


def test_pose_payload_matches_renderer_schema() -> None:
    w = KinematicWorld(((1.0, 2.0, 12.0), (3.0, 4.0, 12.0)), 60.0, 40.0, 3.0, 0.1)
    data = json.loads(pose_payload(w))
    assert data["t"] == 0.0 and [p["drone_id"] for p in data["poses"]] == [0, 1]
    assert data["poses"][1]["position"] == [3.0, 4.0, 12.0] and len(data["poses"][0]["orientation"]) == 4


def test_observation_message_round_trip_and_size() -> None:
    obs = Observation(2, 1.5, np.array([[0, 1], [3, 4], [7, 7]]), np.array([0.1, 0.9, 0.5]))
    m = ObservationMessage.from_observation(obs, seq=9)
    assert m.nbytes == 20 + 3 * (4 + 4)
    back = ObservationMessage.from_bytes(m.to_bytes())
    assert back == m
    np.testing.assert_array_equal(back.to_observation().cells, obs.cells)
    np.testing.assert_allclose(back.to_observation().values, obs.values, atol=1e-7)
    with pytest.raises(ValueError, match="needs"):
        ObservationMessage.from_bytes(m.to_bytes()[:-3])


@dataclass
class _FakeBridge:
    latest: dict[int, ObservationMessage] = field(default_factory=dict)
    spins: int = 0

    def spin(self) -> None:
        self.spins += 1


def test_remote_scorer_serves_latest_or_empty() -> None:
    bridge = _FakeBridge()
    scorer = RemoteScorer(bridge)  # type: ignore[arg-type]
    view = View(0, 5.0, 1.0, 1.0, 12.0, 0.0, 70.0)
    empty = scorer.score(view)
    assert len(empty.cells) == 0 and scorer.empty == 1
    obs = Observation(0, 4.0, np.array([[1, 1]]), np.array([0.7]))
    bridge.latest[0] = ObservationMessage.from_observation(obs, seq=0)
    served = scorer.score(view)
    assert served.t == 5.0 and served.values[0] == pytest.approx(0.7, abs=1e-7)
    assert bridge.spins == 2


def test_on_step_hook_and_scorer_injection(tmp_path) -> None:  # type: ignore[no-untyped-def]
    calls: list[float] = []
    cfg = RunConfig.model_validate({"name": "t", "sim": {"duration_s": 1.0}})
    bridge = _FakeBridge()
    sim = build(cfg, scorer=RemoteScorer(bridge), on_step=lambda s, t: calls.append(t))  # type: ignore[arg-type]
    from vlm_swarm_coverage.artifacts import RunDir

    sim.run(RunDir.create(tmp_path, cfg))
    assert len(calls) == cfg.sim.n_steps and calls[-1] == pytest.approx(1.0)
    assert isinstance(sim.scorer, RemoteScorer)
