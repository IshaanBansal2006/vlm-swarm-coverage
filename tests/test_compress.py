from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.compress import (
    DenseEncoder,
    PosesOnly,
    QuantisedEncoder,
    TopKEncoder,
    build_encoder,
)
from vlm_swarm_coverage.config import MessageConfig, RunConfig
from vlm_swarm_coverage.consensus import AgeWeightedAverage
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.schemas import AGE_NEVER, QuantisedBeliefMessage, SparseBeliefMessage
from vlm_swarm_coverage.simulation import build, run_from_config

if TYPE_CHECKING:
    from pathlib import Path

G = Grid(6.0, 4.0, 2.0)  # 2 rows x 3 cols


def _field() -> ImportanceField:
    f = ImportanceField.prior(G, 0.05)
    f.values[:] = [[0.9, 0.05, 0.3], [0.05, 0.7, 0.05]]
    f.stamps[:] = [[9.0, 8.0, np.nan], [np.nan, 5.0, 7.0]]
    return f


def test_top_k_picks_highest_observed_values_and_round_trips() -> None:
    m = SparseBeliefMessage.top_k(1, 4, 10.0, _field(), k=2)
    assert m.cells == (0, 4) and m.values == pytest.approx((0.9, 0.7)) and m.ages == (10, 50)
    assert m.nbytes == 28 + 2 * 8
    back = SparseBeliefMessage.from_bytes(m.to_bytes())
    assert back.cells == m.cells and back.ages == m.ages
    np.testing.assert_allclose(back.values, m.values, atol=1e-7)
    f = back.to_field(G)
    assert f.observed.sum() == 2 and f.stamps[0, 0] == pytest.approx(9.0) and np.isnan(f.stamps[0, 2])
    assert f.values[1, 1] == pytest.approx(0.7, abs=1e-7) and f.values[0, 1] == 0.0


def test_top_k_never_sends_unobserved_cells_and_handles_small_k() -> None:
    f = _field()
    m = SparseBeliefMessage.top_k(0, 0, 10.0, f, k=10)
    assert set(m.cells) == {0, 1, 4, 5}  # the four observed cells, not the six
    assert SparseBeliefMessage.top_k(0, 0, 10.0, f, k=0).cells == ()
    assert SparseBeliefMessage.top_k(0, 0, 10.0, f, k=0).nbytes == 28
    empty = SparseBeliefMessage.top_k(0, 0, 10.0, ImportanceField.prior(G, 0.05), k=3)
    assert empty.cells == () and np.isnan(empty.to_field(G).stamps).all()
    with pytest.raises(ValueError, match="k must be"):
        SparseBeliefMessage.top_k(0, 0, 0.0, f, k=-1)
    with pytest.raises(ValidationError, match="unique"):
        SparseBeliefMessage(sender=0, seq=0, t=0.0, rows=2, cols=3, cells=(1, 1), values=(0.0, 0.0), ages=(0, 0))
    with pytest.raises(ValueError, match="needs"):
        SparseBeliefMessage.from_bytes(m.to_bytes()[:-3])


def test_quantised_round_trip_precision_and_size() -> None:
    f = _field()
    m = QuantisedBeliefMessage.from_field(2, 1, 10.0, f)
    assert m.nbytes == 28 + 2 * 6 and m.scale == pytest.approx(0.9)
    back = QuantisedBeliefMessage.from_bytes(m.to_bytes()).to_field(G)
    np.testing.assert_allclose(back.values, f.values, atol=0.9 / 255 / 2 + 1e-9)
    assert back.stamps[0, 0] == pytest.approx(9.0) and back.stamps[1, 1] == pytest.approx(5.0)
    assert np.isnan(back.stamps[0, 2]) and back.observed.sum() == 4
    with pytest.raises(ValueError, match="needs"):
        QuantisedBeliefMessage.from_bytes(m.to_bytes()[:-1])


def test_encoders_and_config() -> None:
    f = _field()
    assert DenseEncoder().encode(0, 0, 1.0, f).nbytes == 24 + 6 * 6
    assert TopKEncoder(3).encode(0, 0, 1.0, f).nbytes == 28 + 3 * 8
    assert QuantisedEncoder().encode(0, 0, 1.0, f).nbytes == 28 + 2 * 6
    assert PosesOnly().encode(0, 0, 1.0, f) is None
    assert isinstance(build_encoder(MessageConfig(kind="topk", k=5)), TopKEncoder)
    assert isinstance(build_encoder(MessageConfig(kind="none")), PosesOnly)
    with pytest.raises(ValidationError, match="needs message.k"):
        MessageConfig(kind="topk")
    with pytest.raises(ValidationError, match="only applies"):
        MessageConfig(kind="dense", k=3)
    with pytest.raises(ValueError, match="k >= 0"):
        TopKEncoder(-1)


def test_fusion_consumes_sparse_and_quantised_messages_alike() -> None:
    own = ImportanceField.prior(G, 0.05)
    sparse = SparseBeliefMessage.top_k(1, 0, 10.0, _field(), k=2)
    out = AgeWeightedAverage(float("inf")).fuse(own, [sparse], t=10.0)
    assert out.values[0, 0] == pytest.approx(0.9, abs=1e-6) and out.values[0, 1] == 0.05 and np.isnan(out.stamps[0, 1])
    own2 = ImportanceField.prior(G, 0.05)
    quant = QuantisedBeliefMessage.from_field(1, 0, 10.0, _field())
    out2 = AgeWeightedAverage(float("inf")).fuse(own2, [quant], t=10.0)
    assert out2.observed.sum() == 4 and out2.values[1, 1] == pytest.approx(0.7, abs=0.01)


def _short(**kw: object) -> RunConfig:
    base = {"name": "t", "sim": {"dt": 0.1, "duration_s": 20.0, "seed": 1}}
    base.update(kw)
    return RunConfig.model_validate(base)


def test_poses_only_sends_no_belief_bytes_and_still_runs(tmp_path: Path) -> None:
    run = run_from_config(_short(message={"kind": "none"}), tmp_path)
    sends = list(run.iter_events("send"))
    assert sends and all(e["belief_bytes"] == 0 and e["bytes"] == 48 for e in sends)
    assert all(e["msg"] == "PoseMessage" for e in run.iter_events("recv"))
    sim = build(_short(message={"kind": "none"}))
    assert isinstance(sim.encoder, PosesOnly)


def test_top_k_run_lands_between_dense_and_none(tmp_path: Path) -> None:
    def spread(message: dict[str, object]) -> float:
        sim = build(_short(message=message))
        sim.run(__import__("vlm_swarm_coverage.artifacts").artifacts.RunDir.create(tmp_path / str(message), sim.config))
        return float(np.abs(sim.agents[0].belief.values - sim.agents[1].belief.values).sum())

    dense, topk, none = spread({"kind": "dense"}), spread({"kind": "topk", "k": 8}), spread({"kind": "none"})
    assert dense <= topk <= none and dense < none
    run = run_from_config(_short(message={"kind": "topk", "k": 8}), tmp_path / "k")
    assert all(e["belief_bytes"] <= 28 + 8 * 8 for e in run.iter_events("send"))
    assert any(e["ages"] and min(e["ages"]) < AGE_NEVER for e in run.iter_events("belief"))
