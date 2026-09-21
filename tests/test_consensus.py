from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.consensus import AgeWeightedAverage, IgnoreMessages, latest_per_sender
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.schemas import BeliefMessage

G = Grid(4.0, 2.0, 2.0)  # 1 row, 2 cols


def _belief(values: list[float], stamps: list[float], sender: int = 1, seq: int = 0, t: float = 10.0) -> BeliefMessage:
    f = ImportanceField(G, np.array([values]), np.array([stamps]))
    return BeliefMessage.from_field(sender, seq, t, f)


def _own(values: list[float], stamps: list[float]) -> ImportanceField:
    return ImportanceField(G, np.array([values]), np.array([stamps]))


def test_freshest_wins_at_tau_zero() -> None:
    own = _own([0.2, 0.9], [9.0, 1.0])
    out = AgeWeightedAverage(0.0).fuse(own, [_belief([0.8, 0.1], [2.0, 8.0])], t=10.0)
    np.testing.assert_allclose(out.values, [[0.2, 0.1]])
    np.testing.assert_allclose(out.stamps, [[9.0, 8.0]])


def test_equal_weights_at_tau_inf() -> None:
    own = _own([0.2, 0.9], [9.0, 1.0])
    out = AgeWeightedAverage(float("inf")).fuse(own, [_belief([0.8, 0.1], [2.0, 8.0])], t=10.0)
    np.testing.assert_allclose(out.values, [[0.5, 0.5]])


def test_age_weighting_formula() -> None:
    own = _own([1.0, 0.0], [10.0, 0.0])
    out = AgeWeightedAverage(5.0).fuse(own, [_belief([0.0, 1.0], [5.0, 10.0])], t=10.0)
    w_old = np.exp(-1.0)
    np.testing.assert_allclose(out.values[0, 0], 1.0 / (1.0 + w_old))
    np.testing.assert_allclose(out.values[0, 1], 1.0 / (1.0 + np.exp(-2.0)))
    np.testing.assert_allclose(out.stamps, [[10.0, 10.0]])


def test_priors_never_leak_and_priors_give_way() -> None:
    own = _own([0.05, 0.7], [np.nan, 4.0])
    peer = _belief([0.4, 0.05], [3.0, np.nan])
    out = AgeWeightedAverage(10.0).fuse(own, [peer], t=10.0)
    assert out.values[0, 0] == pytest.approx(0.4) and out.stamps[0, 0] == pytest.approx(3.0)
    assert out.values[0, 1] == pytest.approx(0.7) and out.stamps[0, 1] == 4.0


def test_only_the_newest_message_per_sender_counts() -> None:
    old = _belief([0.0, 0.0], [1.0, 1.0], sender=1, seq=0)
    new = _belief([1.0, 1.0], [2.0, 2.0], sender=1, seq=1)
    assert latest_per_sender([old, new, old]) == [new]
    own = _own([0.05, 0.05], [np.nan, np.nan])
    out = AgeWeightedAverage(float("inf")).fuse(own, [old, new], t=10.0)
    np.testing.assert_allclose(out.values, [[1.0, 1.0]])


def test_no_messages_and_no_stamps() -> None:
    own = _own([0.3, 0.3], [1.0, np.nan])
    assert AgeWeightedAverage().fuse(own, [], t=5.0) is own
    with pytest.raises(ValueError, match="prior"):
        AgeWeightedAverage().fuse(ImportanceField.uniform(G, 0.3), [_belief([0, 0], [0, 0])], t=5.0)
    with pytest.raises(ValueError, match="tau_s"):
        AgeWeightedAverage(-1.0)
    assert IgnoreMessages().fuse(own, [_belief([0, 0], [0, 0])], t=5.0) is own


def test_disagreement_survives_a_lost_message() -> None:
    a = _own([0.9, 0.05], [5.0, np.nan])
    b = _own([0.05, 0.2], [np.nan, 5.0])
    rule = AgeWeightedAverage(10.0)
    rule.fuse(a, [BeliefMessage.from_field(1, 0, 5.0, b)], t=5.0)
    assert a.values[0, 1] == pytest.approx(0.2)
    assert b.values[0, 0] == pytest.approx(0.05)
