from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.consensus import IgnoreMessages
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.schemas import BeliefMessage, PoseMessage


def test_pose_round_trip_and_size() -> None:
    p = PoseMessage(sender=2, seq=7, t=1.5, x=1.0, y=2.0, z=12.0, yaw=0.3)
    buf = p.to_bytes()
    assert p.nbytes == len(buf) == 4 + 4 + 8 + 4 * 8
    assert PoseMessage.from_bytes(buf) == p
    with pytest.raises(ValueError, match="bytes"):
        PoseMessage.from_bytes(buf[:-1])


def test_belief_round_trip_size_and_float32_quantisation() -> None:
    g = Grid(6.0, 4.0, 2.0)
    field = ImportanceField(g, np.arange(6, dtype=float).reshape(2, 3) / 7.0)
    m = BeliefMessage.from_field(sender=0, seq=3, t=2.0, field=field)
    assert m.nbytes == 24 + 4 * 6
    back = BeliefMessage.from_bytes(m.to_bytes())
    assert back == m
    np.testing.assert_allclose(back.to_field(g).values, field.values, atol=1e-7)
    assert not np.array_equal(back.to_field(g).values, field.values)


def test_belief_rejects_wrong_lengths_and_grids() -> None:
    with pytest.raises(ValidationError, match="cells"):
        BeliefMessage(sender=0, seq=0, t=0.0, rows=2, cols=2, values=(1.0, 2.0, 3.0))
    m = BeliefMessage(sender=0, seq=0, t=0.0, rows=2, cols=2, values=(1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="same area config"):
        m.to_field(Grid(6.0, 4.0, 2.0))
    with pytest.raises(ValueError, match="needs"):
        BeliefMessage.from_bytes(m.to_bytes()[:-2])


def test_ignore_messages_returns_own_belief_unchanged() -> None:
    g = Grid(4.0, 4.0, 2.0)
    own = ImportanceField.uniform(g, 0.3)
    other = BeliefMessage.from_field(1, 0, 0.0, ImportanceField.uniform(g, 0.9))
    out = IgnoreMessages().fuse(own, [other], t=1.0)
    assert out is own
    assert (out.values == 0.3).all()
