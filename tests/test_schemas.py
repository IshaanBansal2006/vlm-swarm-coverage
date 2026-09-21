from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.consensus import IgnoreMessages
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.schemas import AGE_NEVER, BeliefMessage, PoseMessage


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
    assert m.nbytes == 24 + (4 + 2) * 6
    assert set(m.ages) == {AGE_NEVER}
    back = BeliefMessage.from_bytes(m.to_bytes())
    assert back == m
    np.testing.assert_allclose(back.to_field(g).values, field.values, atol=1e-7)
    assert not np.array_equal(back.to_field(g).values, field.values)
    assert np.isnan(back.to_field(g).stamps).all()


def test_belief_ages_carry_observation_stamps() -> None:
    g = Grid(6.0, 4.0, 2.0)
    field = ImportanceField.prior(g, 0.05)
    field.stamps[0, 1] = 7.3
    field.stamps[1, 2] = 9.0
    m = BeliefMessage.from_field(sender=1, seq=0, t=9.0, field=field)
    assert m.ages[1] == 17 and m.ages[5] == 0 and m.ages[0] == AGE_NEVER
    back = BeliefMessage.from_bytes(m.to_bytes()).to_field(g)
    assert back.stamps[0, 1] == pytest.approx(7.3) and back.stamps[1, 2] == 9.0
    assert back.observed.sum() == 2
    with pytest.raises(ValidationError, match="ages"):
        BeliefMessage(sender=0, seq=0, t=0.0, rows=1, cols=2, values=(1.0, 2.0), ages=(0,))


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
