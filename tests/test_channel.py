from __future__ import annotations

import numpy as np
import pytest

from vlm_swarm_coverage.channel import FaultedChannel, PerfectChannel
from vlm_swarm_coverage.field import Grid, ImportanceField
from vlm_swarm_coverage.schemas import BeliefMessage, PoseMessage


def _pose(sender: int = 0, seq: int = 0, t: float = 0.0) -> PoseMessage:
    return PoseMessage(sender=sender, seq=seq, t=t, x=1.0, y=2.0, z=12.0)


def _belief(sender: int = 0, t: float = 0.0) -> BeliefMessage:
    g = Grid(20.0, 10.0, 2.0)
    return BeliefMessage.from_field(sender, 0, t, ImportanceField.uniform(g, 0.5))


def test_perfect_channel_delivers_everything_once() -> None:
    ch = PerfectChannel()
    ch.send(_pose(), 0.0, [1, 2])
    out = ch.deliver(0.0)
    assert sorted(d.receiver for d in out) == [1, 2]
    assert ch.deliver(0.0) == []
    assert ch.sent_bytes[0] == _pose().nbytes


def test_zero_faults_is_perfect() -> None:
    ch = FaultedChannel(seed=3)
    ch.send(_belief(), 0.0, [1, 2])
    assert len(ch.deliver(0.0)) == 2
    assert ch.stats["dropped_loss"] == 0 and ch.drain_drops() == []


def test_loss_is_per_link_seeded_and_at_the_right_rate() -> None:
    a, b = FaultedChannel(drop_rate=0.3, seed=11), FaultedChannel(drop_rate=0.3, seed=11)
    for i in range(500):
        a.send(_pose(seq=i), float(i), [1, 2])
        b.send(_pose(seq=i), float(i), [1, 2])
    assert a.stats == b.stats
    assert a.stats["dropped_loss"] == pytest.approx(300, abs=45)
    assert all(d.reason == "loss" for d in a.drain_drops())
    assert [(d.receiver, d.message.seq) for d in a.deliver(1000.0)] == [(d.receiver, d.message.seq) for d in b.deliver(1000.0)]


def test_latency_delays_delivery() -> None:
    ch = FaultedChannel(latency_s=1.5)
    ch.send(_pose(), 0.0, [1])
    assert ch.deliver(1.0) == []
    assert len(ch.deliver(1.5)) == 1


def test_byte_budget_admits_in_order_and_resets_per_sync() -> None:
    pose, belief = _pose(), _belief()
    ch = FaultedChannel(bytes_per_sync=pose.nbytes + 10)
    ch.send(pose, 0.0, [1])
    ch.send(belief, 0.0, [1])
    drops = ch.drain_drops()
    assert [d.reason for d in drops] == ["budget"] and drops[0].nbytes == belief.nbytes
    assert len(ch.deliver(0.0)) == 1
    ch.send(pose, 1.0, [1])
    assert len(ch.deliver(1.0)) == 1 and ch.drain_drops() == []
    assert ch.stats["dropped_budget"] == 1


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="drop_rate"):
        FaultedChannel(drop_rate=1.5)
    with pytest.raises(ValueError, match="latency"):
        FaultedChannel(latency_s=-1)
    with pytest.raises(ValueError, match="bytes_per_sync"):
        FaultedChannel(bytes_per_sync=0)


def test_rng_is_isolated_from_global_state() -> None:
    np.random.seed(0)
    a = FaultedChannel(drop_rate=0.5, seed=1)
    np.random.seed(99)
    b = FaultedChannel(drop_rate=0.5, seed=1)
    assert a.rng.random() == b.rng.random()
