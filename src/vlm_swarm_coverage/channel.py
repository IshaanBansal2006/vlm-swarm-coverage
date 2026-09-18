"""The inter-drone channel: messages go in from senders, come out to receivers.

A channel is anything with `send` and `deliver`. `PerfectChannel` delivers every message to
every other drone at the next `deliver` call. Faulted channels implement the same two calls and
own their randomness, so the loop does not know which one it is talking to.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from vlm_swarm_coverage.schemas import BeliefMessage, PoseMessage

@dataclass(frozen=True)
class Delivery:
    """One message arriving at one receiver, with the sim time it arrived."""

    receiver: int
    t: float
    message: BeliefMessage | PoseMessage


class Channel(Protocol):
    def send(self, message: BeliefMessage | PoseMessage, t: float, receivers: list[int]) -> None: ...

    def deliver(self, t: float) -> list[Delivery]:
        """Everything due at or before `t`, removed from the channel."""
        ...


class PerfectChannel:
    def __init__(self) -> None:
        self._pending: list[Delivery] = []
        self.sent_bytes: dict[int, int] = defaultdict(int)

    def send(self, message: BeliefMessage | PoseMessage, t: float, receivers: list[int]) -> None:
        self.sent_bytes[message.sender] += message.nbytes
        for r in receivers:
            self._pending.append(Delivery(r, t, message))

    def deliver(self, t: float) -> list[Delivery]:
        due = [d for d in self._pending if d.t <= t]
        self._pending = [d for d in self._pending if d.t > t]
        return due


@dataclass(frozen=True)
class Drop:
    """A message that did not reach one receiver, and why."""

    sender: int
    receiver: int
    t: float
    reason: str
    nbytes: int


class FaultedChannel:
    """Per-link Bernoulli loss, fixed latency, and a byte budget per sender per sync (decision 031).

    Loss is decided independently for every (message, receiver) pair. The budget resets whenever
    a sender sends at a new sim time; messages are admitted in send order until the next one does
    not fit, and a message that does not fit is dropped whole, never truncated. All randomness
    comes from one seeded generator so a run is reproducible from its config.
    """

    def __init__(self, drop_rate: float = 0.0, latency_s: float = 0.0, bytes_per_sync: int | None = None, seed: int = 0) -> None:
        if not 0.0 <= drop_rate <= 1.0:
            raise ValueError(f"drop_rate must be in [0, 1], got {drop_rate}")
        if latency_s < 0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        if bytes_per_sync is not None and bytes_per_sync <= 0:
            raise ValueError(f"bytes_per_sync must be positive or None, got {bytes_per_sync}")
        self.drop_rate, self.latency_s, self.bytes_per_sync = drop_rate, latency_s, bytes_per_sync
        self.rng = np.random.default_rng(seed)
        self._pending: list[Delivery] = []
        self._budget: dict[int, tuple[float, int]] = {}
        self.drops: list[Drop] = []
        self.stats: dict[str, int] = defaultdict(int)

    def _within_budget(self, sender: int, t: float, nbytes: int) -> bool:
        if self.bytes_per_sync is None:
            return True
        last_t, used = self._budget.get(sender, (None, 0))
        if last_t != t:
            used = 0
        if used + nbytes > self.bytes_per_sync:
            self._budget[sender] = (t, used)
            return False
        self._budget[sender] = (t, used + nbytes)
        return True

    def send(self, message: BeliefMessage | PoseMessage, t: float, receivers: list[int]) -> None:
        n = message.nbytes
        self.stats["sent"] += len(receivers)
        self.stats["bytes_offered"] += n * len(receivers)
        if not self._within_budget(message.sender, t, n):
            for r in receivers:
                self.drops.append(Drop(message.sender, r, t, "budget", n))
            self.stats["dropped_budget"] += len(receivers)
            return
        for r in receivers:
            if self.rng.random() < self.drop_rate:
                self.drops.append(Drop(message.sender, r, t, "loss", n))
                self.stats["dropped_loss"] += 1
                continue
            self._pending.append(Delivery(r, t + self.latency_s, message))
            self.stats["bytes_sent"] += n

    def deliver(self, t: float) -> list[Delivery]:
        due = [d for d in self._pending if d.t <= t + 1e-9]
        self._pending = [d for d in self._pending if d.t > t + 1e-9]
        self.stats["delivered"] += len(due)
        return due

    def drain_drops(self) -> list[Drop]:
        out, self.drops = self.drops, []
        return out
