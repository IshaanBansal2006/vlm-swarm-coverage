"""The inter-drone channel: messages go in from senders, come out to receivers.

A channel is anything with `send` and `deliver`. `PerfectChannel` delivers every message to
every other drone at the next `deliver` call. Faulted channels implement the same two calls and
own their randomness, so the loop does not know which one it is talking to.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vlm_swarm_coverage.schemas import BeliefMessage, PoseMessage

Message = "BeliefMessage | PoseMessage"


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
