"""Belief fusion: how a drone folds received beliefs into its own.

Two rules. `IgnoreMessages` is the zero-communication endpoint: nothing received changes the
belief. `AgeWeightedAverage` (decision 022) is a per-cell weighted mean of the drone's own belief
and every peer belief delivered since the last sync, where each contribution is weighted by how
recently the underlying observation was made: w = exp(-age / tau). Cells a contributor has never
observed carry zero weight, so a peer's prior never leaks into a cell the receiver has seen, and
the receiver's own prior gives way to any observation at all.

The rule is decentralized (each drone runs it on what it received), deterministic, and does not
force agreement: two drones that miss each other's messages keep different beliefs, and two that
exchange them converge on the cells both have observed. `tau` interpolates between two textbook
rules: tau -> 0 is "freshest observation wins", tau -> inf is equal-weight consensus averaging.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from vlm_swarm_coverage.field import ImportanceField
    from vlm_swarm_coverage.schemas import AnyBeliefMessage


class BeliefFusion(Protocol):
    def fuse(
        self, own: ImportanceField, received: Sequence[AnyBeliefMessage], t: float
    ) -> ImportanceField:
        """Return the drone's updated belief given its own field and the messages delivered
        since the last sync. May mutate and return `own`, or return a new field on the same grid."""
        ...


class IgnoreMessages:
    """No fusion. The drone's belief is only what it has seen itself."""

    def fuse(
        self, own: ImportanceField, received: Sequence[AnyBeliefMessage], t: float
    ) -> ImportanceField:
        return own


def latest_per_sender(received: Sequence[AnyBeliefMessage]) -> list[AnyBeliefMessage]:
    """One message per sender, the highest sequence number. A belief message is a snapshot, so an
    older one delivered late (latency bunches them) adds nothing the newer one does not."""
    latest: dict[int, AnyBeliefMessage] = {}
    for m in received:
        cur = latest.get(m.sender)
        if cur is None or m.seq > cur.seq:
            latest[m.sender] = m
    return [latest[k] for k in sorted(latest)]


class AgeWeightedAverage:
    """Per-cell mean of own and received values, weighted by exp(-age / tau_s) (decision 022)."""

    def __init__(self, tau_s: float = 10.0) -> None:
        if tau_s < 0 or math.isnan(tau_s):
            raise ValueError(f"tau_s must be >= 0 (0 = freshest wins, inf = equal weights), got {tau_s}")
        self.tau_s = tau_s

    def weights(self, stamps: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        """(k, rows, cols) weights for (k, rows, cols) stamps; zero where never observed."""
        observed = ~np.isnan(stamps)
        if self.tau_s == 0.0:
            newest = np.where(observed, stamps, -np.inf).max(axis=0)
            return (observed & (stamps == newest)).astype(np.float64)
        if math.isinf(self.tau_s):
            return observed.astype(np.float64)
        with np.errstate(invalid="ignore"):
            age = np.clip(t - stamps, 0.0, None)
            return np.where(observed, np.exp(-age / self.tau_s), 0.0)

    def fuse(
        self, own: ImportanceField, received: Sequence[AnyBeliefMessage], t: float
    ) -> ImportanceField:
        if own.stamps is None:
            raise ValueError(
                "AgeWeightedAverage needs a belief with observation stamps; "
                "build it with ImportanceField.prior(grid, value)"
            )
        messages = latest_per_sender(received)
        if not messages:
            return own
        fields = [own] + [m.to_field(own.grid) for m in messages]
        values = np.stack([f.values for f in fields])
        stamps = np.stack([f.stamps for f in fields])  # type: ignore[misc]
        w = self.weights(stamps, t)
        total = w.sum(axis=0)
        has = total > 0.0
        fused = (w * values).sum(axis=0) / np.where(has, total, 1.0)
        own.values = np.where(has, fused, own.values)
        newest = np.where(np.isnan(stamps), -np.inf, stamps).max(axis=0)
        own.stamps = np.where(has, newest, own.stamps)
        return own
