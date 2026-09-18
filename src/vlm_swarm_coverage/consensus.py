"""Belief fusion: how a drone folds received beliefs into its own.

The fusion rule is the author's to write. This module fixes the interface the rig calls and
provides the one rule that is not a rule: ignore everything received. That is the
no-communication case, useful as a baseline and as the stand-in that lets the loop run before a
real rule exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vlm_swarm_coverage.field import ImportanceField
    from vlm_swarm_coverage.schemas import BeliefMessage


class BeliefFusion(Protocol):
    def fuse(
        self, own: ImportanceField, received: Sequence[BeliefMessage], t: float
    ) -> ImportanceField:
        """Return the drone's updated belief given its own field and the messages delivered
        since the last sync. May mutate and return `own`, or return a new field on the same grid."""
        ...


class IgnoreMessages:
    """No fusion. The drone's belief is only what it has seen itself."""

    def fuse(
        self, own: ImportanceField, received: Sequence[BeliefMessage], t: float
    ) -> ImportanceField:
        return own
