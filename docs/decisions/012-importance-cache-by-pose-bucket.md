# 012 — Importance cache keyed by position and altitude

**Status:** Accepted 2026-09-18
**Area:** perception

## Context

Model inference per drone per look is the dominant cost of any run that uses a real scorer, and
a static scene scored from the same place gives the same answer. Runs that differ only in the
swarm's behaviour re-score the same poses over and over.

## Decision

`CachedScorer` wraps any scorer and memoises by bucket: the grid cell under the drone and its
altitude rounded to a step. Yaw is not part of the key. The cache persists to a JSON file and is
loaded on construction, so a scene is scored once and every later run reads.

## Options considered

| Key | Trade-off |
|---|---|
| Position only | Cheapest; loses altitude, which changes the footprint and what the model can resolve. |
| **Position × altitude** | Keeps the altitude effect; assumes the scorer is view-invariant in yaw. |
| Position × altitude × yaw | Exact, but multiplies the cache by the yaw resolution and removes most of the saving. |

## Consequences

- The yaw-invariance assumption is stated on the class and must be stated wherever cached results
  are reported. A live-scorer check of that assumption is a separate piece of work.
- The cache is part of the apparatus: its file is an input to a run and belongs in the run's
  provenance, not in git (`cache/importance/` is ignored).
- Oracle runs are cached too. That is harmless and keeps one code path.
