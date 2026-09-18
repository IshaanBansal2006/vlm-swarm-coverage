# 014 — The cache is keyed to a scorer; the stability study stores raw scores

**Status:** Accepted 2026-09-18
**Area:** evaluation harness

## Context

The rig is meant to outlive its first study: a later study swaps a different model into the
scorer slot and asks whether anything moved. That only works if nothing collected for the first
scorer can be silently mistaken for the second's, and if the first study's protocol reruns on the
second scorer without new design.

## Decision

- **The importance cache carries a header** (scorer id, grid, cell size, altitude step) and
  refuses to load a file written for any other combination. Caches for two scorers live side by
  side under different paths; a sweep replays against either by config.
- **The stability study records raw per-view scores**, not only fitted summaries: one JSONL
  record per view with the pose, the scorer id and the observation. Frames are not stored; the
  pose plus the scene regenerate them. A pose sweep is a fixed-order Cartesian product so two
  scorers see the identical protocol.

## Options considered

| Option | Why not |
|---|---|
| One cache file, overwrite on model change | The failure is silent: a sweep "with model B" reads model A's cache and reports A's numbers. |
| Store fitted parameters only | A second scorer needs the study re-collected, and the two cannot be compared frame for frame. |
| Store frames too | Tens of gigabytes for nothing: the simulator regenerates any frame from a pose. |

## Consequences

- Pre-existing cache files without a header are rejected with the instruction to re-score.
- A second study's only change is the scorer id in the config and a new cache path.
- The score store is an input to analysis, not an analysis: no metric is computed here.
