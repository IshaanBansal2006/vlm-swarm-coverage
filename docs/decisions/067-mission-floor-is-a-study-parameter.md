# 067 — The mission floor is a study parameter, and the study's is 0.005

**Status:** Accepted 2026-09-21
**Area:** scene and evaluation

## Context

Every cell carries a floor importance so that a density-weighted controller always has a
gradient. The default floor of 0.05 on a 600-cell grid is 30 units of mass against about 12 for
the three targets: the mission's own answer key said the background was 70 % of what matters,
and Lloyd descent on it spent most of its effort spreading drones over area. A target-focused
measure of coverage then had no controller that optimised it, not even the oracle.

## Options

| Option | Trade-off |
|---|---|
| Keep 0.05 and measure area coverage | Consistent, but not the question the mission asks |
| Lower the default in the scene module | Changes the demo and every earlier tag's meaning |
| **A per-run override, `scene.floor`, with the study at 0.005** | The demo is untouched; the study's answer key is 91 % target mass; the floor still exists so beliefs are never all-zero |

## Decision

`scene.floor` overrides the mission's floor when the scene is built (the Isaac scene script
honours it too, for the oracle). The study base sets 0.005. The primary coverage measure is the
locational cost on the true density relative to an oracle-controlled swarm on the same layout and
seed ("locational efficiency", in (0, 1]), which has no window to choose and whose optimum is by
construction a run the study contains.

## Consequences

- Caches are rebuilt from the raw stores with the calibration's floor at 0.005; the oracle cache is
  re-derived from the answer key.
- A belief that has seen nothing is nearly flat; a drone with no peers goes to the centroid of a
  nearly flat field, which is still the map's centre. The random road offset (066) is what keeps
  that from being a good place.
