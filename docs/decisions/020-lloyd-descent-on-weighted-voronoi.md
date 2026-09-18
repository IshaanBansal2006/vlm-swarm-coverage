# 020 — Coverage controller: Lloyd descent on the density-weighted Voronoi partition

**Status:** Accepted 2026-09-18
**Area:** coverage control

## Context

The importance field has to move the drones, decentrally, in a way the literature recognises so
that the study is about the field and not about the controller.

## Options

| Option | Trade-off |
|---|---|
| **Lloyd descent on weighted Voronoi** (Cortés et al. 2004) | The canonical decentralized coverage controller. Gradient descent on the locational cost; each drone needs only its own density and its peers' positions. Converges to a local optimum, not a global one. |
| Lawnmower / serpentine sweep of a Voronoi cell | Deterministic, complete coverage of the cell, but ignores the density inside it. The importance field would only choose cell shapes, not where a drone spends its time. |
| Information-gain / next-best-view planning | Directly optimises what to look at next. Richer, but a planning problem with its own horizon, cost and approximation choices that would compete with the field for explanatory credit. |
| Potential-field attraction to hot cells | Simple and reactive; no partition, so drones pile onto the same peak. |

## Decision

Lloyd descent. Each drone owns the cells nearer to it than to any peer (ties to the lower id),
computes the centroid of that region under its own belief as the density, and commands velocity
`gain × (centroid − position)`, saturated by the world's speed limit. Peers are known only
through the last pose message received.

## Why

The locational cost `H = Σ_i ∫_{V_i} ‖q − p_i‖² φ(q) dq` has partial derivative
`−2 M_i (C_i − p_i)` with respect to `p_i`, so moving toward the centroid is descent on `H`, and
a drone at its centroid has no gradient. With `φ` equal to the importance belief, the controller
spends time in proportion to importance without any extra machinery. Every effect of a wrong
belief on motion is then a property of the belief, not of a planner's heuristics.

## Consequences

- The controller is memoryless: it reads the belief and the last-known peer poses each step.
  Stale peer poses on a faulted channel change the partition, which is part of what a channel
  fault does to coverage.
- Convergence is local. Start positions matter and are part of the scene, hence of the seed.
- The zero-density case cannot occur because the mission floor keeps `φ` positive everywhere.
- The controller only ever sees the drone's own belief. What "own belief" contains after sharing
  is the fusion rule's business.
