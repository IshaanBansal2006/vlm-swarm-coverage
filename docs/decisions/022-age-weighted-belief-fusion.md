# 022 — Belief fusion: per-cell average weighted by observation age

**Status:** Accepted 2026-09-20
**Area:** coverage control and the decentralized layer

## Context

Until this record each drone's belief was only what it had seen itself; messages were carried
and logged but never used. The rig needs a rule that turns received beliefs into an update of the
receiver's own, that is decentralized (no leader, no global state), deterministic (a run is
reproducible from its seed), and that does not silently mix a peer's unobserved prior into cells
the receiver has real evidence about.

## Options

| Rule | For | Against |
|---|---|---|
| Max per cell | Never loses a detection; what most VLM-swarm systems do | Every agent converges to the same field as soon as messages get through; one spurious high value propagates everywhere and never decays; no way to weigh a stale value less |
| Freshest observation wins | Zero parameters; consistent with the local overwrite update (011) | No averaging at all, so per-observation noise passes straight through; a late-arriving old message can be ignored but a fresh wrong one is taken whole |
| Equal-weight consensus averaging | The textbook decentralized rule (linear consensus); converges to agreement on a connected graph | Treats a 30-second-old value the same as a fresh one; staleness from latency is invisible to it |
| **Age-weighted average, `w = exp(-age/τ)`** | One parameter that interpolates the two rules above (τ→0 freshest, τ→∞ equal); staleness from latency enters the update directly; priors excluded by construction | τ must be chosen and defended; averaging without a variance model double-counts values that circulate (see consequences) |
| Covariance intersection / Dempster–Shafer discounting | Principled handling of correlated inputs and conflict | Needs per-cell uncertainty on the wire and a conflict model; a method contribution in its own right, not a baseline |

## Decision

`AgeWeightedAverage(tau_s)`: for every cell, the new value is the mean of the drone's own value
and each peer's value, each weighted by `exp(-age/τ)` where age is the time since the observation
behind that value, on the receiver's clock. Cells a contributor never observed have weight zero.
The fused cell takes the newest contributing stamp. Only the newest message per sender in a sync
is used. Default τ = 10 s (ten sync intervals in the study config); `0` and `inf` are accepted and
give the two limiting rules, so a sensitivity check is one config line.

This requires the belief message to carry per-cell observation age (decision 032) and the belief
to carry per-cell stamps (`ImportanceField.stamps`).

## Consequences

- Two drones that exchange messages converge on the cells both have observed; two that miss each
  other's messages keep different beliefs. The rule does not force agreement.
- A drone's prior is never transmitted as evidence: a peer's unobserved cells contribute nothing,
  and the drone's own prior yields to any observation at all.
- Values that circulate get counted more than once (A fuses B's value, B later averages a value
  that already contains its own). With convex weights the result cannot leave the range of the
  underlying observations, but it does drift toward the most-shared values. This is a property of
  naive averaging, stated here rather than engineered away.
- The message grows from 4 to 6 bytes per cell.
- `configs/demo.toml` and `configs/study.toml` set `[fusion]` explicitly; the code default is the
  same, so an unconfigured run fuses.
