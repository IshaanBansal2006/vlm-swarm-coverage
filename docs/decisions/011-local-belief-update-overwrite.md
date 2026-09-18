# 011 — Local belief update: overwrite

**Status:** Accepted 2026-09-18
**Area:** perception

## Context

When a drone scores the cells in its view, those values must enter its own belief before it
shares anything. Three rules were weighed.

| Rule | For | Against |
|---|---|---|
| **Overwrite** | No parameters. Fastest response. Belief error equals scorer error, cell for cell. | No noise rejection; a single wrong reading holds until the next visit. |
| Exponential moving average | Smooths jitter. Bounded memory. | A rate to defend; delays recovery exactly as much as acquisition; first contact reads mostly prior; a second memory alongside fusion. |
| Count-weighted mean | Converges under stationary noise; count is a free confidence signal. | Never forgets; response shrinks with visits; needs a count grid; scene changes tracked badly. |

## Decision

Overwrite. `apply_observation` writes the observed values into the observed cells and nothing
else.

## Why

The study perturbs the scorer's output on purpose to see what the swarm does. That is only
interpretable if belief error is a known function of scorer error. Overwrite makes it the
identity. Both memory rules make it depend on visit history, which depends on the controller,
which depends on the belief: the perturbation would then be filtered by the thing it is meant to
perturb.

Noise handling, if the model needs it, belongs upstream inside the scorer (frame averaging,
coarser cells), where it does not touch the belief. Memory across agents belongs in the fusion
rule, which is a separate decision. One memory in the system, in one place.

## Consequences

- The scorer's frame-to-frame stability sets the floor on belief quality. That is a property to
  measure before any other experiment, not to hide.
- If a memory rule ever becomes necessary, this record is where the reasoning to overturn starts.
