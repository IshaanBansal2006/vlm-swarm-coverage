# 021 — Drone motion: first-order kinematics at fixed altitude

**Status:** Accepted 2026-09-18
**Area:** coverage control

## Decision

Drones are velocity-commanded points. Per step: clip the command to the speed limit, advance by
`v·dt`, clamp to the survey rectangle, set yaw to the direction of travel. Altitude is constant.

## Options considered

| Option | Why not |
|---|---|
| Second-order (acceleration-limited) | Adds an inertia parameter that smooths trajectories and would be one more thing between belief error and motion. Not a variable in this study. |
| Simulator physics (Isaac articulation + flight controller) | Realistic, slow, and would make every run depend on a physics stack that is not the object of study. The simulator renders poses; it does not integrate them. |
| Variable altitude | Altitude changes the footprint and the scorer's view. Held fixed per run so it can be swept as a run-level variable rather than a control variable. |

## Consequences

- Runs are deterministic and fast: the 120 s demo runs in under two seconds without a renderer.
- Yaw tracking the velocity means a body-fixed nadir camera rotates with the drone, so the
  footprint rotates too. That is realistic for a fixed camera and is why yaw is in the view.
- A drone commanded into a wall stops at the wall. The controller never commands that because
  centroids lie inside the area.
