# 010 — Scorer interface and the oracle scorer

**Status:** Accepted 2026-09-18
**Area:** perception

## Context

The model that turns a camera view into importance does not exist in the repository yet, and the
GPU host for it is not the host that runs the swarm loop. The rig still has to run end to end.

## Decision

- A scorer is any object with `score(View) -> Observation`. A `View` is a pose, the camera's
  field of view, an optional RGB frame and the mission text. An `Observation` is the cells in the
  footprint and one value each. No host, no ROS, no model in the interface.
- The camera is modelled as nadir (straight down). The footprint is the rectangle
  `2·z·tan(fov/2)` wide, that over the aspect ratio tall, rotated by yaw. A cell is in view if its
  centre is inside.
- `OracleScorer` returns the ground-truth value for each cell in view. It is the perfect model.
- `VLMScorer` exists as a typed slot that checks its dependencies at construction and refuses to
  score, so nothing can accidentally run "with a VLM" before one exists.

## Options considered

| Option | Why not |
|---|---|
| Scorer returns a whole field | Hides which cells were actually seen; every later question about observation coverage becomes unanswerable. |
| Scorer bound to ROS messages | Ties the interface to the transport; could not be tested in CI or run inside the simulator's Python. |
| Oblique / gimballed camera model | More realistic footprint, more parameters. Nadir is the standard survey configuration and keeps the footprint a rotated rectangle. Revisit if viewpoint becomes an experimental axis. |

## Consequences

- The loop runs and is testable with no renderer and no model.
- Anything that scores a frame plugs in behind one method. The model choice is a later decision
  in this series.
- "Cell centre in footprint" means a cell straddling the footprint edge is either fully in or
  fully out. At 2 m cells and a footprint tens of metres across, that is a small effect.
