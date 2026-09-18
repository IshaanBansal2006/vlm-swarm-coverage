# 060 — Simulator integration: the loop owns motion, Isaac renders and may score

**Status:** Accepted 2026-09-18. **Live status: written against the predecessor's proven capture
loop; not yet run against Isaac at this tag.**
**Area:** infrastructure

## Context

Isaac Sim runs on Windows; the swarm loop runs on WSL under the ROS 2 Humble interpreter
(Python 3.10). The bridge between them is a loopback-only Fast DDS profile that drops large
messages. Camera frames are large; importance observations are small.

## Options

| Option | Trade-off |
|---|---|
| Ship frames to WSL (image topics) | Cleanest ROS design; needs the transport's max message size raised and is unproven on this bridge; frames are the largest thing in the system. |
| Frames to disk, WSL polls | Proven at 15 fps for one camera; N nadir cameras plus a chase camera multiply the disk traffic and the latency. |
| **Score on the renderer's host, ship observations** | Only small messages cross. The scorer interface is host-agnostic so the same class runs there. Ties the model stack to Isaac's bundled Python and shares its GPU. |
| Isaac integrates motion (physics + flight controller) | Realistic and slow; motion is not the variable under study (decision 021). |

## Decision

- **The loop owns motion.** Isaac never moves a drone; it applies the poses the loop publishes.
  One run is therefore identical with or without a renderer, and the renderer-free run is the
  reference.
- **Poses cross as JSON strings** on `/drone_poses`, the schema the predecessor's renderer
  already consumes. **Observations cross as the binary `ObservationMessage`** on `/importance`
  (`UInt8MultiArray`), tens of bytes per cell, far under the bridge's size ceiling.
- **Scoring can run on either host.** The scene script hosts the oracle today and the model
  slot when it exists; `--remote-scorer` on the WSL side takes those observations in place of a
  local scorer. Without it, the loop scores locally and Isaac only renders.
- **The loop paces to wall clock** when a renderer is attached, so the rendered run plays at
  real speed and the frame timestamps line up with the log.
- **The package supports Python 3.10**, because that is the interpreter with `rclpy`.

## Consequences

- A rendered demo needs no model: oracle on WSL, Isaac renders, frames on disk from the chase
  camera. The model path is the same wiring with `--score vlm`.
- Rendering never changes a result. If a rendered run's log differs from the same config run
  headless, that is a bug in the bridge, not a property of the study.
- Isaac's Python needs `pydantic` installed once for the schema module.
- The nadir camera FOV is derived from the configured FOV via the aperture, so the rendered
  footprint matches the footprint the oracle scores. That equality is asserted by construction,
  not yet by measurement.
