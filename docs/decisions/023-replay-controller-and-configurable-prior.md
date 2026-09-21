# 023 — A replay controller, and a configurable initial belief

**Status:** Accepted 2026-09-20
**Area:** coverage control and the decentralized layer

## Context

Two things a study of this loop needs that a coverage controller cannot give it. First, a way
to hold motion fixed while everything downstream of motion (channel, fusion, scoring) changes,
so an effect can be attributed to the loop rather than to the components. Second, a way to
choose what the drones believe before they have looked, since the start-up belief is an input
to the controller from the first step.

## Options

**Fixing motion.**

| Option | Trade-off |
|---|---|
| Scripted paths (lawnmower, fixed waypoints) | Simple; but a different trajectory from the closed-loop run, so any comparison confounds path shape with the loop |
| **Replay a logged run's trajectory** | Identical motion to a reference run by construction; needs a source run and a matching `dt` and drone count |
| Freeze the controller's density at the prior | Still closed-loop through peers' positions; not actually open |

**Initial belief.**

| Option | Trade-off |
|---|---|
| Floor everywhere (the only option before this record) | Neutral start; nothing to say about what a wrong start does |
| **Floor · truth · shifted truth · another layout's truth**, each optionally stamped as observed at a time | One switch covers "no prior", "perfect prior", "plausibly wrong prior" and "wrong prior that gets shared" |
| Arbitrary array from a file | Maximal, but the study only needs the four shapes above and a file is one more artifact to version |

## Decision

- `ReplayController(trajectories, dt)` commands, at time t, the velocity from the current
  position to the source run's position at t + dt; beyond the source's end it holds.
  `from_run(run_dir)` reads a run's `pose` events. `controller.kind = "replay"` with
  `controller.replay_run` selects it; the builder refuses a source with a different `dt` or
  drone count. The controller protocol gains `t` so a controller can be a function of time.
- `[prior]` with `kind ∈ {floor, truth, shifted_truth, other_scene}`, `shift_m`, `scene_seed`
  and `observed_at`. Unset `observed_at` leaves the prior unobserved: it shapes the drone's own
  motion but is never transmitted as evidence. Set, the prior carries that stamp and is shared.
- Pose events are logged to 1e-5 m (was 1e-3) so a replayed run retraces its source to within
  the log's own rounding.

## Consequences

- A replayed run is deterministic and identical in motion to its source; its log differs only in
  what the drones believed and sent. Comparing the two isolates the loop.
- A replay is only as long as its source; a longer `duration_s` holds the drones at the end.
- `shifted_truth` fills the uncovered strip with the floor rather than wrapping, so a shift never
  invents importance on the far edge.
