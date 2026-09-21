# 063 — The nadir camera as rendered, and rendering frames by pose sweep

**Status:** Accepted 2026-09-21. Amends 060's "not yet run live".
**Area:** infrastructure

## Context

The first live run (2026-09-20) proved the bridge: the scene launched headless from WSL through
interop over the UNC path, came up in 55 s, rendered the chase camera at 30 fps, took the loop's
poses and returned the Isaac-hosted scorer's observations. It also exposed what the oracle could
not: the nadir cameras were never checked, and the calibration study needs frames from a list
of poses without any loop running.

## What was wrong, and the fix

`isaacsim.core.experimental.objects.Camera` is a plain transform prim: a USD camera looks down
its local −Z with +Y as image up, and the identity orientation is therefore already nadir with
image x along world x. The fixed quaternion `[0, 1, 0, 0]` rotated that by 180° about x and
looked *up*, and did not turn with the drone. The camera is now oriented with the drone's own yaw
quaternion, 0.1 m below the body so the quad's geometry sits behind the lens. Dumped frames
confirm the convention `scoring.footprint_cells` and `models.ground_to_pixel` assume: at yaw 0
the road along world x is horizontal with +y up; at yaw 90° it is vertical; a camera above the
stalled vehicle centres it.

## Options for producing calibration frames

| Option | Trade-off |
|---|---|
| Run the loop with a scripted controller and record | Motion-coupled; pose set is whatever the controller does; slow (real-time paced) |
| Ship frames over the bridge to WSL | Frames are the largest thing in the system and the loopback profile drops large messages (060) |
| **Pose-sweep mode: the scene places one camera at each listed pose, renders, saves a PNG and a manifest line** | No ROS, no loop, no model on the Isaac host; 37 views in 28 s including start-up; frames scored later by any scorer on any host |

## Decision

- `--sweep views.json --record DIR --headless`: one nadir camera, one PNG per view under
  `DIR/sweep/`, `DIR/manifest.jsonl` with pose, tag, group, camera parameters and size. ROS is
  skipped. After each camera move the script settles and forces a renderer step
  (`rep.orchestrator.step`), because the annotator otherwise returns the previous pose's frame.
- `--record-nadir`: in the live loop, every scored nadir frame is saved as JPEG under
  `DIR/nadir/` with `nadir_manifest.jsonl`.
- `--no-ros`: launch without the bridge (implied by `--sweep`).
- `calibration.py` gains the other half: `views_to_json` writes the renderer's input,
  `load_manifest`/`frame_view` read frames back as views, `score_frames` scores a manifest into
  the score store with the manifest's tags, and `python -m vlm_swarm_coverage.calibration
  FRAMES CONFIG --model {oracle,<model id>} --out STORE` is the command.
- The default ground plane (a grid texture) is replaced by a plain slab far beyond the area, so
  what a model sees outside the painted area is ground colour, not a grid.

## Consequences

- The calibration study is: generate views on WSL, render on Windows, score on WSL, analyse on
  WSL. Each step is a file; no step needs the others running.
- The cache fill for a model is the same pipeline over the position × altitude buckets.
- A frame from the live loop lags the pose by one renderer update (about 33 ms), which is
  negligible for a 1 Hz scorer and is stated here rather than fixed.
