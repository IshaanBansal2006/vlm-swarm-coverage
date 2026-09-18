# 061 — Run visualisation is computed from the event log, never from live state

**Status:** Accepted 2026-09-18
**Area:** infrastructure

## Decision

`viz.py` reads a finished run directory and draws: each drone's shared belief (or the mean) as a
heatmap next to the answer key, the scene's footprints, each drone's camera footprint, its
track, and the Voronoi partition it was acting on. Output is an mp4 (via ffmpeg) or a summary
png of trajectories and cumulative channel bytes. Isaac captures are turned into footage by
`scripts/build_footage.py`, separately.

## Options considered

| Option | Why not |
|---|---|
| Draw live during the run | Couples the loop to matplotlib and slows it; a rendered picture that could not be regenerated from the log would be a picture of something unreproducible. |
| Only render in Isaac | The belief is invisible in a rendered scene; the heatmap is the point. |
| A dashboard | Nothing to interact with until there are sweeps. |

## Consequences

- The belief snapshot logged at each sync is what gets drawn, so the video's time resolution is
  the sync interval. Poses are interpolated to those times.
- What is drawn is exactly what was logged. A discrepancy between video and log is impossible
  by construction, which matters when a picture goes into a talk.
- No metric is drawn. Cumulative bytes are counts from the log, not a result.
