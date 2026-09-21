# 066 — Layouts: the road's offset is drawn per seed, and a study runs on several layouts

**Status:** Accepted 2026-09-21. Amends `random_scene`.
**Area:** infrastructure (scene)

## Context

The fixed demo layout and every random layout put the road on the area's centre line, so the
mission's targets always lay along the middle of the map. A decentralised Lloyd controller that
loses contact with its peers has a Voronoi region equal to the whole area, and under a uniform
prior its centroid is the map's centre; drones that hear nothing therefore converge on the middle.
On a centred layout that accident covers the targets, and a pre-sweep check showed runs with 90 %
packet loss covering more target importance than runs with a perfect link. Any conclusion drawn
on such a scene would be about where the road is, not about the link or the model.

## Options

| Option | Trade-off |
|---|---|
| Keep one centred layout and note it | The artefact stays in every number |
| One off-centre layout | Replaces "centre is good" with "centre is bad"; still one layout |
| **Road offset drawn per layout seed from the middle 40 % of the height; a study averages over several layouts** | No fixed position is good on every layout; costs one render and cache per layout |
| Randomise the drone starts as well | Left as it was: the start line is part of the task, and a study can add it as an axis later |

## Decision

`random_scene` draws the road's centre from `height · U(0.3, 0.7)`; distractors are placed in the
room that leaves on each side. A study declares its layouts with the sweep axis `*` (decision
062's multi-key level), one level per layout carrying the scene seed and the cache rendered for
it; the study harness keeps that axis in its clean stage so every run has a reference on its own
layout. The pose-sweep renderer takes the layout through `--config`.

## Consequences

- Four layouts times five seeds replace twenty seeds on one layout; the run count is unchanged.
- The calibration study stays on one layout: it measures the model, not the map.
- The demo layout is unchanged and still centred; it is a demo.
