# 017 — Score stores hold raw scores; the calibration is applied when a cache is built

**Status:** Accepted 2026-09-21. Amends 014 and 015.
**Area:** perception

## Context

Decision 014 made the score store a record of what a model said, raw. The first cache-fill
pipeline applied the affine calibration inside the scorer at scoring time, so the store held
calibrated values and a change of calibration meant re-running the model on every frame.

## Decision

`calibration score` writes raw scores. `calibration cache --calibration FILE` applies the affine
map while turning a store into a pose-bucket cache. One raw store therefore yields a cache for any
calibration, and a calibration can be revised without a GPU.

The calibration itself is anchored on the two classes of the calibration set: the median
background cell maps to the floor and the median target cell to one. The earlier choice, the 5th
and 95th percentiles of all scores, left the median background cell well above the floor and made
every belief diffuse, which a coverage controller reads as "everything matters a little".

## Consequences

- Caches carry the model id and are still refused by a scorer with a different id; the
  calibration used is recorded beside the cache in the study's runbook.
- A model whose background median equals its target median cannot be calibrated this way; that
  is a gate failure, not a calibration problem.
