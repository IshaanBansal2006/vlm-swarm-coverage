# 032 — The belief message carries a per-cell observation age

**Status:** Accepted 2026-09-20. Amends 030.
**Area:** communication model

## Context

Decision 030 chose a dense belief message: every cell, float32. A receiver could not tell a cell
the sender observed at the floor from a cell the sender never saw and still holds at its prior.
Any fusion rule that averages would therefore pull well-observed cells toward every peer's prior,
and no rule could weigh a value by how stale it is.

## Options

| Addition | Bytes per cell (30×20 grid) | What it enables |
|---|---|---|
| Nothing | 4 (2424 B) | Only rules that ignore provenance |
| 1-bit observed mask | 4.125 (2499 B) | Exclude priors; no staleness |
| **uint16 age, 0.1 s units, relative to message time** | 6 (3624 B) | Exclude priors *and* weigh by staleness; self-contained (no clock assumption beyond the header's `t`); exact for the rig's 0.1 s step |
| float32 absolute stamp | 8 (4824 B) | Same as above at twice the cost, and requires clock alignment |

## Decision

Append one uint16 per cell after the values: the age of the observation behind the value in
tenths of a second before the message's `t`, with 65535 meaning never observed. The receiver
reconstructs stamps on its own clock as `t − age·0.1`. Values that were never observed are still
transmitted (the layout stays dense and fixed-size) but are marked as such.

## Consequences

- A belief message on the study grid is 3624 bytes instead of 2424. Bytes are an experimental
  variable; sweep levels are chosen relative to this size.
- The local update (`apply_observation`) stamps the cells it writes; a belief is constructed with
  `ImportanceField.prior` so unobserved cells start as NaN.
- The `belief` log event records the ages as sent, so provenance is available offline.
- Ages are quantised to 0.1 s and saturate at 6553 s; both are far outside anything a run does,
  and both are stated here so a change to `dt` or run length re-examines them.
