# 030 — Message schema: a dense belief grid

**Status:** Accepted 2026-09-17
**Area:** communication model

## Context

Drones exchange importance information over a channel whose byte budget is an experimental
variable. The wire format therefore decides what a byte cap actually constrains, and it has to be
fixed before the channel, the fusion rule, or any bandwidth analysis is written.

All options share the same grid of cells over the survey area (`field.py`); they differ in what a
message carries.

## Options

| Option | Payload | Trade-off |
|---|---|---|
| **Dense belief grid** | every cell, float32, row-major | Fixed size per message, no framing overhead, trivially fusable cell-for-cell. Carries the sender's *belief*, so an unobserved cell is indistinguishable from one observed at the prior. Largest per message. |
| Sparse observation list | (cell index, value) for cells observed since the last sync | Size scales with what was seen, so a byte cap bites on observation volume rather than grid size. Needs per-cell framing. A receiver only learns about cells the sender looked at. |
| Sparse with provenance | as above plus per-cell timestamp and sender | Only format that supports staleness- or trust-weighted fusion without a schema change. Most bytes per cell. |

## Decision

Dense belief grid. `BeliefMessage` carries the whole field; `PoseMessage` carries position only.
Both encode to a fixed little-endian header plus payload and report their own `nbytes`.

## Consequences

- A message is a belief, not an observation. Fusion operates belief-to-belief.
- Any future rule that must distinguish "never seen" from "seen, low" needs an observation mask
  added to the message. That is an additive change to the schema, recorded here so it is not a
  surprise later.
- Payload size is a property of the grid, not of the run: a 30 × 20 grid is 24 + 2400 bytes per
  belief message, every time. Any bandwidth analysis has to account for that constancy.
- JSON is for logs only. Transport bytes are the binary encoding, so the byte count is a property
  of the message and not of the encoder.
