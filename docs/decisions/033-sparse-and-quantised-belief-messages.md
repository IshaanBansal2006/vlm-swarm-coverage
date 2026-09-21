# 033 — Sparse and quantised belief messages, and the poses-only option

**Status:** Accepted 2026-09-20. Extends 030 and 032.
**Area:** communication model

## Context

The dense belief message costs 6 bytes per cell (3624 bytes on the study grid). Bytes on the
channel are an experimental variable, and the only way to move that variable at the sender is to
send less. The channel's byte budget (031) drops whole messages that do not fit; it cannot make a
message smaller. That is a sender's job.

## Options

| Encoder | Bytes on the study grid | What it keeps | What it loses |
|---|---|---|---|
| Dense (030 + 032) | 3624 | Everything | — |
| **Top-k observed cells** | 28 + 8k | The peaks: the k highest-valued cells the sender has actually observed, with exact value and age | Everything else, including "I looked there and saw nothing" |
| **Quantised dense** | 28 + 2·cells = 1228 | The shape of the whole field; which cells were observed | Value precision (1/255 of the per-message maximum) and age resolution (whole seconds, 254 s cap) |
| Basis-sparse (DCT coefficients) | header + 6m | Smooth structure | Per-cell provenance; needs a separate mask; not implemented at this tag |
| **Poses only** | 0 | Where the drone is | What it believes |

Each encoder is a claim about which aspect of an importance field is load-bearing for coverage
(peaks, shape, or nothing), which is why more than one exists.

## Decision

- `SparseBeliefMessage`: uint16 row-major cell index, float32 value, uint16 age per cell;
  the sender's `top_k` picks the k highest-valued *observed* cells, ties by index, fewer if fewer
  are observed. Unsent cells decode as never observed, so the fusion rule ignores them exactly as
  it ignores a peer's prior.
- `QuantisedBeliefMessage`: per-message float32 scale (the field's maximum), then one uint8
  value level and one uint8 age in whole seconds (255 = never) per cell.
- `[message] kind = dense | topk | quantised | none`, with `k` for `topk`. `none` sends poses
  only; the loop still runs, fuses nothing, and logs zero belief bytes.
- The encoder is chosen once per run and applied at every sync. The `send` event records the
  belief bytes separately from the pose bytes.

## Consequences

- All three belief messages decode to the same `ImportanceField` with stamps, so `consensus.py`
  and the log are unchanged by the choice of encoder.
- A top-k message at k = 0 is a 28-byte header; at k = 1 it is one cell, 36 bytes, roughly the
  size of a pose message. Whether that limit behaves like the poses-only case is a measurement,
  not an assumption.
- Quantisation is relative to each message's own maximum, so a field with one hot cell loses
  precision everywhere else. That is a property of the encoder and is stated wherever it is used.
- The dense message stays the default so earlier tags and configs mean what they meant.
