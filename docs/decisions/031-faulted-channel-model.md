# 031 — Channel model: per-link loss, fixed latency, byte budget per sync

**Status:** Accepted 2026-09-18
**Area:** communication model

## Context

The rig needs a communication model that is faithful enough to stand in for a lossy radio link,
simple enough that every parameter is a clean experimental axis, and reproducible from a seed.

## Decision

`FaultedChannel` with three parameters, plus the loop's sync interval:

| Parameter | Model | Alternative rejected |
|---|---|---|
| `drop_rate` | Bernoulli loss decided independently per (message, receiver) pair | Per-message loss (all receivers or none) models a failed transmitter, not a lossy medium. Bursty / Gilbert-Elliott loss adds a second parameter; revisit if burstiness becomes a question. |
| `latency_s` | Fixed delay on every delivered message | Jitter adds a distribution to defend. Fixed latency isolates the staleness effect. |
| `bytes_per_sync` | Budget per sender, reset each time the sender sends at a new sim time; messages admitted in send order until one does not fit, which is dropped whole | Truncation would deliver a message the schema cannot decode. Priority queues would be a protocol design decision, not a channel property. Poses are sent before beliefs in the loop, so a small budget delivers positions first. |
| `sync_interval_s` (loop) | How often each drone shares | Event-triggered sharing is a protocol choice for later. |

Randomness comes from one `numpy` generator seeded from the run seed, isolated from global
state. Zero parameters make the channel perfect, so one class serves every run.

## Consequences

- Every drop is logged with sender, receiver, time, reason and size, and the end-of-run event
  carries the channel counters, so channel behaviour is reconstructible offline.
- A byte budget below one belief message means beliefs never cross and only poses do. That is a
  legitimate, deliberately reachable operating point.
- Loss per link means with N drones a broadcast is N−1 independent trials; the expected number of
  peers reached is (N−1)(1−p).
