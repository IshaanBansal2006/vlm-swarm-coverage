# 069 — `sim.seed` becomes an independent replicate via seeded start jitter

**Status:** Adopted 2026-09-22. Public (infrastructure). Default off.

## What was found

A sweep whose arms are all cached scorers on a perfect channel has nothing stochastic in it. The
swarm's initial positions come from the scene (a fixed line along the west edge), the channel is
the only consumer of `sim.seed`, and a perfect channel never draws. Measured on such a sweep, 20 of
21 (arm, layout) groups produced **bit-identical** results across all five seeds.

The consequence is not a wrong number, it is a wrong interval: a bootstrap over the seed axis was
resampling five copies of one value, so the reported confidence interval described nothing. The
effective sample size was the number of layouts.

The first study is unaffected and this was checked rather than assumed: in its two surface sweeps
only 4–5 % of multi-seed groups are seed-identical, with a median relative spread of 26–29 %. The
25 % identical groups in the compression, prior, scale and ablation sweeps are exactly the
no-fault cells, which are deterministic by construction and correctly so. Its intervals stand.

## The decision

Add `swarm.start_jitter_m` (default `0.0`). When positive, each drone's start is offset by a
draw uniform over a disc of that radius, from `sim.seed`, clipped to the area and horizontal only.

This is the right independent replicate for a coverage experiment. The world is held fixed and the
swarm's initial condition is resampled, which is precisely the variation the locational cost is
supposed to be robust to — as opposed to resampling the world, which changes the question.

## Options

| Option | Verdict |
|---|---|
| **Seeded start jitter, opt-in** | **Chosen.** Cheap, is the variation the metric should be robust to, and default `0.0` reproduces every earlier run bit for bit. |
| Make `sim.seed` perturb the scene | Changes the world, so it confounds replicate variation with layout variation — the two things the transfer design most needs to keep apart. |
| Add process noise to the dynamics | Defensible, but it would change what the controller is being measured on, and the study's claims are about perception, not control robustness. |
| Report only per-layout results and drop the seed axis | Honest but wasteful: it accepts an effective n of 3 when independent replicates are nearly free. |
| Jitter by default | Rejected. It would silently change every existing config's meaning and break the frozen study's reproducibility, which is a hard requirement here. |

## The limit worth stating

Start jitter makes seeds independent *replicates of an initial condition*. It does not make them
independent samples of a scene, so it does not licence treating runs across one layout as
independent evidence about layouts. A design that wants to generalise over scenes still needs more
scenes; this only stops an interval from being computed over copies.
