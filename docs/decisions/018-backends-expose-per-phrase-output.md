# 018 — Backends expose per-phrase output, not only the combined map

**Status:** Accepted 2026-09-22. Refines 015 and 016.
**Area:** perception

## Context

Every backend produced one importance map per frame: the per-phrase logits were computed, reduced
by `combine`, and discarded. That is all a coverage run needs. It is not all an analysis needs: a
caller that wants to know *what the model confuses with what* needs the distribution over phrases,
not only the winner.

## Decision

A small `Backend` base class defines the split. Each backend implements `phrase_logits`, returning
the `(phrases, height, width)` stack it already computed. The base class provides:

- `importance_map` — `combine(phrase_logits(...))`, exactly the previous behaviour;
- `phrase_probabilities` — the same softmax-or-sigmoid `combine` applies, without the mission
  weighting and without the maximum, so every phrase's share is visible.

`probabilities` is factored out of `combine`, which now calls it, so there is one definition of
how logits become probabilities rather than two that can drift.

## The detector is the exception

OWLv2 has no logit field; it has boxes. Its `phrase_logits` paints each target phrase's detections
into its own map as a logit, so the sigmoid path returns the detection confidence unchanged. It
overrides `importance_map` to combine over target phrases only. Passing a detector's output
through the contrast softmax would hand every undetected cell a share of the probability mass,
which for a detector means inventing importance where it found none.

## Consequences

- No behaviour change for any existing run: the four backends were re-tested against rendered
  frames and produce identical maps.
- The scoring path gains no cost; the probabilities are computed either way.
- What this enables is a model of perception error conditioned on the scene rather than added to
  it, which is recorded privately.
