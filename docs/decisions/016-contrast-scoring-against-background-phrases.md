# 016 — Phrases are scored by contrast against background phrases

**Status:** Accepted 2026-09-21. Amends 015.
**Area:** perception

## Context

On the first textured frames the tile models' raw similarities were nearly identical for every
target phrase: each map said "this tile is aerial imagery of grass" for "cracked road", "debris"
and "vehicle" alike, with the whole range inside 0.16–0.26 of cosine. CLIPSeg's "cracked road
surface" map covered the entire asphalt road. A similarity model answers the question it is
asked; asked one phrase at a time, it reports how much the tile looks like *anything* in the
phrase's neighbourhood. Zero-shot classification with CLIP-style models has always been a
softmax over a set of candidate phrases for that reason.

## Options

| Option | Trade-off |
|---|---|
| Raw per-phrase score (015 as landed) | Simplest; dominated by image content shared by all phrases; not what the models were built for |
| Per-frame normalisation | Removes the common mode but destroys frame-to-frame comparability, which the stability study measures |
| **Softmax across target and background phrases, keep the target probabilities** | The standard zero-shot recipe; the background set is a design choice stated once; comparable across frames; applies to every backend that produces per-phrase logits |
| Learned linear probe | Needs labels; a fine-tuning question, out of scope for paper one |

## Decision

- Every backend returns per-phrase **logits** on a comparable scale: CLIPSeg's decoder logits;
  SigLIP's `logit_scale · cos + logit_bias`; `100 · cos` for CLIP-style checkpoints (their trained
  logit scale). OWLv2 is a detector and keeps target queries only.
- `build_prompts(..., contrast=True)` appends six background phrases at weight zero: grass, bare
  soil, an asphalt road, a tree, a building roof, an empty field. `combine` takes the softmax over
  all phrases per pixel or tile and keeps the strongest weighted *target* probability. With
  `contrast=False` it takes each target's own sigmoid, which is the previous behaviour.
- `scorer.contrast` (default true) in the config; `--no-contrast` on the frame-scoring command;
  the model id in a store records `:raw` when contrast is off.

## Consequences

- The background set is part of the prompt strategy and is versioned here. Changing it is a
  decision, not a tweak.
- The same set is used for the mission and the null prompt set, so the ablation between them
  isolates the language of the *targets*.
- Per-tile probabilities now sum to one across phrases; "how much of the tile is target-like" is
  bounded and comparable, which is what the affine calibration assumes.
- Numbers on the same 37 textured views before and after are in the private Q12 note.
