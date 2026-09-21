# 015 — The model slot: three backend families, one dense-scoring path

**Status:** Accepted 2026-09-20. Implements 013.
**Area:** perception

## Context

Decision 013 chose the families (dense image-text similarity first, an open-vocabulary detector
as the comparison point). The slot needs a concrete way to turn a nadir frame and a mission into
one value per grid cell, for models that are not natively dense, without a different scoring path
per model.

## Options

| Question | Options | Choice |
|---|---|---|
| How does a similarity model become dense? | (a) patch-token similarity with last-layer surgery (MaskCLIP-style); (b) **tile the frame, embed each tile as an image**; (c) sliding windows with overlap | **(b)**: model-agnostic (works unchanged for SigLIP 2, CLIP, RemoteCLIP, GeoRSCLIP), one batched forward pass per frame, deterministic, and at 6×8 tiles on a 12 m nadir frame a tile is roughly one grid cell |
| How do phrases combine? | sum of weighted scores; **max of weighted scores**; softmax over phrases | **max**, because the answer key takes each cell's most important feature; a sum would rank a cell with two weak matches above one with a strong match |
| Map → cells | nearest pixel at the cell centre; **mean over the cell's pixel box** (integral image); bilinear | **mean over the box**: a cell is an area, and a nearest sample would flicker as the drone moves |
| Raw score → importance | per-frame min–max; **affine per model from a calibration set**; none | **affine**: per-frame normalisation destroys frame-to-frame consistency, and the stability study exists to measure exactly that |
| Prompt text | mission text verbatim; **one phrase per weighted label** plus a **null** set | phrases match how similarity models are trained; the null set is the ablation rung |

## Decision

`models.py`:

- `ClipSegBackend` — native dense: one logit map per phrase, sigmoid.
- `TileBackend` (transformers: SigLIP 2) and `OpenClipTileBackend` (open_clip: RemoteCLIP) — the
  frame cut into 6×8 tiles, each embedded, cosine against each phrase; SigLIP models return the
  sigmoid of their scaled, biased cosine (a probability), CLIP models the cosine.
- `Owlv2Backend` — boxes above a threshold painted into a map at frame resolution.
- `combine`: the strongest weighted phrase per pixel.
- `map_to_cells`: for each footprint cell, the mean of the map over the pixel box of the cell's
  four corners, via an integral image. The camera model is the one `footprint_cells` uses: image x
  along the heading, image up to the drone's left.
- `ScoreCalibration(lo, hi, floor)`: `floor + (1 − floor)·clip((s − lo)/(hi − lo))`, identity
  by default, fitted once per model and loaded from JSON.
- `VLMScorer(model, grid, mission, prompt_set, calibration, device)` delegates to the above;
  `scorer.prompt_set`, `scorer.calibration` and `scorer.device` are config fields.
- Score records carry a `tag` naming the protocol phase they belong to.
- Model ids: `clipseg`, `siglip2`, `siglip2-so400m`, `remoteclip-b32`, `remoteclip-l14`, `owlv2`.

## Consequences

- All four backends load and score on the laptop's 8 GB GPU; a synthetic frame scores identically
  on repeated calls for every backend (the opt-in test `VSC_MODEL_TESTS=1`).
- Tiling trades spatial resolution for generality: a 6×8 map on a 16.8 × 12.6 m footprint is
  2.1 × 1.6 m per tile, close to the 2 m cell. Finer tiles cost a linear factor in inference.
- The importance map's units differ per backend until calibrated; nothing downstream may compare
  raw values across models.
- `open_clip_torch` joins the `[vlm]` extra. CI still installs neither torch nor the models; the
  backend tests skip without the opt-in flag.
