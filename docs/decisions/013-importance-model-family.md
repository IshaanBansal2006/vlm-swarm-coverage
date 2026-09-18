# 013 — The importance model: dense image-text similarity first, a detector as the comparison point

**Status:** Accepted 2026-09-18
**Area:** perception

## Context

The scorer slot needs a pretrained vision-language model that turns a nadir frame plus the mission
text into an importance value per grid cell. It has to produce a dense map, be deterministic enough
that frame-to-frame stability is measurable, and be cheap enough to score tens of thousands of
frames in the model-stability study that precedes any experiment. No model is fine-tuned in this
project.

## Options

| Family | Examples | For | Against |
|---|---|---|---|
| **Dense image-text similarity** | SigLIP 2, CLIPSeg, RemoteCLIP / GeoRSCLIP | Cheapest; deterministic; dense by construction; aerial-pretrained variants exist; stability is cleanly measurable | No reasoning; mission text reduced to embeddings; CLIP-style models are known to be unstable around unusual viewpoints, and nadir is one |
| Open-vocabulary detection / segmentation | OWLv2, Grounding DINO, SAM3 | Localized, interpretable confidences; SAM3 is what the nearest published multi-UAV system uses, so the comparison is direct | Zero-shot aerial detection is weak; sparse output needs a floor rule for empty cells |
| Generative VLM | Qwen2.5-VL / Qwen3-VL, PaliGemma 2, Florence-2 | The only family that reads mission intent as language | Seconds per frame; stochastic unless forced; structured-output failures; dense per-cell output is awkward |
| Remote-sensing VLM | GeoChat, SkySenseGPT | Domain-matched to nadir imagery | Trained on photoreal satellite imagery; the simulated scene is primitives |

## Decision

- **Primary:** dense image-text similarity. Both an aerial-pretrained variant (RemoteCLIP or
  GeoRSCLIP) and a general one (SigLIP 2) go through the stability study, so the effect of aerial
  pretraining is measured rather than assumed.
- **Comparison point:** one open-vocabulary detector (SAM3 or OWLv2), for a direct comparison
  with the published multi-UAV baseline.
- **Validity slice only:** a generative VLM, on a small frame subset.

## Why

The study's claim is about what faults do to a semantic density, not about which model makes the
best density. A weak but stable and cheap scorer serves that claim; a strong, slow, stochastic one
would make every downstream number depend on sampling. The stability study needs volume, and the
scorer interface wants density.

## The scene-realism consequence

The simulated scene is boxes and cylinders. Any model sees a red box, not a stalled vehicle. The
stability study on the simulated scene therefore measures geometry sensitivity; the same study on
real aerial imagery measures semantics. Either the scene gets realistic assets before the study
runs, or the paper states that split explicitly. This is decided with the study design, not
deferred.

## Compute

Every candidate runs on a single consumer GPU for the demo. The stability study and the live-model
validity slice run on a single A10G- or L4-class cloud instance. Sweeps run on the cache and need
no GPU.
