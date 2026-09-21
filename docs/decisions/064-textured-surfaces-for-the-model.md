# 064 — Textured surfaces: what the model looks at is no longer flat colour

**Status:** Accepted 2026-09-21. Addresses the scene-realism consequence of 013.
**Area:** infrastructure (simulator scene)

## Context

Every surface in the scene was a flat colour. A model scoring a nadir frame therefore saw a red
box on a grey band on a green plane. On the first 37 rendered views, similarity models still
found some signal in that (a rank correlation with the answer key of 0.4–0.6) and the detector
found none. The study's perception axis is calibrated to whatever the model sees, so the scene has
to look like ground before the calibration runs.

## Options

| Option | Trade-off |
|---|---|
| Keep flat colours; state the split in the paper | Cheapest; the perception axis then measures response to colour and shape, not to imagery |
| Real orthophoto draped on the ground | Most realistic ground; anything in the photo (cars, damage) is importance the answer key does not know about, so the photo must be benign, and a benign photo of a road corridor is hard to find |
| **CC0 physically-sized textures projected in world space onto the existing surfaces** | Ground, road, damaged road, debris and canopy look like their materials at the right scale; the geometry and the answer key are unchanged; no UVs needed; five downloads |
| Photoreal assets (vehicle models, buildings) | Better still for the props; needs asset sourcing and conversion; deferred until the textured scene's calibration says the props are the limit |

## Decision

`scripts/fetch-textures.sh` downloads six Poly Haven CC0 diffuse maps (2k JPEG) into
`sim/assets/textures/` (ignored by git). The scene builds one OmniPBR material per surface kind
with `project_uvw` in world space and `texture_scale = 1 / physical size`, so grass repeats every
2 m, asphalt every 2.1 m, damaged asphalt every 2.2 m, rubble every 4 m. Vehicles, the shed, the
quads and the tree trunk keep their flat colours. `--textures none` restores the flat scene;
`--textures DIR` points at a copy on the Windows side, because the RTX texture loader cannot read
the WSL UNC path the scene script itself is loaded from.

## Consequences

- The oracle is unaffected: it reads the scene description, not pixels.
- The same 37-view sweep is rescored on the textured scene so the change is measured, not
  assumed. The full calibration (private protocol) runs on the textured scene.
- Texture tiling is visible from 12 m; a model that keys on repetition would be fooled by the
  scene, not by the world. That is stated in the paper's limitations if the calibration shows it.
- Attribution: Poly Haven textures are CC0 and need none; the script names its source anyway.
