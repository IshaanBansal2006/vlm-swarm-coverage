# 065 — What a nadir camera keys on: vehicle detail, a smooth road, a rutted damaged patch

**Status:** Accepted 2026-09-21. Extends 064.
**Area:** infrastructure (simulator scene)

## Context

With textured surfaces (064) the models became stable frame to frame, and per-phrase maps on
the rendered frames showed three specific failures: no model saw the flat red box as a vehicle;
"cracked road surface" lit the entire road because the road's own asphalt texture is worn and
cracked; and the damaged-road texture was another asphalt, so the target was not distinguishable
from the road it sits on. Two of the three targets were therefore invisible to the scorer by
construction of the scene, not by any property of the models.

## Decision

- **Vehicles** gain what a top-down view of a car has: a roof panel a shade darker than the body,
  dark glass at both ends of the cabin (windscreen and rear window), thin dark seams where bonnet
  and boot meet the body, and side mirrors. Still primitives; still no asset import.
- **Road**: a smooth tarmac texture (`asphalt_pit_lane`, 2 m) instead of worn asphalt.
- **Damaged road**: rutted wet mud (`aerial_mud_1`, 8 m physical size, so the 6 m patch shows one
  set of ruts) instead of a second asphalt.
- **Tree canopy**: forest-floor foliage (`forrest_ground_01`) instead of grass.
- The fetch script downloads these; the earlier textures stay available for comparison.

## Options considered

| Option | Why not |
|---|---|
| Import photoreal car and debris assets | The right long-term move; needs sourcing, conversion and licence checks, and the study can start once the primitives read as their class. Revisit if the calibration says the props are the limit. |
| Real orthophoto ground | Rejected in 064 for the answer-key reason. |
| Change the phrases instead of the scene | Phrases are the mission's words; bending them to the scene would calibrate the model to the rig. |

## Consequences

- The same 37 views were re-rendered and re-scored; before/after numbers are in the private
  calibration notes, not here.
- Scene changes after this point re-render the calibration study and the cache; the scene is
  therefore frozen for the study once the gate passes on it.
