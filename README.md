# VLM Swarm Coverage

A simulation rig for **decentralized multi-drone coverage driven by a vision-language model**.

Classical coverage control moves a group of robots so they spend more time over regions of high
*importance* — but that importance map is conventionally hand-specified. This is a testbed for
generating it instead from what the drones actually see, plus a natural-language description of the
mission: each drone scores its own view, the drones share those scores, and a coverage controller
moves them to match the shared field.

Vision-language models are well explored for single-robot manipulation and much less so for aerial
multi-agent systems. This repository is the apparatus for working in that gap.

---

## Status

**Early — infrastructure.** The simulator bridge, project scaffolding, the world description, the
run configuration, the run-artifact layout, the importance grid, the message schema, the scorer interface with an oracle, the kinematic world,
the Lloyd controller, and the loop that closes them are in place. The demo config runs end to end
without a simulator or a model:

```bash
bash scripts/demo.sh          # = vsc-run configs/demo.toml, then viz --video --summary
```

Three drones start at the west edge, score what is under them with the oracle, and Lloyd
descent spreads them along the corridor. The video shows each drone's belief filling in beside
the answer key, with camera footprints, tracks and the Voronoi partition drawn from the log.

The Isaac scene and the bridge have run live: the scene launches headless from WSL, the loop drives
it over ROS 2, and observations come back. The nadir camera's orientation is verified from dumped
frames, and a pose-sweep mode renders frames for any list of poses (decision 063). Drones fuse
received beliefs by an age-weighted per-cell average (decision 022). No experiments have run yet.

Four model backends (CLIPSeg, SigLIP 2, RemoteCLIP, OWLv2) sit behind the scorer interface and score
synthetic frames deterministically on a laptop GPU; none has yet been run on a rendered frame or
calibrated. The near-term milestones are the first live rendered run with a model in the loop and
the model-stability study.

---

## What's here

| | |
|---|---|
| `sim/` | The Isaac Sim scene (`scenes/coverage_scene.py`): follows the loop's poses, scores or records nadir frames, or renders a pose sweep; and the Windows-side launcher |
| `config/` | DDS transport profile for the simulator ↔ WSL boundary |
| `scripts/` | `demo.sh` (run + render), environment setup, a bridge verification check, the Isaac-capture footage builder, `fetch-textures.sh` (CC0 surface textures for the scene) |
| `configs/` | Run configurations: `demo.toml` (three drones, the rendered walkthrough) and `study.toml` (five drones, the baseline sweeps derive from) |
| `src/vlm_swarm_coverage/` | The rig |
| `docs/decisions/` | Decision records — context, options, choice, consequences |

Inside the rig so far:

| Module | Role |
|---|---|
| `scene.py` | The world: survey area, road, mission, features with labels and footprints, drone start poses. Pure dataclasses so the simulator's Python can import it by path. Fixed demo layout plus a seeded generator for sweeps. |
| `config.py` | One typed, validated `RunConfig` that fully determines a run, including the message encoder, the fusion rule, the controller and the drones' initial belief. TOML in, JSON snapshot out. Unknown keys are errors. |
| `artifacts.py` | A directory per run: config snapshot, provenance (package version, git sha), and an append-only JSONL event log that every metric is computed from offline. |
| `field.py` | The importance grid over the area and the values on it, plus the ground-truth rasteriser that turns the scene's mission weights into the answer key. |
| `schemas.py` | Typed inter-drone messages (pose, belief) with a fixed binary wire format, so every message knows its own byte size. The belief message carries a per-cell observation age; sparse and quantised variants exist. Decisions 030, 032, 033. |
| `compress.py` | Sender-side encoders: the whole field, the top-k observed cells, a quantised field, or poses only. What goes on the wire is a run parameter. Decision 033. |
| `consensus.py` | Belief fusion: the no-communication endpoint, and the age-weighted per-cell average that lets drones converge where they exchange messages and disagree where they do not. Decision 022. |
| `scoring.py` | The scorer interface, the nadir camera footprint, the oracle scorer that reads the answer key, the model slot, and the pose-bucket cache keyed to one scorer. Decisions 010–014. |
| `models.py` | The models behind the slot: CLIPSeg, SigLIP 2 and RemoteCLIP made dense by tiling, OWLv2 as a detector; the nadir image-to-cell mapping and the affine score calibration. Needs the `[vlm]` extra. Decision 015. |
| `world.py` | First-order drone kinematics at fixed altitude. Decision 021. |
| `control.py` | Lloyd descent on the density-weighted Voronoi partition, decentralized (decision 020); a replay controller that retraces a logged run so motion can be held fixed (decision 023). |
| `channel.py` | The channel interface, a perfect channel, and a faulted one: per-link loss, fixed latency, byte budget per sync, seeded. Decision 031. |
| `simulation.py` | The loop: observe, score, share, fuse, control, step, log. Assembled from config. |
| `__main__.py` | `vsc-run configs/demo.toml --out runs` |
| `bridge.py` | WSL side of the simulator bridge: poses out, observations in, wall-clock pacing. Decision 060. |
| `render.py` | `python3 -m vlm_swarm_coverage.render configs/demo.toml` under the ROS interpreter, with Isaac running `sim/scenes/coverage_scene.py`. |
| `calibration.py` | Raw per-view score store, a fixed-order pose sweep, and the frame side: views to JSON for the renderer, frames and manifest back to views, `python -m vlm_swarm_coverage.calibration FRAMES CONFIG --model ...` to score them. Decisions 014, 063. |
| `sweep.py` | `python -m vlm_swarm_coverage.sweep sweep.toml --out sweeps`: expands axes over a base config into runs, executes them in parallel, indexes them, resumes by config hash. Decision 062. |
| `viz.py` | `python -m vlm_swarm_coverage.viz runs/<run> --video out.mp4 --summary out.png`: the belief heatmap over time beside the answer key, footprints, tracks and the Voronoi partition, all from the log. Decision 061. |

Design notes and experimental records are kept privately while the work is in progress. What is
here is the engineering: the simulator integration, the transport configuration, and the structure
around them.

---

## Getting started

**Requirements** — Python 3.10+ for the library and tests (3.10 is what ROS 2 Humble ships, and the
bridge runs there). The simulation additionally needs Linux
or WSL2, ROS 2 Humble, and NVIDIA Isaac Sim. Model inference needs a GPU.

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

The core install is `numpy` + `pydantic` on purpose. Model inference (`[vlm]`) and analysis
(`[analysis]`) are optional extras, so the rig stays importable and testable on a machine with no
GPU and no model weights — which is also what lets CI run it.

For the simulator boundary:

```bash
source scripts/ros-env.sh
bash scripts/verify-bridge.sh
```

The comments in `scripts/ros-env.sh` explain why the DDS transport is configured the way it is —
briefly, mirrored WSL networking gives Windows and Linux the same IP address, which causes the
default transport to select shared memory and announce locators that route back to the sender, so
both sides are pinned to a loopback-only UDP profile with explicit peer ports.

---

## Checkpoints

Each merged step is tagged `step-NN-<slug>` and stands alone: its tests pass and this README
describes what exists at that commit. `git checkout step-04-closed-loop` gives the first
end-to-end run; `step-07-viz-and-footage` the first video.

---

## Lineage

The simulator bridge, multi-drone scene pattern, coverage-planner interface and typed-schema
discipline come from
[**drone-swarm-autonomy**](https://github.com/IshaanBansal2006/drone-swarm-autonomy), a four-layer
autonomy platform built and concluded before this project. Its
[bridge log](https://github.com/IshaanBansal2006/drone-swarm-autonomy/blob/master/docs/isaac-wsl-ros2-bridge.md)
documents the twelve issues behind the DDS configuration vendored here.

Built by [Ishaan Bansal](https://github.com/IshaanBansal2006).
