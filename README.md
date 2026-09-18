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
vsc-run configs/demo.toml --out runs
```

No fusion rule exists yet: each drone's belief is only what it has seen. Nothing has been rendered. No experiments have run yet.

The near-term milestones are the simulator rendering a run and the model behind the scorer
interface.

---

## What's here

| | |
|---|---|
| `sim/` | Isaac Sim scenes and the Windows-side launcher |
| `config/` | DDS transport profile for the simulator ↔ WSL boundary |
| `scripts/` | Environment setup and a bridge verification check |
| `configs/` | Run configurations (`demo.toml` is the reference run) |
| `src/vlm_swarm_coverage/` | The rig |
| `docs/decisions/` | Decision records — context, options, choice, consequences |

Inside the rig so far:

| Module | Role |
|---|---|
| `scene.py` | The world: survey area, road, mission, features with labels and footprints, drone start poses. Pure dataclasses so the simulator's Python can import it by path. Fixed demo layout plus a seeded generator for sweeps. |
| `config.py` | One typed, validated `RunConfig` that fully determines a run. TOML in, JSON snapshot out. Unknown keys are errors. |
| `artifacts.py` | A directory per run: config snapshot, provenance (package version, git sha), and an append-only JSONL event log that every metric is computed from offline. |
| `field.py` | The importance grid over the area and the values on it, plus the ground-truth rasteriser that turns the scene's mission weights into the answer key. |
| `schemas.py` | Typed inter-drone messages (pose, belief) with a fixed binary wire format, so every message knows its own byte size. Decision 030. |
| `consensus.py` | The belief-fusion interface. Only the no-fusion endpoint exists so far. |
| `scoring.py` | The scorer interface, the nadir camera footprint, the oracle scorer that reads the answer key, the model slot, and the pose-bucket cache. Decisions 010–012. |
| `world.py` | First-order drone kinematics at fixed altitude. Decision 021. |
| `control.py` | Lloyd descent on the density-weighted Voronoi partition, decentralized. Decision 020. |
| `channel.py` | The channel interface, a perfect channel, and a faulted one: per-link loss, fixed latency, byte budget per sync, seeded. Decision 031. |
| `simulation.py` | The loop: observe, score, share, fuse, control, step, log. Assembled from config. |
| `__main__.py` | `vsc-run configs/demo.toml --out runs` |

Design notes and experimental records are kept privately while the work is in progress. What is
here is the engineering: the simulator integration, the transport configuration, and the structure
around them.

---

## Getting started

**Requirements** — Python 3.11+ for the library and tests. The simulation additionally needs Linux
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

## Lineage

The simulator bridge, multi-drone scene pattern, coverage-planner interface and typed-schema
discipline come from
[**drone-swarm-autonomy**](https://github.com/IshaanBansal2006/drone-swarm-autonomy), a four-layer
autonomy platform built and concluded before this project. Its
[bridge log](https://github.com/IshaanBansal2006/drone-swarm-autonomy/blob/master/docs/isaac-wsl-ros2-bridge.md)
documents the twelve issues behind the DDS configuration vendored here.

Built by [Ishaan Bansal](https://github.com/IshaanBansal2006).
