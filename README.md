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

**Early — infrastructure.** The simulator bridge and project scaffolding are in place. No
experiments have run yet.

The near-term milestone is the pipeline coming online end to end on a single drone: rendered camera
frames out of the simulator, a model scoring them, and a coverage controller closing the loop on
the result.

---

## What's here

| | |
|---|---|
| `sim/` | Isaac Sim scenes and the Windows-side launcher |
| `config/` | DDS transport profile for the simulator ↔ WSL boundary |
| `scripts/` | Environment setup and a bridge verification check |
| `src/vlm_swarm_coverage/` | The rig |
| `docs/decisions/` | Decision records — context, options, choice, consequences |

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
