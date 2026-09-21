"""The loop. Observe, score, share, fuse, control, step, log — every control interval.

`Simulation.run` is the only place the components meet. Each is passed in as an object of its
interface, so a run with an oracle scorer and a perfect channel and a run with a model and a
faulted channel are the same code path with different arguments. `build` maps a `RunConfig` to
those objects for the kinds this repository implements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.artifacts import RunDir
from vlm_swarm_coverage.channel import FaultedChannel
from vlm_swarm_coverage.compress import build_encoder
from vlm_swarm_coverage.consensus import AgeWeightedAverage, IgnoreMessages
from vlm_swarm_coverage.control import HoldController, LloydController, ReplayController
from vlm_swarm_coverage.field import Grid, ImportanceField, rasterize_ground_truth
from vlm_swarm_coverage.scene import default_scene, random_scene
from vlm_swarm_coverage.schemas import PoseMessage, encode_ages
from vlm_swarm_coverage.scoring import (
    CachedScorer,
    OracleScorer,
    View,
    VLMScorer,
    apply_observation,
)
from vlm_swarm_coverage.world import KinematicWorld

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from vlm_swarm_coverage.artifacts import EventLog
    from vlm_swarm_coverage.channel import Channel
    from vlm_swarm_coverage.compress import BeliefEncoder
    from vlm_swarm_coverage.config import RunConfig
    from vlm_swarm_coverage.consensus import BeliefFusion
    from vlm_swarm_coverage.control import CoverageController
    from vlm_swarm_coverage.scene import Scene
    from vlm_swarm_coverage.schemas import AnyBeliefMessage
    from vlm_swarm_coverage.scoring import ImportanceScorer

log = logging.getLogger(__name__)


@dataclass
class Agent:
    """What one drone knows: its belief, the last pose it heard from each peer, message counters."""

    drone_id: int
    belief: ImportanceField
    peers: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    seq: int = 0
    inbox: list[AnyBeliefMessage] = field(default_factory=list)


@dataclass
class Simulation:
    config: RunConfig
    scene: Scene
    grid: Grid
    world: KinematicWorld
    scorer: ImportanceScorer
    fusion: BeliefFusion
    controller: CoverageController
    channel: Channel
    agents: list[Agent]
    encoder: BeliefEncoder
    on_step: Callable[[Simulation, float], None] | None = None

    def run(self, run_dir: RunDir) -> RunDir:
        cfg = self.config
        score_every = max(1, int(round(cfg.scorer.period_s / cfg.sim.dt)))
        sync_every = max(1, int(round(cfg.channel.sync_interval_s / cfg.sim.dt)))
        with run_dir.events() as ev:
            ev.write("start", 0.0, n_drones=len(self.agents), grid=list(self.grid.shape))
            for step in range(cfg.sim.n_steps):
                t = self.world.t
                if step % score_every == 0:
                    self._observe(t, ev)
                if step % sync_every == 0:
                    self._share(t, ev)
                self._receive(t, ev)
                self.world.step(self._commands())
                self._log_poses(t, ev)
                if self.on_step is not None:
                    self.on_step(self, self.world.t)
            ev.write("end", self.world.t, steps=cfg.sim.n_steps, channel=dict(getattr(self.channel, "stats", {})))
        log.info("run complete: %s (%d steps)", run_dir.path, cfg.sim.n_steps)
        return run_dir

    def _observe(self, t: float, ev: EventLog) -> None:
        sw = self.config.swarm
        for agent, d in zip(self.agents, self.world.drones, strict=True):
            view = View(agent.drone_id, t, float(d.position[0]), float(d.position[1]), d.z, d.yaw,
                        sw.camera_fov_deg, sw.camera_aspect, None, self.scene.mission.text)
            obs = self.scorer.score(view)
            apply_observation(agent.belief, obs)
            ev.write("score", t, drone=agent.drone_id, n_cells=int(len(obs.cells)), mass=float(obs.values.sum()))

    def _share(self, t: float, ev: EventLog) -> None:
        for agent, d in zip(self.agents, self.world.drones, strict=True):
            others = [a.drone_id for a in self.agents if a is not agent]
            pose = PoseMessage(sender=agent.drone_id, seq=agent.seq, t=t, x=float(d.position[0]), y=float(d.position[1]), z=d.z, yaw=d.yaw)
            belief = self.encoder.encode(agent.drone_id, agent.seq, t, agent.belief)
            self.channel.send(pose, t, others)
            belief_bytes = 0
            if belief is not None:
                self.channel.send(belief, t, others)
                belief_bytes = belief.nbytes
            agent.seq += 1
            ev.write("send", t, drone=agent.drone_id, seq=agent.seq - 1, bytes=pose.nbytes + belief_bytes,
                     belief_bytes=belief_bytes, receivers=len(others))
            ev.write("belief", t, drone=agent.drone_id, values=agent.belief.values.ravel().round(4).tolist(),
                     ages=[int(a) for a in encode_ages(agent.belief.stamps, t, self.grid.shape).ravel()])

    def _receive(self, t: float, ev: EventLog) -> None:
        by_agent = {a.drone_id: a for a in self.agents}
        for delivery in self.channel.deliver(t):
            agent = by_agent[delivery.receiver]
            msg = delivery.message
            if isinstance(msg, PoseMessage):
                agent.peers[msg.sender] = np.array([msg.x, msg.y])
            else:
                agent.inbox.append(msg)
            ev.write("recv", t, drone=agent.drone_id, sender=msg.sender, seq=msg.seq, msg=type(msg).__name__, bytes=msg.nbytes)
        for agent in self.agents:
            if agent.inbox:
                agent.belief = self.fusion.fuse(agent.belief, agent.inbox, t)
                agent.inbox = []
        drain = getattr(self.channel, "drain_drops", None)
        if drain is not None:
            for drop in drain():
                ev.write("drop", t, sender=drop.sender, receiver=drop.receiver, reason=drop.reason, bytes=drop.nbytes)

    def _commands(self) -> dict[int, NDArray[np.float64]]:
        return {
            a.drone_id: self.controller.command(a.drone_id, d.xy, a.peers, a.belief, self.world.t)
            for a, d in zip(self.agents, self.world.drones, strict=True)
        }

    def _log_poses(self, t: float, ev: EventLog) -> None:
        for d in self.world.drones:
            ev.write("pose", t, drone=d.drone_id, x=round(float(d.position[0]), 5), y=round(float(d.position[1]), 5), z=d.z, yaw=round(d.yaw, 4))


def build_scene(cfg: RunConfig) -> Scene:
    if cfg.scene.kind == "random":
        assert cfg.scene.seed is not None
        return random_scene(cfg.scene.seed, cfg.scene.n_targets, cfg.scene.n_distractors,
                            cfg.area.width, cfg.area.height, cfg.swarm.n_drones, cfg.swarm.altitude)
    return default_scene(cfg.swarm.n_drones, cfg.swarm.altitude)


def build_scorer(cfg: RunConfig, grid: Grid, ground_truth: ImportanceField, scene: Scene) -> ImportanceScorer:
    if cfg.scorer.kind == "vlm":
        assert cfg.scorer.model is not None
        return VLMScorer(cfg.scorer.model, grid, scene.mission, cfg.scorer.prompt_set,
                         cfg.scorer.calibration, cfg.scorer.device)
    if cfg.scorer.kind == "cached":
        return CachedScorer(OracleScorer(ground_truth), grid, path=cfg.scorer.cache_path, model_id=cfg.scorer.model or "oracle")
    return OracleScorer(ground_truth)


def build_fusion(cfg: RunConfig) -> BeliefFusion:
    if cfg.fusion.kind == "none":
        return IgnoreMessages()
    return AgeWeightedAverage(cfg.fusion.tau_s)


def build_controller(cfg: RunConfig) -> CoverageController:
    if cfg.controller.kind == "hold":
        return HoldController()
    if cfg.controller.kind == "replay":
        assert cfg.controller.replay_run is not None
        source = RunDir.open(cfg.controller.replay_run)
        if source.config.sim.dt != cfg.sim.dt or source.config.swarm.n_drones != cfg.swarm.n_drones:
            raise ValueError(
                f"replay source {source.path} ran dt={source.config.sim.dt}, n_drones={source.config.swarm.n_drones}; "
                f"this config has dt={cfg.sim.dt}, n_drones={cfg.swarm.n_drones}. Replay needs both to match."
            )
        return ReplayController.from_run(source)
    return LloydController(gain=cfg.controller.gain)


def build_prior(cfg: RunConfig, scene: Scene, grid: Grid, ground_truth: ImportanceField) -> ImportanceField:
    """The belief every drone starts from. Returns a fresh field; callers copy per drone."""
    p = cfg.prior
    floor = scene.mission.floor
    if p.kind == "floor":
        values = np.full(grid.shape, floor)
    elif p.kind == "truth":
        values = ground_truth.values.copy()
    elif p.kind == "shifted_truth":
        dc, dr = (int(round(p.shift_m[0] / grid.cell_size)), int(round(p.shift_m[1] / grid.cell_size)))
        values = np.full(grid.shape, floor)
        rows, cols = grid.shape
        src_r = slice(max(0, -dr), min(rows, rows - dr))
        src_c = slice(max(0, -dc), min(cols, cols - dc))
        dst_r = slice(max(0, dr), min(rows, rows + dr))
        dst_c = slice(max(0, dc), min(cols, cols + dc))
        values[dst_r, dst_c] = ground_truth.values[src_r, src_c]
    else:
        assert p.scene_seed is not None
        other = random_scene(p.scene_seed, cfg.scene.n_targets, cfg.scene.n_distractors,
                             cfg.area.width, cfg.area.height, cfg.swarm.n_drones, cfg.swarm.altitude)
        values = rasterize_ground_truth(other, grid).values
    stamps = np.full(grid.shape, np.nan if p.observed_at is None else float(p.observed_at))
    return ImportanceField(grid, values, stamps)


def build_channel(cfg: RunConfig) -> Channel:
    ch = cfg.channel
    return FaultedChannel(ch.drop_rate, ch.latency_s, ch.bytes_per_sync, seed=cfg.sim.seed)


def build(
    cfg: RunConfig,
    fusion: BeliefFusion | None = None,
    channel: Channel | None = None,
    scorer_wrap: Callable[[ImportanceScorer], ImportanceScorer] | None = None,
    scorer: ImportanceScorer | None = None,
    on_step: Callable[[Simulation, float], None] | None = None,
) -> Simulation:
    """Assemble a simulation from config. `fusion` overrides the configured rule; the channel is
    injectable so tests can substitute one; `scorer_wrap` lets an outer layer wrap the scorer
    without this module knowing how; `scorer` replaces the configured scorer outright (a remote
    one, say); `on_step` runs after every world step (a renderer bridge)."""
    scene = build_scene(cfg)
    if (scene.width, scene.height) != (cfg.area.width, cfg.area.height):
        raise ValueError(
            f"scene is {scene.width}x{scene.height} m but config area is "
            f"{cfg.area.width}x{cfg.area.height}; make them agree"
        )
    grid = Grid(cfg.area.width, cfg.area.height, cfg.area.cell_size)
    ground_truth = rasterize_ground_truth(scene, grid)
    world = KinematicWorld(scene.drone_starts, scene.width, scene.height, cfg.swarm.max_speed, cfg.sim.dt)
    prior = build_prior(cfg, scene, grid, ground_truth)
    agents = [Agent(i, prior.copy()) for i in range(cfg.swarm.n_drones)]
    scorer = scorer or build_scorer(cfg, grid, ground_truth, scene)
    if scorer_wrap is not None:
        scorer = scorer_wrap(scorer)
    return Simulation(
        config=cfg, scene=scene, grid=grid, world=world,
        scorer=scorer,
        fusion=fusion if fusion is not None else build_fusion(cfg),
        controller=build_controller(cfg),
        channel=channel or build_channel(cfg),
        agents=agents,
        encoder=build_encoder(cfg.message),
        on_step=on_step,
    )


def run_from_config(cfg: RunConfig, out_root: Path, **overrides: object) -> RunDir:
    sim = build(cfg, **overrides)  # type: ignore[arg-type]
    return sim.run(RunDir.create(out_root, cfg))
