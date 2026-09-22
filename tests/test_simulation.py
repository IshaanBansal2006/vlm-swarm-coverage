from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.__main__ import main
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.field import Grid
from vlm_swarm_coverage.simulation import build, build_scene, jittered_starts, run_from_config

if TYPE_CHECKING:
    from pathlib import Path


def _short(**kw: object) -> RunConfig:
    base = {"name": "t", "sim": {"dt": 0.1, "duration_s": 20.0, "seed": 1}}
    base.update(kw)
    return RunConfig.model_validate(base)


def test_loop_logs_every_kind_and_drones_move_toward_targets(tmp_path: Path) -> None:
    cfg = _short()
    run = run_from_config(cfg, tmp_path)
    kinds = {e["kind"] for e in run.iter_events()}
    assert {"start", "pose", "score", "send", "recv", "belief", "end"} <= kinds
    poses = list(run.iter_events("pose"))
    assert len(poses) == cfg.sim.n_steps * cfg.swarm.n_drones
    first = np.array([[e["x"], e["y"]] for e in poses[: cfg.swarm.n_drones]])
    last = np.array([[e["x"], e["y"]] for e in poses[-cfg.swarm.n_drones :]])
    assert last[:, 0].mean() > first[:, 0].mean() + 5.0


def test_belief_fills_in_from_observations(tmp_path: Path) -> None:
    sim = build(_short())
    floor = sim.scene.mission.floor
    assert all((a.belief.values == floor).all() for a in sim.agents)
    sim.run(__import__("vlm_swarm_coverage.artifacts").artifacts.RunDir.create(tmp_path, sim.config))
    assert any((a.belief.values > floor).any() for a in sim.agents)


def test_hold_controller_keeps_drones_still(tmp_path: Path) -> None:
    run = run_from_config(_short(controller={"kind": "hold"}), tmp_path)
    poses = list(run.iter_events("pose"))
    xs = {e["drone"]: set() for e in poses}
    for e in poses:
        xs[e["drone"]].add((e["x"], e["y"]))
    assert all(len(v) == 1 for v in xs.values())


def test_peers_are_learned_over_the_channel(tmp_path: Path) -> None:
    sim = build(_short(swarm={"n_drones": 2}))
    sim.run(__import__("vlm_swarm_coverage.artifacts").artifacts.RunDir.create(tmp_path, sim.config))
    assert set(sim.agents[0].peers) == {1} and set(sim.agents[1].peers) == {0}


def test_grid_matches_config(tmp_path: Path) -> None:
    sim = build(_short(area={"width": 60.0, "height": 40.0, "cell_size": 4.0}))
    assert sim.grid == Grid(60.0, 40.0, 4.0)


def test_cli_runs_demo_config(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    cfg = tmp_path / "c.toml"
    cfg.write_text('name = "cli"\n[sim]\nduration_s = 2.0\n')
    assert main([str(cfg), "--out", str(tmp_path / "runs")]) == 0
    assert "cli-" in capsys.readouterr().out


def test_faulted_run_logs_drops_and_stats(tmp_path: Path) -> None:
    run = run_from_config(_short(channel={"drop_rate": 0.5, "latency_s": 0.3}), tmp_path)
    drops = list(run.iter_events("drop"))
    assert drops and all(d["reason"] == "loss" for d in drops)
    end = next(run.iter_events("end"))
    assert end["channel"]["dropped_loss"] == len(drops)
    assert end["channel"]["delivered"] > 0


def test_same_seed_same_run(tmp_path: Path) -> None:
    cfg = _short(channel={"drop_rate": 0.4})
    a = [e for e in run_from_config(cfg, tmp_path / "a").iter_events("pose")]
    b = [e for e in run_from_config(cfg, tmp_path / "b").iter_events("pose")]
    assert a == b


def test_scorer_wrap_is_applied(tmp_path: Path) -> None:
    seen: list[str] = []

    def wrap(inner):  # type: ignore[no-untyped-def]
        seen.append(type(inner).__name__)
        return inner

    build(_short(), scorer_wrap=wrap)
    assert seen == ["OracleScorer"]


def test_fusion_brings_beliefs_closer_than_no_fusion(tmp_path: Path) -> None:
    def spread(kind: str) -> float:
        sim = build(_short(fusion={"kind": kind}))
        sim.run(__import__("vlm_swarm_coverage.artifacts").artifacts.RunDir.create(tmp_path / kind, sim.config))
        a, b = sim.agents[0].belief.values, sim.agents[1].belief.values
        return float(np.abs(a - b).sum())

    assert spread("age_weighted") < 0.5 * spread("none")


def test_belief_events_carry_ages(tmp_path: Path) -> None:
    run = run_from_config(_short(), tmp_path)
    last = list(run.iter_events("belief"))[-1]
    assert len(last["ages"]) == len(last["values"]) and min(last["ages"]) == 0


def test_start_jitter_is_off_by_default_so_old_runs_reproduce() -> None:
    """Every run recorded before start jitter existed used the fixed line-up. The default must
    reproduce it exactly, or the frozen study's index stops describing its own artifacts."""
    cfg = RunConfig.model_validate({"sim": {"seed": 7}})
    scene = build_scene(cfg)
    assert jittered_starts(scene, cfg) == scene.drone_starts


def test_start_jitter_makes_sim_seed_an_independent_replicate() -> None:
    """With a cached scorer and a perfect channel nothing else in a run is stochastic, so without
    jitter five seeds of one layout give bit-identical results and a seed axis carries no
    information. With jitter they differ, and differ only in the initial condition."""
    base = {"swarm": {"start_jitter_m": 4.0}}
    scene = build_scene(RunConfig.model_validate(base))
    starts = [jittered_starts(scene, RunConfig.model_validate({**base, "sim": {"seed": s}}))
              for s in range(5)]
    assert len({tuple(s) for s in starts}) == 5
    for s in starts:
        assert len(s) == len(scene.drone_starts)
        for (x, y, z), (_, _, z0) in zip(s, scene.drone_starts, strict=True):
            assert 0.0 <= x <= scene.width and 0.0 <= y <= scene.height
            assert z == z0, "jitter is horizontal; altitude is a mission parameter"
    # and it is reproducible from the seed alone
    again = jittered_starts(scene, RunConfig.model_validate({**base, "sim": {"seed": 3}}))
    assert again == starts[3]


def test_jitter_radius_is_respected() -> None:
    scene = build_scene(RunConfig.model_validate({}))
    for seed in range(20):
        cfg = RunConfig.model_validate({"swarm": {"start_jitter_m": 2.0}, "sim": {"seed": seed}})
        for (x, y, _), (x0, y0, _) in zip(jittered_starts(scene, cfg), scene.drone_starts, strict=True):
            assert math.hypot(x - x0, y - y0) <= 2.0 + 1e-9
