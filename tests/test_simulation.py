from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.__main__ import main
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.field import Grid
from vlm_swarm_coverage.simulation import build, run_from_config

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
