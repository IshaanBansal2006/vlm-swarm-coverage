from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from vlm_swarm_coverage.artifacts import RunDir
from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.sweep import (
    SweepSpec,
    apply_overrides,
    config_hash,
    expand,
    main,
    read_index,
    run_sweep,
)

if TYPE_CHECKING:
    from pathlib import Path


def hook(cfg: RunConfig, params: dict[str, Any]) -> dict[str, Any]:
    def wrap(inner):  # type: ignore[no-untyped-def]
        inner.tag = params.get("tag", "none")
        return inner

    return {"scorer_wrap": wrap}


def post(run_dir: RunDir) -> dict[str, Any]:
    return {"n_pose": sum(1 for _ in run_dir.iter_events("pose"))}


def _base(tmp_path: Path) -> Path:
    p = tmp_path / "base.toml"
    p.write_text('name = "b"\n[sim]\nduration_s = 2.0\n')
    return p


def test_expand_full_and_one_at_a_time(tmp_path: Path) -> None:
    spec = SweepSpec(name="s", base=_base(tmp_path), axes={"channel.drop_rate": [0.0, 0.5], "_tag": ["a", "b"]}, seeds=[0, 1])
    full = expand(spec)
    assert len(full) == 8 and len({c.cell_id for c in full}) == 8
    assert all(set(c.overrides) == {"channel.drop_rate", "sim.seed"} and set(c.params) == {"tag"} for c in full)
    oat = expand(spec.model_copy(update={"design": "one_at_a_time"}))
    assert len(oat) == (1 + 4) * 2
    explicit = expand(SweepSpec(name="s", base=_base(tmp_path), cells=[{"sim.seed": 7, "fusion.kind": "none"}], seeds=[0, 1]))
    assert len(explicit) == 1 and explicit[0].overrides["sim.seed"] == 7


def test_apply_overrides_and_hash() -> None:
    cfg = RunConfig(name="x")
    out = apply_overrides(cfg, {"channel.drop_rate": 0.3, "message.kind": "topk", "message.k": 5}, name="y")
    assert out.channel.drop_rate == 0.3 and out.message.k == 5 and out.name == "y"
    assert config_hash(out) != config_hash(cfg) and config_hash(cfg) == config_hash(RunConfig(name="x"))
    with pytest.raises(ValueError):
        apply_overrides(cfg, {"channel.nope": 1})


def test_run_sweep_indexes_resumes_and_applies_hooks(tmp_path: Path) -> None:
    spec = SweepSpec(name="tiny", base=_base(tmp_path), axes={"channel.drop_rate": [0.0, 0.5], "_tag": ["a"]},
                     seeds=[0, 1], workers=2, hook="test_sweep:hook", post_run="test_sweep:post")
    rows = run_sweep(spec, tmp_path / "out")
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    assert all(r["n_pose"] == 20 * 3 and r["params"] == {"tag": "a"} for r in rows)
    index = read_index(tmp_path / "out" / "tiny" / "index.jsonl")
    assert len(index) == 4 and (tmp_path / "out" / "tiny" / "spec.json").exists()
    again = run_sweep(spec, tmp_path / "out")
    assert len(again) == 4 and len(read_index(tmp_path / "out" / "tiny" / "index.jsonl")) == 4
    run = RunDir.open(rows[0]["run_dir"])
    assert run.config.channel.drop_rate in (0.0, 0.5) and run.config.name.startswith("tiny-")


def test_failed_cell_is_recorded_not_raised(tmp_path: Path) -> None:
    spec = SweepSpec(name="bad", base=_base(tmp_path), cells=[{"controller.kind": "replay", "controller.replay_run": str(tmp_path / "missing")}], workers=1)
    rows = run_sweep(spec, tmp_path / "out")
    assert len(rows) == 1 and rows[0]["status"] == "error" and "missing" in rows[0]["error"]
    rerun = run_sweep(spec, tmp_path / "out")
    assert len(rerun) == 2  # errors are retried, not skipped


def test_cli_dry_run_and_spec_loading(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _base(tmp_path)
    spec_path = tmp_path / "s.toml"
    spec_path.write_text('name = "cli"\nbase = "base.toml"\nseeds = [0]\n[axes]\n"channel.drop_rate" = [0.0, 0.2]\n')
    spec = SweepSpec.load(spec_path)
    assert spec.base == (tmp_path / "base.toml").resolve()
    assert main([str(spec_path), "--out", str(tmp_path / "out"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("channel.drop_rate") == 2 and "0 rows" in out
    assert main([str(spec_path), "--out", str(tmp_path / "out"), "--workers", "2"]) == 0
    assert json.loads((tmp_path / "out" / "cli" / "spec.json").read_text())["name"] == "cli"


def test_composite_axis_levels_move_several_keys_together(tmp_path: Path) -> None:
    spec = SweepSpec(name="c", base=_base(tmp_path), seeds=[0],
                     axes={"message": [{"kind": "dense"}, {"kind": "topk", "k": 5}], "_error": [{"sigma": 0.1}]})
    cells = expand(spec)
    assert len(cells) == 2
    assert cells[1].overrides == {"message.kind": "topk", "message.k": 5, "sim.seed": 0} and cells[1].params == {"error": {"sigma": 0.1}}
    cfg = apply_overrides(RunConfig(name="x"), cells[1].overrides)
    assert cfg.message.kind == "topk" and cfg.message.k == 5
    explicit = expand(SweepSpec(name="c", base=_base(tmp_path), seeds=[1], cells=[{"prior": {"kind": "other_scene", "scene_seed": 3}}]))
    assert explicit[0].overrides == {"prior.kind": "other_scene", "prior.scene_seed": 3, "sim.seed": 1}
