from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from vlm_swarm_coverage import artifacts
from vlm_swarm_coverage.artifacts import RunDir, git_sha
from vlm_swarm_coverage.config import RunConfig

if TYPE_CHECKING:
    from pathlib import Path


def test_create_writes_config_and_meta(tmp_path: Path) -> None:
    cfg = RunConfig(name="t")
    run = RunDir.create(tmp_path, cfg)
    assert run.path.name.startswith("t-")
    assert RunConfig.load(run.path / "config.json") == cfg
    meta = json.loads((run.path / "meta.json").read_text())
    assert meta["run_id"] == run.path.name
    assert meta["package_version"]


def test_events_round_trip_and_filter(tmp_path: Path) -> None:
    run = RunDir.create(tmp_path, RunConfig(name="t"))
    with run.events(flush_every=2) as ev:
        ev.write("pose", 0.0, drone=0, xy=[1.0, 2.0])
        ev.write("pose", 0.1, drone=1, xy=[3.0, 4.0])
        ev.write("score", 0.1, drone=0, cells=[[0, 0, 0.5]])
        assert ev.count == 3
    reopened = RunDir.open(run.path)
    assert reopened.config == run.config
    assert [e["drone"] for e in reopened.iter_events("pose")] == [0, 1]
    assert len(list(reopened.iter_events())) == 3


def test_iter_events_on_run_that_never_logged(tmp_path: Path) -> None:
    run = RunDir.create(tmp_path, RunConfig(name="t"))
    assert list(run.iter_events()) == []


def test_open_non_run_dir_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        RunDir.open(tmp_path)


def test_git_sha_outside_repo_returns_none_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(artifacts.log, "warning", lambda msg, *a: warnings.append(msg % a))
    assert git_sha(tmp_path) is None
    assert warnings and "git sha unavailable" in warnings[0]
