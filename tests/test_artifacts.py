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


def _compress_log(run: RunDir) -> None:
    """What `aos runs compress` does: events.jsonl -> events.jsonl.zst, original removed."""
    zstandard = pytest.importorskip("zstandard")
    run.compressed_events_path.write_bytes(
        zstandard.ZstdCompressor(level=9).compress(run.events_path.read_bytes())
    )
    run.events_path.unlink()


def test_iter_events_reads_a_compressed_log(tmp_path: Path) -> None:
    run = RunDir.create(tmp_path, RunConfig(name="t"))
    with run.events() as ev:
        for i in range(500):
            ev.write("pose" if i % 2 else "score", i * 0.1, drone=i % 3, xy=[i, -i])
    before = list(run.iter_events())
    _compress_log(run)
    reopened = RunDir.open(run.path)
    assert list(reopened.iter_events()) == before
    assert [e["drone"] for e in reopened.iter_events("pose")][:3] == [1, 0, 2]


def test_appending_to_a_compressed_run_is_refused(tmp_path: Path) -> None:
    run = RunDir.create(tmp_path, RunConfig(name="t"))
    with run.events() as ev:
        ev.write("pose", 0.0, drone=0)
    _compress_log(run)
    with pytest.raises(FileExistsError, match="aos runs restore"):
        run.events()
    assert not run.events_path.exists()


def test_open_non_run_dir_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        RunDir.open(tmp_path)


def test_git_sha_outside_repo_returns_none_and_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(artifacts.log, "warning", lambda msg, *a: warnings.append(msg % a))
    assert git_sha(tmp_path) is None
    assert warnings and "git sha unavailable" in warnings[0]
