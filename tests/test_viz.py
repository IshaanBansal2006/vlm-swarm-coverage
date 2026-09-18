from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("matplotlib")

from vlm_swarm_coverage.config import RunConfig
from vlm_swarm_coverage.simulation import run_from_config
from vlm_swarm_coverage.viz import load_run, main, render_video, summary_figure

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    cfg = RunConfig.model_validate({"name": "viz", "sim": {"duration_s": 4.0}, "channel": {"drop_rate": 0.3}})
    return run_from_config(cfg, tmp_path_factory.mktemp("runs"))


def test_load_run_shapes(run) -> None:  # type: ignore[no-untyped-def]
    f = load_run(run)
    assert f.n_drones == 3
    assert f.positions.shape == (len(f.times), 3, 2)
    assert f.beliefs.shape == (len(f.times), 3, *f.grid.shape)
    assert f.ground_truth.shape == f.grid.shape
    assert len(f.times) == 4


def test_summary_and_video_write_files(run, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    png = summary_figure(run, tmp_path / "s.png")
    assert png.exists() and png.stat().st_size > 1000
    out = render_video(run, tmp_path / "v.mp4", fps=4, size=(640, 360))
    assert out.exists()


def test_cli_requires_an_output(run) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit):
        main([str(run.path)])
