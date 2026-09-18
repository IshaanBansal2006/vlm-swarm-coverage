from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_swarm_coverage.config import AreaConfig, RunConfig

REPO = Path(__file__).resolve().parents[1]


def test_demo_config_loads_and_round_trips(tmp_path: Path) -> None:
    cfg = RunConfig.load(REPO / "configs" / "demo.toml")
    assert cfg.name == "demo"
    assert cfg.swarm.n_drones == 3
    out = tmp_path / "config.json"
    cfg.dump_json(out)
    assert RunConfig.load(out) == cfg


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "bad.toml"
    p.write_text('name = "x"\n[swarm]\nn_drone = 2\n')
    with pytest.raises(ValidationError, match="n_drone"):
        RunConfig.load(p)


def test_scorer_and_scene_kinds_require_their_inputs() -> None:
    with pytest.raises(ValidationError, match="scorer.model"):
        RunConfig.model_validate({"scorer": {"kind": "vlm"}})
    with pytest.raises(ValidationError, match="cache_path"):
        RunConfig.model_validate({"scorer": {"kind": "cached"}})
    with pytest.raises(ValidationError, match="scene.seed"):
        RunConfig.model_validate({"scene": {"kind": "random"}})


def test_grid_shape_rounds_partial_cells_up() -> None:
    assert AreaConfig(width=60.0, height=40.0, cell_size=2.0).shape == (20, 30)
    assert AreaConfig(width=61.0, height=40.5, cell_size=2.0).shape == (21, 31)


def test_unsupported_suffix_is_actionable(tmp_path: Path) -> None:
    p = tmp_path / "config.yaml"
    p.write_text("name: x\n")
    with pytest.raises(ValueError, match=r"\.toml or \.json"):
        RunConfig.load(p)
