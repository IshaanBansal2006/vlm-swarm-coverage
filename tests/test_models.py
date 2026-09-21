from __future__ import annotations

import math
import os

import numpy as np
import pytest

from vlm_swarm_coverage.field import Grid
from vlm_swarm_coverage.models import (
    MODEL_IDS,
    DenseImportanceScorer,
    Prompt,
    ScoreCalibration,
    build_prompts,
    combine,
    ground_to_pixel,
    load_backend,
    map_to_cells,
    tile_boxes,
)
from vlm_swarm_coverage.scene import STORM_DAMAGE_MISSION, Mission
from vlm_swarm_coverage.scoring import View, footprint_cells

G = Grid(60.0, 40.0, 2.0)


def _view(x: float = 30.0, y: float = 20.0, z: float = 12.0, yaw: float = 0.0, frame=None) -> View:  # type: ignore[no-untyped-def]
    return View(0, 0.0, x, y, z, yaw, 70.0, 4 / 3, frame, "m")


def test_prompts_from_mission_and_null_set() -> None:
    ps = build_prompts(STORM_DAMAGE_MISSION)
    assert [p.phrase for p in ps] == ["washed-out or cracked road surface", "debris blocking the road", "a stranded or abandoned vehicle"]
    assert [p.weight for p in ps] == [1.0, 0.8, 0.9]
    assert build_prompts(STORM_DAMAGE_MISSION, "null") == [Prompt("something important", 1.0)]
    assert Mission("t", {"tree": 0.5}).phrase("tree") == "tree" and Mission("t", {"debris": 1.0}).phrase("debris") == "debris"
    with pytest.raises(ValueError, match="prompt_set"):
        build_prompts(STORM_DAMAGE_MISSION, "other")
    with pytest.raises(ValueError, match="unknown labels"):
        Mission("t", {}, phrases={"nope": "x"})


def test_calibration_is_affine_clipped_and_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cal = ScoreCalibration(lo=0.2, hi=0.6, floor=0.05)
    np.testing.assert_allclose(cal.apply(np.array([0.0, 0.2, 0.4, 0.6, 1.0])), [0.05, 0.05, 0.525, 1.0, 1.0])
    cal.save(tmp_path / "c.json")
    assert ScoreCalibration.load(tmp_path / "c.json") == cal
    with pytest.raises(ValueError, match="hi > lo"):
        ScoreCalibration(lo=1.0, hi=1.0)


def test_ground_to_pixel_geometry() -> None:
    v = _view(yaw=0.0)
    half_w = 12.0 * math.tan(math.radians(35.0))
    px = ground_to_pixel(v, np.array([[30.0, 20.0], [30.0 + half_w, 20.0], [30.0, 20.0 + half_w / (4 / 3)]]), 800, 600)
    np.testing.assert_allclose(px, [[400, 300], [800, 300], [400, 0]], atol=1e-9)
    turned = _view(yaw=math.pi / 2)
    px = ground_to_pixel(turned, np.array([[30.0, 20.0 + half_w]]), 800, 600)
    np.testing.assert_allclose(px, [[800, 300]], atol=1e-9)


def test_map_to_cells_lights_the_right_cell_and_turns_with_yaw() -> None:
    v = _view()
    cells = footprint_cells(G, v.x, v.y, v.z, v.yaw, v.fov_deg, v.aspect)
    imp = np.zeros((600, 800))
    target = np.array([[35.0, 21.0]])  # 5 m ahead, 1 m left of the drone
    u, w = ground_to_pixel(v, target, 800, 600)[0]
    imp[int(w) - 20:int(w) + 20, int(u) - 30:int(u) + 30] = 1.0
    vals = map_to_cells(imp, v, G, cells)
    best = cells[np.argmax(vals)]
    assert tuple(best) == G.cell_of(35.0, 21.0)
    assert vals.max() > 0.2 and (vals > 0).sum() <= 6
    turned = _view(yaw=math.pi / 2)
    cells_t = footprint_cells(G, turned.x, turned.y, turned.z, turned.yaw, turned.fov_deg, turned.aspect)
    vals_t = map_to_cells(imp, turned, G, cells_t)
    assert tuple(cells_t[np.argmax(vals_t)]) == G.cell_of(29.0, 25.0)


def test_combine_takes_strongest_weighted_phrase_and_tiles_cover_the_frame() -> None:
    maps = np.array([[[0.5, 0.1]], [[0.4, 0.9]]])
    np.testing.assert_allclose(combine(maps, [Prompt("a", 1.0), Prompt("b", 0.5)]), [[0.5, 0.45]])
    boxes = tile_boxes(600, 800, 6, 8)
    assert len(boxes) == 48 and boxes[0] == (0, 100, 0, 100) and boxes[-1] == (500, 600, 700, 800)


def test_dense_scorer_needs_a_frame_and_applies_calibration() -> None:
    class Flat:
        def importance_map(self, frame, prompts):  # type: ignore[no-untyped-def]
            return np.full((6, 8), 0.5)

    scorer = DenseImportanceScorer(Flat(), G, [Prompt("x", 1.0)], ScoreCalibration(0.0, 1.0, 0.05))
    with pytest.raises(ValueError, match="needs a frame"):
        scorer.score(_view())
    obs = scorer.score(_view(frame=np.zeros((60, 80, 3), dtype=np.uint8)))
    assert len(obs.cells) > 0 and np.allclose(obs.values, 0.05 + 0.95 * 0.5) and scorer.frames_scored == 1
    with pytest.raises(ValueError, match="unknown model"):
        load_backend("nope")


@pytest.mark.skipif(os.environ.get("VSC_MODEL_TESTS") != "1", reason="needs torch, weights and a GPU; set VSC_MODEL_TESTS=1")
@pytest.mark.parametrize("model", ["clipseg", "siglip2", "remoteclip-b32", "owlv2"])
def test_real_backends_produce_maps(model: str) -> None:
    pytest.importorskip("torch")
    backend = load_backend(model)
    frame = np.full((600, 800, 3), (60, 110, 50), dtype=np.uint8)
    frame[250:350, 300:420] = (200, 30, 25)
    prompts = build_prompts(STORM_DAMAGE_MISSION)
    a = backend.importance_map(frame, prompts)
    b = backend.importance_map(frame, prompts)
    assert a.ndim == 2 and a.shape[0] >= 6 and np.isfinite(a).all()
    np.testing.assert_allclose(a, b, atol=1e-5)
    assert MODEL_IDS[model][0] in ("clipseg", "tiles", "openclip", "owlv2")
