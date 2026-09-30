from __future__ import annotations

import os

import numpy as np
import pytest

from vlm_swarm_coverage import models
from vlm_swarm_coverage.models import MODEL_IDS, build_prompts, gaussian_splat_max, load_backend
from vlm_swarm_coverage.scene import STORM_DAMAGE_MISSION


def test_splat_peaks_at_the_box_centre_with_the_score() -> None:
    m = gaussian_splat_max(np.array([[10.0, 20.0, 30.0, 40.0]]), np.array([0.7]), 60, 50)
    assert m.shape == (60, 50)
    v, u = np.unravel_index(np.argmax(m), m.shape)
    assert (u, v) in {(19, 29), (20, 30), (19, 30), (20, 29)}
    assert 0.65 < m.max() <= 0.7
    # sigma is a quarter of the box: pixel 10 (centre 10.5) is 9.5 px = 1.9 sigma from x = 20
    np.testing.assert_allclose(m[30, 10] / m[30, 19], np.exp(-0.5 * 1.9**2) / np.exp(-0.5 * 0.1**2), rtol=1e-6)
    assert m[0, 0] == 0.0


def test_splat_combines_overlaps_by_maximum_not_sum() -> None:
    box = np.array([[10.0, 10.0, 30.0, 30.0]])
    one = gaussian_splat_max(box, np.array([0.5]), 40, 40)
    many = gaussian_splat_max(np.repeat(box, 5, axis=0), np.full(5, 0.5), 40, 40)
    np.testing.assert_array_equal(one, many)
    both = gaussian_splat_max(np.concatenate([box, box]), np.array([0.2, 0.9]), 40, 40)
    np.testing.assert_allclose(both, gaussian_splat_max(box, np.array([0.9]), 40, 40))


def test_splat_handles_no_boxes_and_boxes_off_the_map() -> None:
    assert not gaussian_splat_max(np.zeros((0, 4)), np.zeros(0), 8, 8).any()
    assert not gaussian_splat_max(np.array([[100.0, 100.0, 120.0, 120.0]]), np.array([1.0]), 8, 8).any()


def test_new_models_dispatch_to_their_own_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[str] = []
    for name in ("Owlv2DensityBackend", "SiglipPatchBackend", "OpenClipPatchBackend"):
        monkeypatch.setattr(models, name, lambda spec, device, _n=name: made.append(_n) or _n)
    assert load_backend("owlv2-density") == "Owlv2DensityBackend"
    assert load_backend("siglip2-dense") == "SiglipPatchBackend"
    assert load_backend("remoteclip-l14-dense") == "OpenClipPatchBackend"
    monkeypatch.setitem(MODEL_IDS, "mystery", ("nofamily", "x"))
    with pytest.raises(ValueError, match="family"):
        load_backend("mystery")


@pytest.mark.skipif(os.environ.get("VSC_MODEL_TESTS") != "1", reason="needs torch, weights and a GPU; set VSC_MODEL_TESTS=1")
@pytest.mark.parametrize("model", ["owlv2-density", "siglip2-dense", "remoteclip-l14-dense"])
def test_dense_backends_produce_maps(model: str) -> None:
    pytest.importorskip("torch")
    backend = load_backend(model)
    frame = np.full((336, 448, 3), (60, 110, 50), dtype=np.uint8)
    frame[130:200, 180:280] = (200, 30, 25)
    prompts = build_prompts(STORM_DAMAGE_MISSION)
    a = backend.importance_map(frame, prompts)
    assert a.ndim == 2 and np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0
    np.testing.assert_allclose(a, backend.importance_map(frame, prompts), atol=1e-5)
