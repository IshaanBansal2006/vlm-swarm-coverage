"""The models behind the scorer slot (decision 015).

A model here is an `ImageScorer`: frame in, importance map out, one value per pixel in the
model's own units. Three families:

- `ClipSegBackend` — CLIPSeg, dense by construction: one logit map per phrase.
- `TileBackend` — any image-text model (SigLIP 2 through `transformers`, RemoteCLIP and other
  open_clip checkpoints through `open_clip`): the frame is cut into a grid of tiles, each tile is
  embedded as an image, and each tile's similarity to each phrase is its score. Dense by tiling.
- `Owlv2Backend` — an open-vocabulary detector: boxes with scores, painted into a map.

`DenseImportanceScorer` turns a map into an `Observation`: it maps every grid cell in the nadir
footprint to its pixel box and averages the map there, then applies an affine `ScoreCalibration`
so the value lands in importance units. Prompts come from the mission (`build_prompts`).

Every backend returns per-phrase *logits* (k, h, w) on a comparable scale: the decoder logits
for CLIPSeg, the model's scaled and biased cosine for SigLIP, 100 x cosine for CLIP-style models.
`combine` then either takes each phrase's own sigmoid (`contrast=False`) or the softmax across
all phrases, target and background alike (`contrast=True`, decision 016), and keeps the strongest
weighted target. Contrast is what makes a similarity model say "this tile is more road than
grass" instead of "this tile is aerial imagery", which raw cosines mostly say.

torch is imported inside the backends so this module imports without it; the backends are only
constructed when a model is asked for.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from vlm_swarm_coverage.scoring import Observation, footprint_cells

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from vlm_swarm_coverage.field import Grid
    from vlm_swarm_coverage.scene import Mission
    from vlm_swarm_coverage.scoring import View

log = logging.getLogger(__name__)

MODEL_IDS: dict[str, tuple[str, str]] = {
    "clipseg": ("clipseg", "CIDAS/clipseg-rd64-refined"),
    "siglip2": ("tiles", "google/siglip2-base-patch16-384"),
    "siglip2-so400m": ("tiles", "google/siglip2-so400m-patch16-384"),
    "remoteclip-b32": ("openclip", "ViT-B-32|chendelong/RemoteCLIP|RemoteCLIP-ViT-B-32.pt"),
    "remoteclip-l14": ("openclip", "ViT-L-14|chendelong/RemoteCLIP|RemoteCLIP-ViT-L-14.pt"),
    "owlv2": ("owlv2", "google/owlv2-base-patch16-ensemble"),
}
NULL_PHRASE = "something important"
BACKGROUND_PHRASES: tuple[str, ...] = ("grass", "bare soil", "an asphalt road", "a tree", "a building roof", "an empty field")
CLIP_LOGIT_SCALE = 100.0


@dataclass(frozen=True)
class Prompt:
    """A phrase and its importance weight; weight 0 marks a background phrase that exists only to
    be contrasted against (a target's probability is taken relative to it, never reported)."""

    phrase: str
    weight: float

    @property
    def is_target(self) -> bool:
        return self.weight > 0


def build_prompts(mission: Mission, prompt_set: str = "mission", contrast: bool = True) -> list[Prompt]:
    """`mission`: one phrase per weighted label. `null`: a single generic phrase at weight one.
    With `contrast`, the background phrases follow at weight zero."""
    if prompt_set == "null":
        prompts = [Prompt(NULL_PHRASE, 1.0)]
    elif prompt_set == "mission":
        prompts = [Prompt(mission.phrase(label), w) for label, w in sorted(mission.weights.items()) if w > 0]
        if not prompts:
            raise ValueError("the mission has no positively weighted labels to prompt for")
    else:
        raise ValueError(f"prompt_set must be 'mission' or 'null', got {prompt_set!r}")
    if contrast:
        prompts += [Prompt(b, 0.0) for b in BACKGROUND_PHRASES]
    return prompts


@dataclass(frozen=True)
class ScoreCalibration:
    """Affine map from a model's raw score to importance: floor + (1 - floor) · clip((s - lo) / (hi - lo)).

    `lo` and `hi` are fitted once per model (percentiles of raw scores on a calibration set) so
    every model lands on the same [floor, 1] scale the answer key uses. Identity by default."""

    lo: float = 0.0
    hi: float = 1.0
    floor: float = 0.0

    def __post_init__(self) -> None:
        if self.hi <= self.lo:
            raise ValueError(f"calibration needs hi > lo, got lo={self.lo}, hi={self.hi}")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError(f"calibration floor must lie in [0, 1), got {self.floor}")

    def apply(self, raw: NDArray[np.float64]) -> NDArray[np.float64]:
        unit = np.clip((np.asarray(raw, dtype=np.float64) - self.lo) / (self.hi - self.lo), 0.0, 1.0)
        return self.floor + (1.0 - self.floor) * unit

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> ScoreCalibration:
        return cls(**json.loads(Path(path).read_text()))


# --- Geometry: nadir frame <-> ground -------------------------------------------------------------


def ground_to_pixel(view: View, points: NDArray[np.float64], width: int, height: int) -> NDArray[np.float64]:
    """(k, 2) ground points -> (k, 2) pixel (u, v), continuous, for a nadir camera whose image
    x axis runs along the drone's heading and whose image up is the drone's left."""
    half_w = view.z * math.tan(math.radians(view.fov_deg) / 2)
    half_h = half_w / view.aspect
    c, s = math.cos(view.yaw), math.sin(view.yaw)
    d = points - np.array([view.x, view.y])
    along = d[:, 0] * c + d[:, 1] * s
    left = -d[:, 0] * s + d[:, 1] * c
    u = (along / half_w + 1.0) / 2.0 * width
    v = (1.0 - left / half_h) / 2.0 * height
    return np.stack([u, v], axis=1)


def map_to_cells(imp_map: NDArray[np.float64], view: View, grid: Grid, cells: NDArray[np.int64]) -> NDArray[np.float64]:
    """Mean of the map over each cell's pixel box (the box around the cell's four corners)."""
    if len(cells) == 0:
        return np.zeros(0)
    height, width = imp_map.shape
    centres = grid.cell_centers()[cells[:, 0], cells[:, 1]]
    h = grid.cell_size / 2.0
    corners = np.concatenate([centres + np.array(o) * h for o in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    px = ground_to_pixel(view, corners, width, height).reshape(4, len(cells), 2)
    u0 = np.clip(np.floor(px[..., 0].min(axis=0)), 0, width - 1).astype(int)
    u1 = np.clip(np.ceil(px[..., 0].max(axis=0)), 1, width).astype(int)
    v0 = np.clip(np.floor(px[..., 1].min(axis=0)), 0, height - 1).astype(int)
    v1 = np.clip(np.ceil(px[..., 1].max(axis=0)), 1, height).astype(int)
    u1, v1 = np.maximum(u1, u0 + 1), np.maximum(v1, v0 + 1)
    integral = np.zeros((height + 1, width + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(np.asarray(imp_map, dtype=np.float64), axis=0), axis=1)
    total = integral[v1, u1] - integral[v0, u1] - integral[v1, u0] + integral[v0, u0]
    return total / ((v1 - v0) * (u1 - u0))


# --- Backends --------------------------------------------------------------------------------------


class ImageScorer(Protocol):
    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """(h, w) map of raw importance for the frame; h, w need not equal the frame's."""
        ...


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _features(out):  # type: ignore[no-untyped-def]
    """transformers returns a tensor or an output object depending on version; take the tensor."""
    return getattr(out, "pooler_output", None) if hasattr(out, "pooler_output") else out


def combine(logits: NDArray[np.float64], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
    """(k, h, w) per-phrase logits -> (h, w) importance in [0, 1]: each target phrase's probability
    times its weight, keeping the strongest, mirroring the answer key's rule that a cell takes its
    most important feature. With any background phrase present the probabilities are a softmax
    across all phrases (contrast); without, each phrase's own sigmoid."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[0] != len(prompts):
        raise ValueError(f"{logits.shape[0]} logit maps for {len(prompts)} prompts")
    if any(not p.is_target for p in prompts):
        shifted = logits - logits.max(axis=0, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=0, keepdims=True)
    else:
        probs = 1.0 / (1.0 + np.exp(-logits))
    weights = np.array([p.weight for p in prompts])[:, None, None]
    return (probs * weights).max(axis=0)


class ClipSegBackend:
    """CLIPSeg: one 352x352 logit map per phrase, sigmoid to [0, 1]."""

    def __init__(self, hf_id: str = MODEL_IDS["clipseg"][1], device: str | None = None) -> None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = _device(device)
        self.processor = CLIPSegProcessor.from_pretrained(hf_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_id).to(self.device).eval()
        self._torch = torch

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        from PIL import Image

        image = Image.fromarray(np.asarray(frame))
        texts = [p.phrase for p in prompts]
        inputs = self.processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits
        maps = logits.float().cpu().numpy().reshape(len(texts), *logits.shape[-2:])
        return combine(maps, prompts)


def tile_boxes(height: int, width: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    ys = np.linspace(0, height, rows + 1).astype(int)
    xs = np.linspace(0, width, cols + 1).astype(int)
    return [(ys[r], ys[r + 1], xs[c], xs[c + 1]) for r in range(rows) for c in range(cols)]


class TileBackend:
    """Any image-text model, made dense by tiling: each tile is embedded as an image and scored
    against each phrase. SigLIP-style models give a per-tile probability (sigmoid of the scaled,
    biased cosine); CLIP-style models give the cosine itself. The map is `rows x cols`."""

    def __init__(self, hf_id: str = MODEL_IDS["siglip2"][1], device: str | None = None, tiles: tuple[int, int] = (6, 8)) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.device = _device(device)
        self.tiles = tiles
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModel.from_pretrained(hf_id).to(self.device).eval()
        self._torch = torch
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _text(self, phrases: tuple[str, ...]):  # type: ignore[no-untyped-def]
        if phrases not in self._text_cache:
            inputs = self.processor(text=list(phrases), padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                emb = _features(self.model.get_text_features(**inputs))
            self._text_cache[phrases] = emb / emb.norm(dim=-1, keepdim=True)
        return self._text_cache[phrases]

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        from PIL import Image

        frame = np.asarray(frame)
        rows, cols = self.tiles
        crops = [Image.fromarray(frame[y0:y1, x0:x1]) for y0, y1, x0, x1 in tile_boxes(frame.shape[0], frame.shape[1], rows, cols)]
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            img = _features(self.model.get_image_features(**inputs))
        text = self._text(tuple(p.phrase for p in prompts))
        scale = getattr(self.model, "logit_scale", None)
        bias = getattr(self.model, "logit_bias", None)
        with self._torch.inference_mode():
            img = img / img.norm(dim=-1, keepdim=True)
            cos = img @ text.T  # (tiles, phrases)
            if scale is not None and bias is not None:
                cos = cos * scale.exp() + bias
            else:
                cos = cos * CLIP_LOGIT_SCALE
            maps = cos.T.detach().float().cpu().numpy().reshape(len(prompts), rows, cols)
        return combine(maps, prompts)


class OpenClipTileBackend(TileBackend):
    """The tile scorer for open_clip checkpoints (RemoteCLIP, GeoRSCLIP): `arch|repo|filename`."""

    def __init__(self, spec: str = MODEL_IDS["remoteclip-b32"][1], device: str | None = None, tiles: tuple[int, int] = (6, 8)) -> None:
        import open_clip
        import torch
        from huggingface_hub import hf_hub_download

        arch, repo, filename = spec.split("|")
        self.device = _device(device)
        self.tiles = tiles
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch)
        state = torch.load(hf_hub_download(repo, filename), map_location="cpu")
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(arch)
        self._text_cache = {}

    def _text(self, phrases: tuple[str, ...]):  # type: ignore[no-untyped-def]
        if phrases not in self._text_cache:
            with self._torch.inference_mode():
                emb = self.model.encode_text(self.tokenizer(list(phrases)).to(self.device))
            self._text_cache[phrases] = emb / emb.norm(dim=-1, keepdim=True)
        return self._text_cache[phrases]

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        from PIL import Image

        frame = np.asarray(frame)
        rows, cols = self.tiles
        crops = [self.preprocess(Image.fromarray(frame[y0:y1, x0:x1])) for y0, y1, x0, x1 in tile_boxes(frame.shape[0], frame.shape[1], rows, cols)]
        batch = self._torch.stack(crops).to(self.device)
        with self._torch.inference_mode():
            img = self.model.encode_image(batch)
        img = img / img.norm(dim=-1, keepdim=True)
        logits = (img @ self._text(tuple(p.phrase for p in prompts)).T).T * CLIP_LOGIT_SCALE
        return combine(logits.float().cpu().numpy().reshape(len(prompts), rows, cols), prompts)


class Owlv2Backend:
    """OWLv2: boxes with scores per phrase, painted into a map at the frame's resolution; a
    pixel takes the strongest weighted detection covering it, zero elsewhere."""

    def __init__(self, hf_id: str = MODEL_IDS["owlv2"][1], device: str | None = None, threshold: float = 0.1) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = _device(device)
        self.threshold = threshold
        self.processor = Owlv2Processor.from_pretrained(hf_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(hf_id).to(self.device).eval()
        self._torch = torch

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        from PIL import Image

        frame = np.asarray(frame)
        image = Image.fromarray(frame)
        targets = [p for p in prompts if p.is_target]
        texts = [[p.phrase for p in targets]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(image.size)
        results = self.processor.post_process_grounded_object_detection(outputs, threshold=self.threshold, target_sizes=[(side, side)])[0]
        out = np.zeros(frame.shape[:2])
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"], strict=True):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            x0, x1 = max(0, x0), min(frame.shape[1], x1)
            y0, y1 = max(0, y0), min(frame.shape[0], y1)
            if x1 > x0 and y1 > y0:
                out[y0:y1, x0:x1] = np.maximum(out[y0:y1, x0:x1], float(score) * targets[int(label)].weight)
        return out


# --- The scorer ------------------------------------------------------------------------------------


class DenseImportanceScorer:
    """A backend plus prompts plus calibration, applied to the cells in the camera footprint."""

    def __init__(self, backend: ImageScorer, grid: Grid, prompts: Sequence[Prompt], calibration: ScoreCalibration | None = None) -> None:
        self.backend, self.grid, self.prompts = backend, grid, list(prompts)
        self.calibration = calibration or ScoreCalibration()
        self.frames_scored = 0

    def score(self, view: View) -> Observation:
        if view.frame is None:
            raise ValueError(
                f"the model scorer needs a frame on the view (drone {view.drone_id}, t={view.t}); "
                "attach a renderer, or run with scorer.kind='cached' against a filled cache"
            )
        cells = footprint_cells(self.grid, view.x, view.y, view.z, view.yaw, view.fov_deg, view.aspect)
        imp = self.backend.importance_map(view.frame, self.prompts)
        raw = map_to_cells(imp, view, self.grid, cells)
        self.frames_scored += 1
        return Observation(view.drone_id, view.t, cells, self.calibration.apply(raw))


def load_backend(model: str, device: str | None = None) -> ImageScorer:
    if model not in MODEL_IDS:
        raise ValueError(f"unknown model {model!r}; known: {sorted(MODEL_IDS)}")
    family, spec = MODEL_IDS[model]
    if family == "clipseg":
        return ClipSegBackend(spec, device)
    if family == "tiles":
        return TileBackend(spec, device)
    if family == "openclip":
        return OpenClipTileBackend(spec, device)
    return Owlv2Backend(spec, device)


def load_scorer(
    model: str, grid: Grid, mission: Mission, prompt_set: str = "mission",
    calibration: ScoreCalibration | None = None, device: str | None = None, contrast: bool = True,
) -> DenseImportanceScorer:
    log.info("loading model %s (%s) for prompt set %s, contrast=%s", model, MODEL_IDS.get(model, ("?", "?"))[1], prompt_set, contrast)
    return DenseImportanceScorer(load_backend(model, device), grid, build_prompts(mission, prompt_set, contrast), calibration)
