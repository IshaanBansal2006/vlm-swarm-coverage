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
    "owlv2-density": ("owlv2-density", "google/owlv2-base-patch16-ensemble"),
    "siglip2-dense": ("siglip-patches", "google/siglip2-base-patch16-384"),
    "remoteclip-l14-dense": ("openclip-patches", "ViT-L-14|chendelong/RemoteCLIP|RemoteCLIP-ViT-L-14.pt"),
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
    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """(k, h, w) per-phrase logits for the frame; h, w need not equal the frame's."""
        ...

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """(h, w) map of importance: the phrase logits reduced by `combine`."""
        ...


class Backend:
    """What every backend shares: a map is its per-phrase logits, combined. Splitting the two
    lets a caller read the distribution over phrases rather than only the winner, which is what
    a model of *what the model confuses with what* needs."""

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        raise NotImplementedError

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        return combine(self.phrase_logits(frame, prompts), prompts)

    def phrase_probabilities(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """(k, h, w) probabilities: the same softmax or sigmoid `combine` applies, without the
        weighting and the maximum, so every phrase's share is visible."""
        return probabilities(self.phrase_logits(frame, prompts), prompts)


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _features(out):  # type: ignore[no-untyped-def]
    """transformers returns a tensor or an output object depending on version; take the tensor."""
    return getattr(out, "pooler_output", None) if hasattr(out, "pooler_output") else out


def probabilities(logits: NDArray[np.float64], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
    """(k, h, w) logits -> (k, h, w) probabilities: softmax across phrases when any background
    phrase is present (the contrast recipe of decision 016), else each phrase's own sigmoid."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape[0] != len(prompts):
        raise ValueError(f"{logits.shape[0]} logit maps for {len(prompts)} prompts")
    if any(not p.is_target for p in prompts):
        shifted = logits - logits.max(axis=0, keepdims=True)
        return np.exp(shifted) / np.exp(shifted).sum(axis=0, keepdims=True)
    return 1.0 / (1.0 + np.exp(-logits))


def combine(logits: NDArray[np.float64], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
    """(k, h, w) per-phrase logits -> (h, w) importance in [0, 1]: each target phrase's probability
    times its weight, keeping the strongest, mirroring the answer key's rule that a cell takes its
    most important feature. With any background phrase present the probabilities are a softmax
    across all phrases (contrast); without, each phrase's own sigmoid."""
    probs = probabilities(logits, prompts)
    weights = np.array([p.weight for p in prompts])[:, None, None]
    return (probs * weights).max(axis=0)


class ClipSegBackend(Backend):
    """CLIPSeg: one 352x352 logit map per phrase, sigmoid to [0, 1]."""

    def __init__(self, hf_id: str = MODEL_IDS["clipseg"][1], device: str | None = None) -> None:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.device = _device(device)
        self.processor = CLIPSegProcessor.from_pretrained(hf_id)
        self.model = CLIPSegForImageSegmentation.from_pretrained(hf_id).to(self.device).eval()
        self._torch = torch
        self._text_inputs: dict[tuple[str, ...], object] = {}

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """The image is preprocessed once and repeated per phrase (the processor would resize and
        normalise k identical copies on the CPU); the tokenised phrases are cached."""
        from PIL import Image

        image = Image.fromarray(np.asarray(frame))
        texts = tuple(p.phrase for p in prompts)
        if texts not in self._text_inputs:
            self._text_inputs[texts] = self.processor.tokenizer(list(texts), padding=True, return_tensors="pt").to(self.device)
        pixels = self.processor.image_processor(images=[image], return_tensors="pt")["pixel_values"].to(self.device)
        with self._torch.inference_mode():
            logits = self.model(**self._text_inputs[texts], pixel_values=pixels.repeat(len(texts), 1, 1, 1)).logits
        return logits.float().cpu().numpy().reshape(len(texts), *logits.shape[-2:])


def tile_boxes(height: int, width: int, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    ys = np.linspace(0, height, rows + 1).astype(int)
    xs = np.linspace(0, width, cols + 1).astype(int)
    return [(ys[r], ys[r + 1], xs[c], xs[c + 1]) for r in range(rows) for c in range(cols)]


class TileBackend(Backend):
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

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
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
            return cos.T.detach().float().cpu().numpy().reshape(len(prompts), rows, cols)


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

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        from PIL import Image

        frame = np.asarray(frame)
        rows, cols = self.tiles
        crops = [self.preprocess(Image.fromarray(frame[y0:y1, x0:x1])) for y0, y1, x0, x1 in tile_boxes(frame.shape[0], frame.shape[1], rows, cols)]
        batch = self._torch.stack(crops).to(self.device)
        with self._torch.inference_mode():
            img = self.model.encode_image(batch)
        img = img / img.norm(dim=-1, keepdim=True)
        logits = (img @ self._text(tuple(p.phrase for p in prompts)).T).T * CLIP_LOGIT_SCALE
        return logits.float().cpu().numpy().reshape(len(prompts), rows, cols)


class Owlv2Backend(Backend):
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

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """A detector has no logit field. Each target phrase gets a map of its own detections,
        as a logit so `combine`'s sigmoid returns the detection confidence; background phrases
        get a constant that leaves them at the softmax's mercy only where nothing was found."""
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
        conf = np.zeros((len(prompts), *frame.shape[:2]))
        index = {id(p): i for i, p in enumerate(prompts)}
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"], strict=True):
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            x0, x1 = max(0, x0), min(frame.shape[1], x1)
            y0, y1 = max(0, y0), min(frame.shape[0], y1)
            if x1 > x0 and y1 > y0:
                k = index[id(targets[int(label)])]
                conf[k, y0:y1, x0:x1] = np.maximum(conf[k, y0:y1, x0:x1], float(score))
        eps = 1e-6
        return np.log(np.clip(conf, eps, 1 - eps) / (1 - np.clip(conf, eps, 1 - eps)))

    def importance_map(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        """Detections are already per-phrase confidences, so they combine through their own
        sigmoid. Passing them through the contrast softmax would hand every undetected cell a
        share of the probability mass, which for a detector means inventing importance."""
        targets = [p for p in prompts if p.is_target]
        logits = self.phrase_logits(frame, prompts)
        keep = [i for i, p in enumerate(prompts) if p.is_target]
        return combine(logits[keep], targets)


def gaussian_splat_max(boxes: NDArray[np.float64], scores: NDArray[np.float64], height: int, width: int,
                       sigma_frac: float = 0.25) -> NDArray[np.float64]:
    """(n, 4) boxes (x0, y0, x1, y1, in map pixels) with scores -> (height, width) density: each box
    is a Gaussian of peak `score`, centred on the box, with sigma = `sigma_frac` of its width and
    height (so the box edge sits at 2 sigma); overlaps take the maximum. The maximum keeps the
    value in the detector's own confidence units whatever the number of overlapping boxes: OWLv2
    predicts one box per image patch, so a large object carries many near-duplicate boxes and a
    sum would grow with object size and duplicate count rather than with confidence."""
    out = np.zeros((height, width))
    if len(boxes) == 0:
        return out
    xs, ys = np.arange(width) + 0.5, np.arange(height) + 0.5
    for (x0, y0, x1, y1), s in zip(boxes, scores, strict=True):
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        sx, sy = max(sigma_frac * (x1 - x0), 0.5), max(sigma_frac * (y1 - y0), 0.5)
        u0, u1 = max(0, int(cx - 3 * sx)), min(width, int(np.ceil(cx + 3 * sx)) + 1)
        v0, v1 = max(0, int(cy - 3 * sy)), min(height, int(np.ceil(cy + 3 * sy)) + 1)
        if u1 <= u0 or v1 <= v0:
            continue
        gx = np.exp(-0.5 * ((xs[u0:u1] - cx) / sx) ** 2)
        gy = np.exp(-0.5 * ((ys[v0:v1] - cy) / sy) ** 2)
        np.maximum(out[v0:v1, u0:u1], float(s) * np.outer(gy, gx), out=out[v0:v1, u0:u1])
    return out


class Owlv2DensityBackend(Owlv2Backend):
    """OWLv2 as a density rather than painted boxes: every predicted box above a low threshold
    (default 0.01, so weak evidence is kept rather than cut to zero) and no larger than
    `max_area_frac` of the frame is splatted as a
    score-weighted Gaussian (`gaussian_splat_max`), per phrase, on a map at `scale` of the frame's
    resolution. A pixel then takes the strongest weighted phrase, as `combine` does for the other
    models, without the background-phrase softmax a detector has no use for."""

    def __init__(self, hf_id: str = MODEL_IDS["owlv2"][1], device: str | None = None, threshold: float = 0.01,
                 scale: float = 0.25, max_area_frac: float = 0.3) -> None:
        super().__init__(hf_id, device, threshold)
        self.scale = scale
        self.max_area_frac = max_area_frac

    def target_density(self, frame: NDArray[np.uint8], targets: Sequence[Prompt]) -> NDArray[np.float64]:
        """(k_targets, h, w) Gaussian densities in [0, 1], one per target phrase."""
        from PIL import Image

        frame = np.asarray(frame)
        image = Image.fromarray(frame)
        inputs = self.processor(text=[[p.phrase for p in targets]], images=image, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(image.size)
        res = self.processor.post_process_grounded_object_detection(outputs, threshold=self.threshold, target_sizes=[(side, side)])[0]
        h, w = max(1, round(frame.shape[0] * self.scale)), max(1, round(frame.shape[1] * self.scale))
        boxes = res["boxes"].float().cpu().numpy()
        scores, labels = res["scores"].float().cpu().numpy(), res["labels"].cpu().numpy()
        # boxes spanning most of the frame are OWLv2 describing the scene, not an object (the largest
        # feature, a 6 m damaged road, fills ~17 % of a 12 m nadir frame); they would paint the whole
        # map at the scene score
        area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
        keep = area <= self.max_area_frac * frame.shape[0] * frame.shape[1]
        boxes, scores, labels = boxes[keep] * self.scale, scores[keep], labels[keep]
        return np.stack([gaussian_splat_max(boxes[labels == k], scores[labels == k], h, w) for k in range(len(targets))])

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        targets = [p for p in prompts if p.is_target]
        dens = self.target_density(frame, targets)
        out = np.zeros((len(prompts), *dens.shape[1:]))
        out[[i for i, p in enumerate(prompts) if p.is_target]] = dens
        eps = 1e-6
        return np.log(np.clip(out, eps, 1 - eps) / (1 - np.clip(out, eps, 1 - eps)))


def _upsample(maps, height: int, width: int) -> NDArray[np.float64]:  # type: ignore[no-untyped-def]
    """(k, h, w) logits, a tensor on any device or an array -> (k, height, width), bilinear."""
    import torch

    t = maps if isinstance(maps, torch.Tensor) else torch.as_tensor(np.asarray(maps, dtype=np.float32))
    out = torch.nn.functional.interpolate(t.float()[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return out.cpu().numpy().astype(np.float64)


class SiglipPatchBackend(TileBackend):
    """Dense SigLIP: every patch token of one forward pass, compared with the phrase embeddings.

    SigLIP pools patches with an attention head (a learned probe attending over the tokens), so a
    patch token is not in the text-aligned space until it goes through that head. Following
    MaskCLIP, each token takes the head's value path alone (value projection, output projection),
    which is what the probe's attention would return if it attended to that one patch, then the
    head's layer norm + MLP residual as the pooled output does. Cosines are scaled and biased with
    the model's own logit scale and bias, and the patch grid is upsampled bilinearly to the frame
    (the frame is resized square to the model's input, so the grid spans the whole frame)."""

    def __init__(self, hf_id: str = MODEL_IDS["siglip2-dense"][1], device: str | None = None, mlp: bool = True) -> None:
        super().__init__(hf_id, device)
        self.mlp = mlp

    def patch_embeddings(self, frame: NDArray[np.uint8]):  # type: ignore[no-untyped-def]
        """(grid_h, grid_w, d) unit-norm patch embeddings in the text-aligned space."""
        from PIL import Image

        vm = self.model.vision_model
        inputs = self.processor(images=[Image.fromarray(np.asarray(frame))], return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            tokens = vm(pixel_values=inputs["pixel_values"]).last_hidden_state[0]   # post-layernorm (n, d)
            att = vm.head.attention
            d = tokens.shape[-1]
            v = tokens @ att.in_proj_weight[2 * d:].T + att.in_proj_bias[2 * d:]
            h = att.out_proj(v)
            if self.mlp:
                h = h + vm.head.mlp(vm.head.layernorm(h))
            h = h / h.norm(dim=-1, keepdim=True)
        side = int(round(math.sqrt(h.shape[0])))
        return h.reshape(side, side, -1)

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        emb = self.patch_embeddings(frame)
        text = self._text(tuple(p.phrase for p in prompts))
        with self._torch.inference_mode():
            cos = (emb @ text.T).permute(2, 0, 1)
            logits = cos * self.model.logit_scale.exp() + self.model.logit_bias
        frame = np.asarray(frame)
        return _upsample(logits, frame.shape[0], frame.shape[1])


class OpenClipPatchBackend(OpenClipTileBackend):
    """Dense CLIP (open_clip checkpoints such as RemoteCLIP) from patch tokens, ClearCLIP-style:
    the last transformer block's attention is replaced by its value path alone, with the residual
    and the MLP dropped, then the visual head's layer norm and projection map each patch into the
    text space. Plain CLIP patch tokens are dominated by global context; the value path is what
    keeps them local. Logits are 100 x cosine, as for the tile backend."""

    def _square(self, image):  # type: ignore[no-untyped-def]
        """The whole frame resized to the model's square input (open_clip's own preprocess centre-
        crops, which would drop the frame's sides and misplace every patch), then normalised."""
        size = self.model.visual.image_size
        size = size if isinstance(size, (tuple, list)) else (size, size)
        norm = next(t for t in self.preprocess.transforms if type(t).__name__ == "Normalize")
        a = np.asarray(image.convert("RGB").resize((int(size[1]), int(size[0])), resample=3), dtype=np.float32) / 255.0
        t = self._torch.as_tensor(a).permute(2, 0, 1)
        mean = self._torch.tensor(norm.mean).view(3, 1, 1)
        std = self._torch.tensor(norm.std).view(3, 1, 1)
        return ((t - mean) / std)[None].to(self.device)

    def patch_embeddings(self, frame: NDArray[np.uint8]):  # type: ignore[no-untyped-def]
        from PIL import Image

        visual = self.model.visual
        x = self._square(Image.fromarray(np.asarray(frame)))
        with self._torch.inference_mode():
            x = visual.conv1(x)
            grid = x.shape[-2:]
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype) + self._torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = self._torch.cat([cls, x], dim=1) + visual.positional_embedding.to(x.dtype)
            x = visual.patch_dropout(x)
            x = visual.ln_pre(x)
            blocks = visual.transformer.resblocks
            for blk in blocks[:-1]:
                x = blk(x)
            last = blocks[-1]
            y = last.ln_1(x)
            attn = last.attn
            d = y.shape[-1]
            v = y @ attn.in_proj_weight[2 * d:].T + attn.in_proj_bias[2 * d:]
            y = attn.out_proj(v)
            y = visual.ln_post(y[:, 1:]) @ visual.proj
            y = y / y.norm(dim=-1, keepdim=True)
        return y[0].reshape(int(grid[0]), int(grid[1]), -1)

    def phrase_logits(self, frame: NDArray[np.uint8], prompts: Sequence[Prompt]) -> NDArray[np.float64]:
        emb = self.patch_embeddings(frame)
        text = self._text(tuple(p.phrase for p in prompts))
        with self._torch.inference_mode():
            logits = (emb @ text.T).permute(2, 0, 1) * CLIP_LOGIT_SCALE
        frame = np.asarray(frame)
        return _upsample(logits, frame.shape[0], frame.shape[1])


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
    if family == "owlv2":
        return Owlv2Backend(spec, device)
    if family == "owlv2-density":
        return Owlv2DensityBackend(spec, device)
    if family == "siglip-patches":
        return SiglipPatchBackend(spec, device)
    if family == "openclip-patches":
        return OpenClipPatchBackend(spec, device)
    raise ValueError(f"model {model!r} has family {family!r}, which load_backend does not know")


def load_scorer(
    model: str, grid: Grid, mission: Mission, prompt_set: str = "mission",
    calibration: ScoreCalibration | None = None, device: str | None = None, contrast: bool = True,
) -> DenseImportanceScorer:
    log.info("loading model %s (%s) for prompt set %s, contrast=%s", model, MODEL_IDS.get(model, ("?", "?"))[1], prompt_set, contrast)
    return DenseImportanceScorer(load_backend(model, device), grid, build_prompts(mission, prompt_set, contrast), calibration)
