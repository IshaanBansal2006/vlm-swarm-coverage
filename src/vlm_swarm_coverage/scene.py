"""The world the drones fly over, described once and shared by everything that needs it.

The Isaac scene script renders this description, the oracle scorer reads ground-truth importance
from it, the plotting code draws it, and the tests assert against it. One source of truth means
the field the drones are scored against is the field that was actually rendered.

Pure dataclasses + numpy on purpose: the Isaac scene script runs under the simulator's bundled
Python, which does not have this package installed, and imports this module by path.

World frame: z up, ground at z = 0, the survey area is the rectangle [0, width] x [0, height]
(areas start at 0 so no coordinate is ever a negative command-line token). A road runs along +x
through the middle of the area. Features sit on or beside the road: some are what the mission is
looking for, the rest are distractors that look plausible from the air but carry no importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Feature footprints (m) as [length, width, height] in the feature's own frame. Targets are the
# things a storm-damage survey is looking for; distractors are ordinary roadside objects.
FEATURE_EXTENTS: dict[str, tuple[float, float, float]] = {
    "damaged_road": (6.0, 6.0, 0.05),
    "debris": (2.5, 2.5, 0.8),
    "stalled_vehicle": (4.6, 1.8, 1.45),
    "parked_vehicle": (4.6, 1.8, 1.45),
    "tree": (2.5, 2.5, 5.0),
    "shed": (4.0, 3.0, 2.5),
}

TARGET_LABELS: frozenset[str] = frozenset({"damaged_road", "debris", "stalled_vehicle"})
DISTRACTOR_LABELS: frozenset[str] = frozenset(FEATURE_EXTENTS) - TARGET_LABELS


@dataclass(frozen=True)
class Feature:
    feature_id: str
    label: str
    position: tuple[float, float, float]
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if self.label not in FEATURE_EXTENTS:
            raise ValueError(
                f"unknown feature label {self.label!r}; add it to FEATURE_EXTENTS or use one of "
                f"{sorted(FEATURE_EXTENTS)}"
            )

    @property
    def extent(self) -> NDArray[np.float64]:
        return np.asarray(FEATURE_EXTENTS[self.label], dtype=float)

    @property
    def footprint(self) -> NDArray[np.float64]:
        """Ground-plane corners (4, 2) of the feature's box, rotated by yaw about its centre."""
        half_l, half_w = self.extent[0] / 2, self.extent[1] / 2
        corners = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rot = np.array([[c, -s], [s, c]])
        return corners @ rot.T + np.asarray(self.position[:2])


@dataclass(frozen=True)
class Mission:
    """What the swarm is looking for, in words for the model and in weights for the ground truth.

    `weights` maps a feature label to its importance. Labels not listed weigh zero. `floor` is a
    uniform background importance added everywhere so the field is never all-zero: a coverage
    controller driven by a zero density has no gradient anywhere and simply stops.
    """

    text: str
    weights: dict[str, float]
    floor: float = 0.05
    phrases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(FEATURE_EXTENTS)
        if unknown:
            raise ValueError(f"mission weights name unknown labels {sorted(unknown)}")
        if self.floor < 0:
            raise ValueError(f"mission floor must be >= 0, got {self.floor}")
        unknown = set(self.phrases) - set(FEATURE_EXTENTS)
        if unknown:
            raise ValueError(f"mission phrases name unknown labels {sorted(unknown)}")

    def weight(self, label: str) -> float:
        return self.weights.get(label, 0.0)

    def phrase(self, label: str) -> str:
        """What a model is asked to look for, for one label; the label's words by default."""
        return self.phrases.get(label, label.replace("_", " "))


@dataclass(frozen=True)
class Road:
    y_center: float
    width: float


@dataclass(frozen=True)
class Scene:
    width: float
    height: float
    road: Road
    mission: Mission
    features: tuple[Feature, ...] = ()
    drone_starts: tuple[tuple[float, float, float], ...] = ()
    seed: int | None = field(default=None)

    def __post_init__(self) -> None:
        ids = [f.feature_id for f in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate feature ids in scene: {sorted(set(i for i in ids if ids.count(i) > 1))}")
        for f in self.features:
            x, y = f.position[:2]
            if not (0 <= x <= self.width and 0 <= y <= self.height):
                raise ValueError(
                    f"feature {f.feature_id!r} at ({x}, {y}) lies outside the "
                    f"[0, {self.width}] x [0, {self.height}] survey area"
                )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, self.width, 0.0, self.height)

    def targets(self) -> tuple[Feature, ...]:
        return tuple(f for f in self.features if f.label in TARGET_LABELS)

    def distractors(self) -> tuple[Feature, ...]:
        return tuple(f for f in self.features if f.label in DISTRACTOR_LABELS)


STORM_DAMAGE_MISSION = Mission(
    text=(
        "Survey the road corridor after a storm. Find washed-out or cracked road surface, "
        "debris blocking the road, and vehicles that are stranded or abandoned."
    ),
    weights={"damaged_road": 1.0, "debris": 0.8, "stalled_vehicle": 0.9},
    phrases={
        "damaged_road": "washed-out or cracked road surface",
        "debris": "debris blocking the road",
        "stalled_vehicle": "a stranded or abandoned vehicle",
    },
)


def default_drone_starts(n_drones: int, width: float, height: float, altitude: float) -> tuple[tuple[float, float, float], ...]:
    """Line the swarm up along the x = 0 edge, evenly spaced in y, all at the same altitude."""
    if n_drones < 1:
        raise ValueError(f"n_drones must be >= 1, got {n_drones}")
    ys = np.linspace(0.2 * height, 0.8 * height, n_drones) if n_drones > 1 else np.array([height / 2])
    return tuple((2.0, float(y), altitude) for y in ys)


def default_scene(n_drones: int = 3, altitude: float = 12.0) -> Scene:
    """The fixed demo layout: a 60 x 40 m corridor with three damage sites and four distractors."""
    width, height = 60.0, 40.0
    road = Road(y_center=height / 2, width=6.0)
    y = road.y_center
    features = (
        Feature("damage_0", "damaged_road", (18.0, y, 0.0)),
        Feature("debris_0", "debris", (33.0, y + 1.0, 0.0), yaw=0.4),
        Feature("stalled_0", "stalled_vehicle", (47.0, y - 1.5, 0.0), yaw=0.15),
        Feature("parked_0", "parked_vehicle", (10.0, y + 4.5, 0.0)),
        Feature("tree_0", "tree", (26.0, y - 9.0, 0.0)),
        Feature("tree_1", "tree", (40.0, y + 10.0, 0.0)),
        Feature("shed_0", "shed", (52.0, y + 8.0, 0.0), yaw=0.3),
    )
    return Scene(
        width=width,
        height=height,
        road=road,
        mission=STORM_DAMAGE_MISSION,
        features=features,
        drone_starts=default_drone_starts(n_drones, width, height, altitude),
    )


def random_scene(
    seed: int,
    n_targets: int = 3,
    n_distractors: int = 4,
    width: float = 60.0,
    height: float = 40.0,
    n_drones: int = 3,
    altitude: float = 12.0,
) -> Scene:
    """A seeded layout for sweeps: targets on the road, distractors on the verges.

    Same seed, same scene, every time. Targets are spread along x so two never overlap; distractors
    are kept off the road so a distractor never sits on top of a target.
    """
    rng = np.random.default_rng(seed)
    road = Road(y_center=height / 2, width=6.0)
    features = list(_place_targets(rng, n_targets, width, road))
    features += _place_distractors(rng, n_distractors, width, height, road)
    return Scene(
        width=width,
        height=height,
        road=road,
        mission=STORM_DAMAGE_MISSION,
        features=tuple(features),
        drone_starts=default_drone_starts(n_drones, width, height, altitude),
        seed=seed,
    )


def _place_targets(rng: np.random.Generator, n: int, width: float, road: Road) -> list[Feature]:
    labels = sorted(TARGET_LABELS)
    margin = 8.0
    xs = np.linspace(margin, width - margin, n) + rng.uniform(-2.0, 2.0, size=n)
    out = []
    for i, x in enumerate(xs):
        label = labels[rng.integers(len(labels))]
        y = road.y_center + rng.uniform(-road.width / 4, road.width / 4)
        out.append(Feature(f"target_{i}", label, (float(x), float(y), 0.0), yaw=float(rng.uniform(-0.5, 0.5))))
    return out


def _place_distractors(
    rng: np.random.Generator, n: int, width: float, height: float, road: Road
) -> list[Feature]:
    labels = sorted(DISTRACTOR_LABELS)
    verge = road.width / 2 + 2.0
    out = []
    for i in range(n):
        label = labels[rng.integers(len(labels))]
        x = float(rng.uniform(4.0, width - 4.0))
        side = 1.0 if rng.random() < 0.5 else -1.0
        y = road.y_center + side * rng.uniform(verge, min(height / 2 - 2.0, verge + 10.0))
        out.append(Feature(f"distractor_{i}", label, (x, float(y), 0.0), yaw=float(rng.uniform(-0.5, 0.5))))
    return out
