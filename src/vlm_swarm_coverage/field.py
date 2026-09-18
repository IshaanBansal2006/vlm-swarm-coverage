"""The importance field: a grid of cells over the survey area, and the values that live on it.

The field is the object every component passes around. The scorer writes into it, the channel
carries it, fusion combines two of them, the controller reads it as the density φ, and the
ground truth is one of them rasterised from the scene. Keeping it a plain numpy array on a
`Grid` means each of those components is a function of arrays, testable without the rest.

Indexing convention: `values[row, col]`, row along y, col along x, matching numpy's image
convention so plots and arrays agree without a transpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vlm_swarm_coverage.scene import Feature, Scene


@dataclass(frozen=True)
class Grid:
    """Cells of edge `cell_size` tiling [0, width] x [0, height]; the far edge rounds up."""

    width: float
    height: float
    cell_size: float

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.cell_size) <= 0:
            raise ValueError(
                f"grid needs positive width, height and cell_size, got "
                f"({self.width}, {self.height}, {self.cell_size})"
            )

    @property
    def rows(self) -> int:
        return math.ceil(self.height / self.cell_size)

    @property
    def cols(self) -> int:
        return math.ceil(self.width / self.cell_size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def cell_area(self) -> float:
        return self.cell_size**2

    def cell_centers(self) -> NDArray[np.float64]:
        """(rows, cols, 2) array of (x, y) centres."""
        xs = (np.arange(self.cols) + 0.5) * self.cell_size
        ys = (np.arange(self.rows) + 0.5) * self.cell_size
        cx, cy = np.meshgrid(xs, ys)
        return np.stack([cx, cy], axis=-1)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """(row, col) containing the point; points on the far edge belong to the last cell."""
        if not (0 <= x <= self.width and 0 <= y <= self.height):
            raise ValueError(f"({x}, {y}) is outside the [0, {self.width}] x [0, {self.height}] grid")
        col = min(int(x // self.cell_size), self.cols - 1)
        row = min(int(y // self.cell_size), self.rows - 1)
        return row, col


@dataclass
class ImportanceField:
    """Values on a grid. Mutable on purpose: the scorer updates cells in place as views arrive."""

    grid: Grid
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.shape != self.grid.shape:
            raise ValueError(
                f"values shape {self.values.shape} does not match grid shape {self.grid.shape}"
            )

    @classmethod
    def uniform(cls, grid: Grid, value: float) -> ImportanceField:
        return cls(grid, np.full(grid.shape, value, dtype=np.float64))

    def copy(self) -> ImportanceField:
        return ImportanceField(self.grid, self.values.copy())

    @property
    def total_mass(self) -> float:
        return float(self.values.sum())

    def normalized(self) -> NDArray[np.float64]:
        """The field as a probability mass over cells. Undefined for an all-zero field."""
        mass = self.total_mass
        if mass <= 0:
            raise ValueError("cannot normalise a field with zero total mass; add a floor")
        return self.values / mass


def rasterize_ground_truth(scene: Scene, grid: Grid, subsamples: int = 3) -> ImportanceField:
    """The answer key: floor everywhere, plus each feature's mission weight scaled by how much of
    the cell its footprint covers, taking the strongest feature where two overlap.

    Coverage is estimated by testing `subsamples` x `subsamples` points inside each cell against
    the feature's rotated rectangle, so a feature narrower than a cell still paints the cells it
    partly occupies instead of being missed by a centre test.
    """
    mission = scene.mission
    field = ImportanceField.uniform(grid, mission.floor)
    for feature in scene.features:
        weight = mission.weight(feature.label)
        if weight <= 0:
            continue
        frac = _footprint_coverage(feature, grid, subsamples)
        field.values = np.maximum(field.values, mission.floor + weight * frac)
    return field


def _footprint_coverage(feature: Feature, grid: Grid, subsamples: int) -> NDArray[np.float64]:
    """(rows, cols) fraction of each cell's sample points lying inside the feature footprint."""
    offsets = (np.arange(subsamples) + 0.5) / subsamples * grid.cell_size
    ox, oy = np.meshgrid(offsets, offsets)
    corners = grid.cell_centers() - grid.cell_size / 2
    px = corners[..., 0, None] + ox.ravel()
    py = corners[..., 1, None] + oy.ravel()

    cx, cy = feature.position[:2]
    c, s = math.cos(feature.yaw), math.sin(feature.yaw)
    u = (px - cx) * c + (py - cy) * s
    v = -(px - cx) * s + (py - cy) * c
    half_l, half_w = feature.extent[0] / 2, feature.extent[1] / 2
    inside = (np.abs(u) <= half_l) & (np.abs(v) <= half_w)
    return inside.mean(axis=-1)
