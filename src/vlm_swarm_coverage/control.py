"""Coverage control: where a drone should go given its belief and where its peers are.

`LloydController` (decision 020) is Lloyd descent on the density-weighted Voronoi partition, the
canonical decentralized coverage controller (Cortés, Martínez, Karataş & Bullo 2004). Each drone
takes the cells nearer to it than to any peer as its region, computes that region's centroid
under its own importance belief as the density, and moves toward the centroid. The locational
cost Σ_i ∫_{V_i} ‖q − p_i‖² φ(q) dq has gradient −2 M_i (C_i − p_i) with respect to p_i, so
moving toward the centroid is gradient descent on it, and a drone at its centroid has zero
gradient and stops.

Decentralized means: own belief only, peers known only through the last positions received.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vlm_swarm_coverage.field import ImportanceField


class CoverageController(Protocol):
    def command(
        self,
        drone_id: int,
        position: NDArray[np.float64],
        peers: dict[int, NDArray[np.float64]],
        belief: ImportanceField,
    ) -> NDArray[np.float64]:
        """Return a planar velocity command (2,) for this drone."""
        ...


def voronoi_mask(grid_centres: NDArray[np.float64], own: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], own_id: int) -> NDArray[np.bool_]:
    """(rows, cols) True where the cell centre is nearer to `own` than to every peer.

    Exact ties go to the lower drone id so two drones never both own a cell.
    """
    d_own = np.linalg.norm(grid_centres - own, axis=-1)
    mask = np.ones(d_own.shape, dtype=bool)
    for pid, p in peers.items():
        d_peer = np.linalg.norm(grid_centres - p, axis=-1)
        mask &= (d_own < d_peer) | ((d_own == d_peer) & (own_id < pid))
    return mask


def weighted_centroid(grid_centres: NDArray[np.float64], density: NDArray[np.float64], mask: NDArray[np.bool_]) -> tuple[NDArray[np.float64], float]:
    """Centroid of the masked cells under `density`, and the mass. Zero mass returns (nan, 0)."""
    w = np.where(mask, density, 0.0)
    mass = float(w.sum())
    if mass <= 0.0:
        return np.full(2, np.nan), 0.0
    centroid = (w[..., None] * grid_centres).sum(axis=(0, 1)) / mass
    return centroid, mass


class LloydController:
    def __init__(self, gain: float = 1.0) -> None:
        if gain <= 0:
            raise ValueError(f"Lloyd gain must be positive, got {gain}")
        self.gain = gain

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField) -> NDArray[np.float64]:
        centres = belief.grid.cell_centers()
        mask = voronoi_mask(centres, position, peers, drone_id)
        centroid, mass = weighted_centroid(centres, belief.values, mask)
        if mass <= 0.0:
            return np.zeros(2)
        return self.gain * (centroid - position)


class HoldController:
    """Stay put. The zero-motion reference and the stand-in when no controller is wanted."""

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField) -> NDArray[np.float64]:
        return np.zeros(2)
