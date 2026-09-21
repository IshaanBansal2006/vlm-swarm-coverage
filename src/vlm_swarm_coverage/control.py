"""Coverage control: where a drone should go given its belief and where its peers are.

`LloydController` (decision 020) is Lloyd descent on the density-weighted Voronoi partition, the
canonical decentralized coverage controller (Cortés, Martínez, Karataş & Bullo 2004). Each drone
takes the cells nearer to it than to any peer as its region, computes that region's centroid
under its own importance belief as the density, and moves toward the centroid. The locational
cost Σ_i ∫_{V_i} ‖q − p_i‖² φ(q) dq has gradient −2 M_i (C_i − p_i) with respect to p_i, so
moving toward the centroid is gradient descent on it, and a drone at its centroid has zero
gradient and stops.

Decentralized means: own belief only, peers known only through the last positions received.

`ReplayController` (decision 023) is the opposite of a controller: it reproduces a logged run's
trajectory step for step, ignoring belief and peers, so everything downstream of motion can be
changed while motion stays fixed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vlm_swarm_coverage.artifacts import RunDir
    from vlm_swarm_coverage.field import ImportanceField


class CoverageController(Protocol):
    def command(
        self,
        drone_id: int,
        position: NDArray[np.float64],
        peers: dict[int, NDArray[np.float64]],
        belief: ImportanceField,
        t: float = 0.0,
    ) -> NDArray[np.float64]:
        """Return a planar velocity command (2,) for this drone at sim time `t`."""
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

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        centres = belief.grid.cell_centers()
        mask = voronoi_mask(centres, position, peers, drone_id)
        centroid, mass = weighted_centroid(centres, belief.values, mask)
        if mass <= 0.0:
            return np.zeros(2)
        return self.gain * (centroid - position)


class HoldController:
    """Stay put. The zero-motion reference and the stand-in when no controller is wanted."""

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        return np.zeros(2)


class ReplayController:
    """Reproduce a logged trajectory. At time t the command is the velocity that carries the drone
    from where it is to where the source run had it at t + dt, so a drone that starts where the
    source started retraces the source exactly, and one that drifts is pulled back.

    `trajectories` maps drone id to an (S, 2) array of positions at successive steps, the first
    row being the position after the first step. Beyond the last logged step the drone holds.
    """

    def __init__(self, trajectories: dict[int, NDArray[np.float64]], dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not trajectories:
            raise ValueError("ReplayController needs at least one trajectory")
        self.trajectories = {i: np.asarray(p, dtype=np.float64).reshape(-1, 2) for i, p in trajectories.items()}
        self.dt = dt

    @classmethod
    def from_run(cls, run_dir: RunDir) -> ReplayController:
        """Trajectories from a run's `pose` events, one row per step in time order."""
        by_drone: dict[int, list[tuple[float, float, float]]] = {}
        for e in run_dir.iter_events("pose"):
            by_drone.setdefault(e["drone"], []).append((e["t"], e["x"], e["y"]))
        if not by_drone:
            raise ValueError(f"{run_dir.path} has no pose events to replay")
        traj = {i: np.array([[x, y] for _, x, y in sorted(rows)]) for i, rows in by_drone.items()}
        return cls(traj, run_dir.config.sim.dt)

    @property
    def n_drones(self) -> int:
        return len(self.trajectories)

    @property
    def n_steps(self) -> int:
        return min(len(p) for p in self.trajectories.values())

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        path = self.trajectories.get(drone_id)
        if path is None:
            raise KeyError(f"no replay trajectory for drone {drone_id}; the source run had drones {sorted(self.trajectories)}")
        k = int(round(t / self.dt))
        target = path[min(k, len(path) - 1)]
        return (target - position) / self.dt
