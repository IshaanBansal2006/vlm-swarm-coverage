"""Coverage control: where a drone should go given its belief and where its peers are.

`LloydController` (decision 020) is Lloyd descent on the density-weighted Voronoi partition, the
canonical decentralized coverage controller (Cortés, Martínez, Karataş & Bullo 2004). Each drone
takes the cells nearer to it than to any peer as its region, computes that region's centroid
under its own importance belief as the density, and moves toward the centroid. The locational
cost Σ_i ∫_{V_i} ‖q − p_i‖² φ(q) dq has gradient −2 M_i (C_i − p_i) with respect to p_i, so
moving toward the centroid is gradient descent on it, and a drone at its centroid has zero
gradient and stops.

Decentralized means: own belief only, peers known only through the last positions received.

`uniform=True` makes Lloyd ignore the belief and use a constant density: the centroidal Voronoi
configuration of the area, the floor any perception has to beat. `LawnmowerController` is the
belief-free survey: each drone flies back and forth along its own horizontal strip.
`ErgodicController` is spectral multiscale coverage (Mathew & Mezic 2011): it steers the team's
time-averaged visit distribution toward the drone's own belief, matching Fourier cosine
coefficients weighted toward the coarse scales.

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
    def __init__(self, gain: float = 1.0, uniform: bool = False) -> None:
        if gain <= 0:
            raise ValueError(f"Lloyd gain must be positive, got {gain}")
        self.gain = gain
        self.uniform = uniform

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        centres = belief.grid.cell_centers()
        mask = voronoi_mask(centres, position, peers, drone_id)
        density = np.ones(belief.grid.shape) if self.uniform else belief.values
        centroid, mass = weighted_centroid(centres, density, mask)
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


class LawnmowerController:
    """A boustrophedon survey that reads nothing. The area's height is split into one horizontal
    strip per drone (drone i takes the i-th from the bottom, matching the default line-up);
    within its strip a drone flies passes along x spaced one cross-track footprint apart, turning
    where its footprint just reaches the area's edge, and runs the passes back and forth for the
    whole mission. A strip narrower than one footprint gets a single pass along its centre line.

    Commands are at `speed` toward the current waypoint, shortened on the last step so the drone
    lands on it; the waypoint advances once reached.
    """

    def __init__(self, width: float, height: float, n_drones: int, altitude: float, fov_deg: float,
                 aspect: float, speed: float, dt: float) -> None:
        if min(width, height, altitude, speed, dt) <= 0 or n_drones < 1:
            raise ValueError("lawnmower needs positive area, altitude, speed, dt and at least one drone")
        half_along = altitude * np.tan(np.radians(fov_deg) / 2)
        self.spacing = 2 * half_along / aspect
        self.speed, self.dt = speed, dt
        x_lo, x_hi = min(half_along, width / 2), max(width - half_along, width / 2)
        strip = height / n_drones
        self.routes: dict[int, NDArray[np.float64]] = {}
        for i in range(n_drones):
            lo, hi = i * strip, (i + 1) * strip
            n_pass = max(1, int(np.ceil((hi - lo) / self.spacing - 1e-9)))
            ys = np.clip((lo + hi) / 2 + (np.arange(n_pass) - (n_pass - 1) / 2) * self.spacing, lo, hi)
            pts = []
            for j, y in enumerate(ys):
                ends = (x_lo, x_hi) if j % 2 == 0 else (x_hi, x_lo)
                pts += [(ends[0], y), (ends[1], y)]
            # retrace the passes in reverse, then repeat
            pts += list(reversed(pts[1:-1]))
            self.routes[i] = np.asarray(pts, dtype=np.float64)
        self._next: dict[int, int] = {}

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        route = self.routes.get(drone_id)
        if route is None:
            raise KeyError(f"lawnmower has strips for drones {sorted(self.routes)}, not {drone_id}")
        k = self._next.get(drone_id, -1)
        if k < 0:
            k = int(np.argmin(np.linalg.norm(route[:2] - position, axis=1)))   # start at the nearer end of pass 0
        step = self.speed * self.dt
        for _ in range(len(route)):
            if np.linalg.norm(route[k] - position) > 1e-6:
                break
            k = (k + 1) % len(route)
        self._next[drone_id] = k
        delta = route[k] - position
        dist = float(np.linalg.norm(delta))
        if dist <= step:
            self._next[drone_id] = (k + 1) % len(route)
            return delta / self.dt
        return delta / dist * self.speed


def ergodic_basis(grid_or_size: tuple[float, float], modes: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wavenumbers (K, 2) as integer indices, the L2 normalisers h_k (K,), and the Sobolev weights
    Lambda_k = (1 + ||k||^2)^(-3/2) (K,) of the cosine basis on [0, L1] x [0, L2]:
    f_k(x) = cos(k1 pi x1 / L1) cos(k2 pi x2 / L2) / h_k, orthonormal on the rectangle."""
    width, height = grid_or_size
    k1, k2 = np.meshgrid(np.arange(modes), np.arange(modes), indexing="ij")
    ks = np.stack([k1.ravel(), k2.ravel()], axis=1).astype(np.float64)
    h = np.sqrt(width * height * np.where(ks[:, 0] == 0, 1.0, 0.5) * np.where(ks[:, 1] == 0, 1.0, 0.5))
    lam = (1.0 + (ks**2).sum(axis=1)) ** -1.5
    return ks, h, lam


def ergodic_eval(points: NDArray[np.float64], ks: NDArray[np.float64], h: NDArray[np.float64],
                 size: tuple[float, float]) -> NDArray[np.float64]:
    """f_k at each point: (P, K) for points (P, 2)."""
    w1, w2 = np.pi * ks[:, 0] / size[0], np.pi * ks[:, 1] / size[1]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.cos(pts[:, :1] * w1) * np.cos(pts[:, 1:] * w2) / h


def ergodic_coefficients(density: NDArray[np.float64], centres: NDArray[np.float64], ks: NDArray[np.float64],
                         h: NDArray[np.float64], size: tuple[float, float]) -> NDArray[np.float64]:
    """mu_k of a density on grid cells, normalised to a probability over the cells (a midpoint
    quadrature of the integral of mu f_k)."""
    d = np.clip(np.asarray(density, dtype=np.float64).ravel(), 0.0, None)
    total = d.sum()
    if total <= 0:
        raise ValueError("ergodic target density has no mass; the mission floor should prevent this")
    return (d / total) @ ergodic_eval(centres.reshape(-1, 2), ks, h, size)


def ergodic_metric(trajectory_coefficients: NDArray[np.float64], target: NDArray[np.float64], lam: NDArray[np.float64]) -> float:
    """Phi = sum_k Lambda_k (c_k - mu_k)^2: zero when the time average matches the target at every scale."""
    return float((lam * (trajectory_coefficients - target) ** 2).sum())


class ErgodicController:
    """Spectral multiscale coverage (Mathew & Mezic 2011, Physica D 240), decentralised.

    Each drone keeps its own copy of the team's accumulated basis values C_k = sum_j int f_k(x_j),
    adding its own position and the last received pose of every peer it has heard from, once
    per control step (a zero-order hold on peer poses between messages), and the agent-time
    those samples represent. Its target is its own fused belief, normalised, whose coefficients
    mu_k are recomputed every step. With S_k = C_k - (agent-time) mu_k and
    B = sum_k Lambda_k S_k grad f_k(x), the command is u = -u_max B / ||B||: move at full speed
    in the direction that most reduces the mismatch at the coarsest scales first.
    """

    def __init__(self, size: tuple[float, float], speed: float, dt: float, modes: int = 10) -> None:
        if speed <= 0 or dt <= 0:
            raise ValueError(f"ergodic control needs positive speed and dt, got {speed}, {dt}")
        self.size = (float(size[0]), float(size[1]))
        self.speed, self.dt = speed, dt
        self.ks, self.h, self.lam = ergodic_basis(self.size, modes)
        self._basis: NDArray[np.float64] | None = None
        self._acc: dict[int, NDArray[np.float64]] = {}
        self._agent_time: dict[int, float] = {}

    def target(self, belief: ImportanceField) -> NDArray[np.float64]:
        if self._basis is None or self._basis.shape[0] != belief.grid.n_cells:
            self._basis = ergodic_eval(belief.grid.cell_centers().reshape(-1, 2), self.ks, self.h, self.size)
        d = np.clip(belief.values.ravel(), 0.0, None)
        total = d.sum()
        if total <= 0:
            raise ValueError("ergodic target density has no mass; the mission floor should prevent this")
        return (d / total) @ self._basis

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """grad f_k at one point: (K, 2)."""
        w1, w2 = np.pi * self.ks[:, 0] / self.size[0], np.pi * self.ks[:, 1] / self.size[1]
        c1, s1 = np.cos(w1 * x[0]), np.sin(w1 * x[0])
        c2, s2 = np.cos(w2 * x[1]), np.sin(w2 * x[1])
        return np.stack([-w1 * s1 * c2, -w2 * c1 * s2], axis=1) / self.h[:, None]

    def command(self, drone_id: int, position: NDArray[np.float64], peers: dict[int, NDArray[np.float64]], belief: ImportanceField, t: float = 0.0) -> NDArray[np.float64]:
        pts = np.vstack([np.asarray(position, dtype=np.float64)[None, :2]] + [np.asarray(p, dtype=np.float64)[None, :2] for p in peers.values()])
        acc = self._acc.get(drone_id)
        visits = ergodic_eval(pts, self.ks, self.h, self.size).sum(axis=0) * self.dt
        self._acc[drone_id] = visits if acc is None else acc + visits
        self._agent_time[drone_id] = self._agent_time.get(drone_id, 0.0) + len(pts) * self.dt
        s = self._acc[drone_id] - self._agent_time[drone_id] * self.target(belief)
        b = (self.lam * s) @ self.gradient(np.asarray(position, dtype=np.float64))
        norm = float(np.linalg.norm(b))
        if norm <= 1e-12:
            return np.zeros(2)
        return -self.speed * b / norm
