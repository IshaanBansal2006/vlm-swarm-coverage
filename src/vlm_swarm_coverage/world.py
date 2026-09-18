"""Drone motion: first-order kinematics at fixed altitude (decision 021).

The drones are velocity-commanded points. Each step, the commanded velocity is clipped to the
speed limit, the position advances by v·dt, and the position is clamped to the survey area. Yaw
follows the direction of travel so a body-fixed nadir camera turns with the drone. Nothing here
is aerodynamic: vehicle dynamics are not a variable in this study, and the simulator renders
whatever pose it is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class DroneState:
    drone_id: int
    position: NDArray[np.float64]
    yaw: float = 0.0
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))

    @property
    def xy(self) -> NDArray[np.float64]:
        return self.position[:2]

    @property
    def z(self) -> float:
        return float(self.position[2])


class KinematicWorld:
    """N drones on a rectangle. `step` applies one control interval to all of them."""

    def __init__(self, starts: tuple[tuple[float, float, float], ...], width: float, height: float, max_speed: float, dt: float) -> None:
        if not starts:
            raise ValueError("KinematicWorld needs at least one start pose")
        self.width, self.height = width, height
        self.max_speed, self.dt = max_speed, dt
        self.drones = [DroneState(i, np.asarray(p, dtype=np.float64)) for i, p in enumerate(starts)]
        self.t = 0.0

    def step(self, commands: dict[int, NDArray[np.float64]]) -> None:
        for d in self.drones:
            v = np.asarray(commands.get(d.drone_id, np.zeros(2)), dtype=np.float64)
            speed = float(np.linalg.norm(v))
            if speed > self.max_speed:
                v = v * (self.max_speed / speed)
            d.velocity = v
            if speed > 1e-9:
                d.yaw = float(np.arctan2(v[1], v[0]))
            d.position[0] = float(np.clip(d.position[0] + v[0] * self.dt, 0.0, self.width))
            d.position[1] = float(np.clip(d.position[1] + v[1] * self.dt, 0.0, self.height))
        self.t += self.dt

    def positions(self) -> dict[int, NDArray[np.float64]]:
        return {d.drone_id: d.xy.copy() for d in self.drones}
