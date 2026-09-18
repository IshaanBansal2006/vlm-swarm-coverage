"""The WSL side of the simulator bridge (decision 060).

The loop owns motion. Each step it publishes every drone's pose to the renderer over ROS 2 and
paces itself to wall time so the rendered run plays at real speed. When a scorer runs on the
renderer's host, `RemoteScorer` receives its observations over the same bridge and hands the
latest one for each drone to the loop in place of a local scorer.

Topics (std_msgs/String JSON for poses, matching the predecessor's renderer; UInt8MultiArray of
the binary `ObservationMessage` for scores):

    /drone_poses    loop -> renderer   {"t": .., "poses": [{"drone_id", "position", "orientation"}]}
    /importance     renderer -> loop   ObservationMessage bytes

`rclpy` is imported inside the classes so this module imports, and its tests run, with no ROS.
Run the loop with the ROS interpreter: `source scripts/ros-env.sh` then `python3 -m ...`.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from vlm_swarm_coverage.schemas import ObservationMessage
from vlm_swarm_coverage.scoring import Observation

if TYPE_CHECKING:
    from vlm_swarm_coverage.scoring import View
    from vlm_swarm_coverage.simulation import Simulation
    from vlm_swarm_coverage.world import KinematicWorld

log = logging.getLogger(__name__)

POSES_TOPIC = "/drone_poses"
IMPORTANCE_TOPIC = "/importance"


def yaw_quat(yaw: float) -> list[float]:
    """[w, x, y, z] for a rotation about z."""
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def pose_payload(world: KinematicWorld) -> str:
    poses = [
        {"drone_id": d.drone_id, "position": [float(v) for v in d.position], "orientation": yaw_quat(d.yaw)}
        for d in world.drones
    ]
    return json.dumps({"t": world.t, "poses": poses}, separators=(",", ":"))


class IsaacBridge:
    """Publishes poses each step and keeps the loop at wall-clock speed.

    Use as the `on_step` of a `Simulation`. `realtime=False` publishes without pacing, for a
    renderer that only needs the poses and not the timing.
    """

    def __init__(self, node_name: str = "vsc_loop", realtime: bool = True) -> None:
        import rclpy
        from std_msgs.msg import String, UInt8MultiArray

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = rclpy.create_node(node_name)
        self._pub = self.node.create_publisher(String, POSES_TOPIC, 10)
        self._String = String
        self._sub = self.node.create_subscription(UInt8MultiArray, IMPORTANCE_TOPIC, self._on_importance, 50)
        self.latest: dict[int, ObservationMessage] = {}
        self.received = 0
        self.realtime = realtime
        self._t0: float | None = None

    def _on_importance(self, msg: Any) -> None:
        try:
            obs = ObservationMessage.from_bytes(bytes(msg.data))
        except ValueError as exc:
            log.warning("dropping undecodable importance message: %s", exc)
            return
        self.latest[obs.sender] = obs
        self.received += 1

    def spin(self) -> None:
        for _ in range(50):
            before = self.received
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            if self.received == before:
                break

    def __call__(self, sim: Simulation, t: float) -> None:
        msg = self._String()
        msg.data = pose_payload(sim.world)
        self._pub.publish(msg)
        self.spin()
        if self.realtime:
            if self._t0 is None:
                self._t0 = time.perf_counter() - t
            lag = t - (time.perf_counter() - self._t0)
            if lag > 0:
                time.sleep(lag)

    def close(self) -> None:
        self.node.destroy_node()


class RemoteScorer:
    """Serve the newest observation the bridge has received for the drone in the view.

    Before the first observation for a drone arrives, returns an empty observation, so the belief
    stays at its prior rather than the loop stalling. The `t` in the returned observation is the
    view's time, not the renderer's, so the log stays on the loop's clock.
    """

    def __init__(self, bridge: IsaacBridge) -> None:
        self.bridge = bridge
        self.served = 0
        self.empty = 0

    def score(self, view: View) -> Observation:
        self.bridge.spin()
        msg = self.bridge.latest.get(view.drone_id)
        if msg is None:
            self.empty += 1
            return Observation(view.drone_id, view.t, np.zeros((0, 2), dtype=np.int64), np.zeros(0))
        self.served += 1
        obs = msg.to_observation()
        return Observation(view.drone_id, view.t, obs.cells, obs.values)
