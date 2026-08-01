# vlm-swarm-coverage — ROS2 cross-boundary environment (WSL side)
# Source this in every WSL shell that talks to Isaac Sim on Windows:
#   source scripts/ros-env.sh
#
# HOW THE BRIDGE WORKS (vendored from drone-swarm-autonomy; hard-won 2026-07-29 —
# do not regress this):
# Mirrored WSL networking gives Windows and WSL the SAME IP, so default Fast DDS
# (a) treats the other side as same-host and tries shared-memory transport, which
# cannot cross the OS boundary, and (b) announces own-IP locators that route back
# to the sender. Raw UDP over 127.0.0.1 DOES cross in both directions, so both
# sides force a loopback-only UDP profile (config/fastdds-loopback.xml): UDP
# transport only (no SHM), interface whitelist 127.0.0.1, peer discovery via
# initial peers at 127.0.0.1 (plain RTPS — no multicast, no discovery server,
# humble<->jazzy compatible). Isaac side gets the same XML via run_scene.bat.
#
# The initial-peers list enumerates explicit ports 7410-7448 (participant ids
# 0-19). Port-less peers only probe ids 0-4, which silently makes the 6th+ node
# on the machine undiscoverable — this project runs N drone nodes plus a
# coordinator, so it will hit that ceiling immediately without them.
#
# NOTE: do NOT set ROS_SUPER_CLIENT=true — it crashes the Humble ros2 daemon
# (`!rclpy.ok()` on every CLI call). If the CLI misbehaves, prefer
# `ros2 topic list --no-daemon` and `ros2 topic echo <topic> <type>`.

# zsh and bash need different setup files (you use zsh).
if [ -n "${ZSH_VERSION:-}" ]; then
  source /opt/ros/humble/setup.zsh
else
  source /opt/ros/humble/setup.bash
fi

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT

_VSC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"
export FASTRTPS_DEFAULT_PROFILES_FILE="$_VSC_ROOT/config/fastdds-loopback.xml"

# Package import path. MUST prepend, never assign — a bare PYTHONPATH=src hides
# the system rclpy and every node dies at import.
export PYTHONPATH="$_VSC_ROOT/src:$PYTHONPATH"

echo "[ros-env] ROS_DISTRO=$ROS_DISTRO DOMAIN=$ROS_DOMAIN_ID RMW=$RMW_IMPLEMENTATION profile=$FASTRTPS_DEFAULT_PROFILES_FILE"
