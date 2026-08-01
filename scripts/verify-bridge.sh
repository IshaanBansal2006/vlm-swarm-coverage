#!/usr/bin/env bash
# vlm-swarm-coverage — verify the Isaac Sim (Windows) <-> WSL ROS2 bridge.
# Vendored from drone-swarm-autonomy; see that repo's docs/isaac-wsl-ros2-bridge.md
# for the twelve issues behind this configuration.
# Usage:  bash scripts/verify-bridge.sh
#
# Verifies the loopback UDP-only Fast DDS path (config/fastdds-loopback.xml on
# BOTH sides — see docs/isaac-wsl-ros2-bridge.md). Isaac must be running a scene
# launched via C:\IsaacSim\run_scene.bat with the timeline PLAYING (in Isaac,
# running != publishing).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/ros-env.sh"

echo
echo "== 1. WSL IP (mirrored mode: should match your Windows IPv4) =="
ip -4 addr show | awk '/inet /{print "   " $2 "  ("$NF")"}'

echo
echo "== 2. Fast DDS profile active? =="
[ -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ] \
  && echo "   OK: $FASTRTPS_DEFAULT_PROFILES_FILE" \
  || echo "   MISSING profile — bridge will not cross!"

echo
echo "== 3. Topics visible from WSL (fresh daemon, ~10s discovery) =="
ros2 daemon stop >/dev/null 2>&1 || true
timeout 12 ros2 topic list 2>/dev/null | sed 's/^/   /' \
  || echo "   (list failed — daemon flaky; trusting step 4 instead)"

echo
echo "== 4. DATA check: /clock (the real test — discovery can pass while data fails) =="
timeout 12 ros2 topic echo /clock rosgraph_msgs/msg/Clock --once 2>/dev/null | head -4 | sed 's/^/   /' \
  && echo "   OK: data crosses the boundary." \
  || echo "   NO DATA — is the scene playing? Is C:\\IsaacSim\\fastdds-loopback.xml in place?"

echo
echo "If step 4 printed a clock value, the bridge is fully up (discovery AND data)."
