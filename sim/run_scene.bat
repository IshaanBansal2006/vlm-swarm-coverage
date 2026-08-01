@echo off
REM vlm-swarm-coverage: launch an Isaac scene (headless-capable) with the ROS2 bridge.
REM
REM This is a WINDOWS launcher. Copy it (and config\fastdds-loopback.xml) to
REM C:\IsaacSim\ and run:
REM     C:\IsaacSim\run_scene.bat C:\IsaacSim\coverage_scene.py
REM
REM Why it exists (vendored from drone-swarm-autonomy):
REM - python.bat (unlike isaac-sim.bat) does NOT call setup_ros_env.bat, so a bare
REM   `python.bat scene.py` has no ROS2 env and the bridge fails to load rcutils.dll.
REM - Mirrored networking shares one IP between Windows and WSL, so default Fast DDS
REM   tries shared-memory / own-IP locators and data never crosses the boundary.
REM   The loopback UDP-only profile (fastdds-loopback.xml) fixes that; discovery is
REM   plain RTPS via initial peers at 127.0.0.1 (no discovery server, no multicast).
REM
REM Sweeps run headless: pass --headless through to the scene script. Rendering N
REM drone cameras is this project's dominant sim cost, so headless is the default
REM mode for anything that is not a demo or a validity slice.
call "C:\IsaacSim\setup_ros_env.bat"
set ROS_DOMAIN_ID=0
set ROS_DISCOVERY_SERVER=
set FASTRTPS_DEFAULT_PROFILES_FILE=C:\IsaacSim\fastdds-loopback.xml
set FASTDDS_DEFAULT_PROFILES_FILE=C:\IsaacSim\fastdds-loopback.xml
call "C:\IsaacSim\python.bat" %*
