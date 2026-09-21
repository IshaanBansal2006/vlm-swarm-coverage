r"""Isaac Sim scene for vlm-swarm-coverage: renders the world, follows the loop's poses, and
optionally scores each drone's nadir camera and publishes the result (decision 060).

RUN ON WINDOWS through the bridge launcher (ROS env + loopback Fast DDS profile):

    C:\IsaacSim\run_scene.bat \\wsl.localhost\Ubuntu-22.04\home\ishaan\projects\vlm-swarm-coverage\sim\scenes\coverage_scene.py ^
        --repo \\wsl.localhost\Ubuntu-22.04\home\ishaan\projects\vlm-swarm-coverage ^
        --record C:\Users\ishaa\AppData\Local\Temp\vsc\run1 --score oracle

Then on WSL: `source scripts/ros-env.sh && python3 -m vlm_swarm_coverage.render configs/demo.toml`.

What it does each frame: apply the newest /drone_poses, move the chase camera, render, write a
JPEG of the chase view (if --record), and every --score-period seconds score each drone's nadir
view with the chosen scorer and publish an ObservationMessage on /importance. The nadir frame is
handed to the scorer as an RGB array; the oracle ignores it, a model does not.

Two more modes. `--record-nadir` saves every scored nadir frame beside the chase frames, with a
manifest, so the frames a model would see are on disk. `--sweep views.json` ignores the loop
entirely: it places one nadir camera at each listed view, renders, and saves the frame and a
manifest line, which is how the calibration study and the cache fill get their frames without a
model on this host. `--no-ros` skips the bridge (implied by `--sweep`).

The nadir camera is oriented with the drone's own yaw quaternion: a USD camera looks down its
local -Z, so the identity orientation already points at the ground with image x along world x;
rotating it by the yaw keeps it nadir with image x along the heading, which is the convention
`scoring.footprint_cells` and `models.ground_to_pixel` use.

Isaac's bundled Python needs pydantic once: `C:\IsaacSim\python.bat -m pip install pydantic`.
The scene, field, scoring and schema modules are imported from --repo/src by path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo", required=True, help="Repo root as seen from Windows")
parser.add_argument("--config", default=None, help="Run config (.json) for area/swarm; default demo")
parser.add_argument("--record", default=None, help="Directory for chase-camera JPEGs + meta.jsonl")
parser.add_argument("--score", choices=["none", "oracle", "vlm"], default="none")
parser.add_argument("--score-period", type=float, default=1.0)
parser.add_argument("--model", default=None, help="Model id for --score vlm")
parser.add_argument("--duration", type=float, default=150.0)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--nadir-res", type=int, default=448)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--record-nadir", action="store_true", help="Save every scored nadir frame under --record/nadir")
parser.add_argument("--sweep", default=None, help="views.json: render one nadir frame per view, no loop, no ROS")
parser.add_argument("--no-ros", action="store_true")
parser.add_argument("--settle", type=int, default=4, help="Renderer updates after moving a camera before grabbing")
parser.add_argument("--textures", default=None, help="Directory of CC0 textures (default: <repo>/sim/assets/textures); 'none' for flat colours")
args, _unknown = parser.parse_known_args()
if args.sweep:
    args.no_ros = True
    if not args.record:
        raise SystemExit("--sweep needs --record (where the frames and manifest go)")

sys.path.insert(0, str(Path(args.repo) / "src"))
from vlm_swarm_coverage import scene as scene_mod  # noqa: E402
from vlm_swarm_coverage.field import Grid, rasterize_ground_truth  # noqa: E402
from vlm_swarm_coverage.schemas import ObservationMessage  # noqa: E402
from vlm_swarm_coverage.scoring import OracleScorer, View, VLMScorer  # noqa: E402

W, H = 1280, 720
from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.timeline  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import Camera, Cube, Cylinder, DistantLight  # noqa: E402
from isaacsim.core.experimental.prims import XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

if not args.no_ros:
    app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("omni.replicator.core")
app_utils.enable_extension("omni.kit.material.library")
simulation_app.update()

import omni.replicator.core as rep  # noqa: E402

if not args.no_ros:
    import rclpy  # noqa: E402
    from std_msgs.msg import String, UInt8MultiArray  # noqa: E402


def log(msg):
    print(f"[vsc-scene] {msg}", flush=True)


# --- Configuration -----------------------------------------------------------
if args.config:
    cfg = json.loads(Path(args.config).read_text())
    n_drones, altitude = cfg["swarm"]["n_drones"], cfg["swarm"]["altitude"]
    fov_deg, aspect = cfg["swarm"]["camera_fov_deg"], cfg["swarm"]["camera_aspect"]
    cell_size = cfg["area"]["cell_size"]
    if cfg["scene"]["kind"] == "random":
        SCENE = scene_mod.random_scene(cfg["scene"]["seed"], cfg["scene"]["n_targets"], cfg["scene"]["n_distractors"],
                                       cfg["area"]["width"], cfg["area"]["height"], n_drones, altitude)
    else:
        SCENE = scene_mod.default_scene(n_drones, altitude)
    if cfg["scene"].get("floor") is not None:
        from dataclasses import replace as _replace

        SCENE = _replace(SCENE, mission=_replace(SCENE.mission, floor=cfg["scene"]["floor"]))
else:
    n_drones, altitude, fov_deg, aspect, cell_size = 3, 12.0, 70.0, 4 / 3, 2.0
    SCENE = scene_mod.default_scene(n_drones, altitude)
GRID = Grid(SCENE.width, SCENE.height, cell_size)
SWEEP_VIEWS = json.loads(Path(args.sweep).read_text()) if args.sweep else None

# --- Stage -------------------------------------------------------------------
stage_utils.create_new_stage()
stage_utils.set_stage_units(meters_per_unit=1.0)
DistantLight("/World/DistantLight").set_intensities(300)
try:
    from isaacsim.core.experimental.objects import DomeLight

    DomeLight("/World/DomeLight").set_intensities(250)
except Exception as exc:  # cosmetic fill light only
    log(f"WARNING dome light skipped: {exc!r}")

COLORS = {
    "ground": (0.32, 0.42, 0.22), "road": (0.16, 0.17, 0.19), "damaged_road": (0.45, 0.30, 0.18),
    "debris": (0.55, 0.45, 0.30), "stalled_vehicle": (0.80, 0.12, 0.10), "parked_vehicle": (0.75, 0.77, 0.80),
    "tree": (0.10, 0.35, 0.12), "shed": (0.50, 0.42, 0.35), "drone": (0.96, 0.96, 0.98), "amber": (1.0, 0.62, 0.11),
    "tyre": (0.06, 0.06, 0.07), "post": (0.45, 0.45, 0.47), "glass": (0.12, 0.14, 0.18), "roof_red": (0.62, 0.09, 0.08),
    "roof_silver": (0.60, 0.62, 0.66),
}
_materials = {}

# Surfaces a model looks at get a real, world-projected texture (decision 064): Poly Haven CC0
# maps fetched by scripts/fetch-textures.sh, tiled at their physical size in metres. Anything
# not listed keeps its flat colour.
TEXTURES = {
    "ground": ("sparse_grass.jpg", 2.0), "road": ("asphalt_pit_lane.jpg", 2.0), "damaged_road": ("aerial_mud_1.jpg", 8.0),
    "debris": ("aerial_ground_rock.jpg", 4.0), "tree": ("forrest_ground_01.jpg", 2.5),
}
TEXTURE_DIR = None if args.textures == "none" else Path(args.textures or (Path(args.repo) / "sim" / "assets" / "textures"))
_textured = {}


def _texture_material(key):
    """An OmniPBR material whose diffuse map is projected in world space, so no UVs are needed."""
    import omni.kit.commands
    from pxr import Sdf, UsdShade

    if key in _textured:
        return _textured[key]
    file, size_m = TEXTURES[key]
    path = TEXTURE_DIR / file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch-textures.sh")
    mtl_path = f"/World/Looks/tex_{key}"
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url="OmniPBR.mdl", mtl_name="OmniPBR", mtl_path=mtl_path)
    stage = stage_utils.get_current_stage()
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{mtl_path}/Shader"))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(str(path))
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / size_m, 1.0 / size_m))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.85)
    _textured[key] = UsdShade.Material(stage.GetPrimAtPath(mtl_path))
    return _textured[key]


def paint(prim, key):
    """What a model sees. The oracle reads the scene description, not pixels."""
    if TEXTURE_DIR is not None and key in TEXTURES:
        try:
            from pxr import UsdShade

            material = _texture_material(key)
            stage = stage_utils.get_current_stage()
            for p in prim.paths if hasattr(prim, "paths") else [prim.prim_path]:
                UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(p)).Bind(material)
            return
        except Exception as exc:
            log(f"WARNING texture({key}) failed, falling back to flat colour: {exc!r}")
    try:
        from isaacsim.core.experimental.materials import PreviewSurfaceMaterial

        if key not in _materials:
            mat = PreviewSurfaceMaterial(f"/World/Looks/{key}")
            mat.set_input_values("diffuseColor", [list(COLORS[key])])
            mat.set_input_values("roughness", [0.7])
            _materials[key] = mat
        prim.apply_visual_materials(_materials[key])
    except Exception as exc:
        log(f"WARNING paint({key}) failed: {exc!r}")


def yaw_quat(yaw):
    return [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]


def wheel_quat():
    return [float(np.cos(np.pi / 4)), float(np.sin(np.pi / 4)), 0.0, 0.0]


def make_box(path, extent, color, position, yaw, z_offset=None):
    L, Wd, Hh = [float(v) for v in extent]
    z = position[2] + (Hh / 2 if z_offset is None else z_offset)
    prim = Cube(paths=path, positions=[[position[0], position[1], z]], sizes=1.0, scales=[[L, Wd, Hh]],
                orientations=[yaw_quat(yaw)])
    paint(prim, color)
    return prim


def make_car(path, extent, color, position, yaw):
    L, Wd, Hh = [float(v) for v in extent]
    stage_utils.define_prim(path, "Xform")
    wheel_r = 0.16 * Hh / 1.5 + 0.18
    body_h = Hh * 0.42
    body = Cube(paths=f"{path}/body", positions=[[0.0, 0.0, wheel_r + body_h / 2]], sizes=1.0, scales=[[L, Wd, body_h]])
    paint(body, color)
    cab_h = Hh - wheel_r - body_h
    cab = Cube(paths=f"{path}/cabin", positions=[[-0.08 * L, 0.0, wheel_r + body_h + cab_h / 2]], sizes=1.0,
               scales=[[0.5 * L, 0.86 * Wd, cab_h]])
    paint(cab, color)
    # What a nadir camera keys on: a roof a shade darker than the body, dark glass at both ends of
    # the cabin (windscreen and rear window), and the bonnet/boot seams as thin dark lines.
    top = wheel_r + body_h + cab_h
    roof = Cube(paths=f"{path}/roof", positions=[[-0.08 * L, 0.0, top + 0.01]], sizes=1.0, scales=[[0.30 * L, 0.80 * Wd, 0.02]])
    paint(roof, "roof_red" if color == "stalled_vehicle" else "roof_silver")
    for name, dx in (("windscreen", 0.13 * L), ("rear_window", -0.30 * L)):
        g = Cube(paths=f"{path}/{name}", positions=[[dx, 0.0, top + 0.01]], sizes=1.0, scales=[[0.11 * L, 0.78 * Wd, 0.02]])
        paint(g, "glass")
    for name, dx in (("seam_front", 0.22 * L), ("seam_rear", -0.36 * L)):
        sm = Cube(paths=f"{path}/{name}", positions=[[dx, 0.0, wheel_r + body_h + 0.01]], sizes=1.0, scales=[[0.02 * L, 0.90 * Wd, 0.02]])
        paint(sm, "tyre")
    for side in (1.0, -1.0):
        m = Cube(paths=f"{path}/mirror_{'l' if side > 0 else 'r'}", positions=[[0.16 * L, side * (Wd / 2 + 0.08), wheel_r + body_h + 0.05]],
                 sizes=1.0, scales=[[0.05 * L, 0.16, 0.08]])
        paint(m, color)
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        w = Cylinder(paths=f"{path}/wheel{i}", positions=[[sx * 0.33 * L, sy * (Wd / 2 - 0.05), wheel_r]],
                     radii=[wheel_r], heights=[0.22], orientations=[wheel_quat()])
        paint(w, "tyre")
    xf = XformPrim(paths=path, reset_xform_op_properties=True)
    xf.set_world_poses(positions=[list(position)], orientations=[yaw_quat(yaw)])
    return xf


def make_tree(path, extent, position):
    stage_utils.define_prim(path, "Xform")
    trunk = Cylinder(paths=f"{path}/trunk", positions=[[0.0, 0.0, 1.0]], radii=[0.2], heights=[2.0])
    paint(trunk, "post")
    canopy = Cylinder(paths=f"{path}/canopy", positions=[[0.0, 0.0, 2.0 + (extent[2] - 2.0) / 2]],
                      radii=[extent[0] / 2], heights=[extent[2] - 2.0])
    paint(canopy, "tree")
    xf = XformPrim(paths=path, reset_xform_op_properties=True)
    xf.set_world_poses(positions=[list(position)], orientations=[yaw_quat(0.0)])
    return xf


def make_quad(path, position, yaw, span=0.9):
    stage_utils.define_prim(path, "Xform")
    body = Cube(paths=f"{path}/body", positions=[[0.0, 0.0, 0.0]], sizes=1.0, scales=[[0.28 * span, 0.28 * span, 0.09 * span]])
    paint(body, "drone")
    for i, ang in enumerate((np.pi / 4, -np.pi / 4)):
        arm = Cube(paths=f"{path}/arm{i}", positions=[[0.0, 0.0, 0.0]], sizes=1.0, scales=[[span, 0.06 * span, 0.04 * span]],
                   orientations=[yaw_quat(ang)])
        paint(arm, "post")
    for i, ang in enumerate((np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4)):
        r = Cylinder(paths=f"{path}/rotor{i}", positions=[[0.5 * span * np.cos(ang), 0.5 * span * np.sin(ang), 0.03 * span]],
                     radii=[0.17 * span], heights=[0.015 * span])
        paint(r, "amber" if i < 2 else "drone")
    xf = XformPrim(paths=path, reset_xform_op_properties=True)
    xf.set_world_poses(positions=[list(position)], orientations=[yaw_quat(yaw)])
    return xf


# A plain slab far beyond the area instead of the default ground plane, whose grid texture would
# show through the painted ground and would be what a model sees.
outer = Cube(paths="/World/ground_outer", positions=[[SCENE.width / 2, SCENE.height / 2, -0.06]], sizes=1.0,
             scales=[[SCENE.width + 400.0, SCENE.height + 400.0, 0.02]])
paint(outer, "ground")
ground = Cube(paths="/World/ground_visual", positions=[[SCENE.width / 2, SCENE.height / 2, -0.01]], sizes=1.0,
              scales=[[SCENE.width, SCENE.height, 0.02]])
paint(ground, "ground")
road = Cube(paths="/World/road_visual", positions=[[SCENE.width / 2, SCENE.road.y_center, 0.005]], sizes=1.0,
            scales=[[SCENE.width, SCENE.road.width, 0.01]])
paint(road, "road")

for f in SCENE.features:
    path = f"/World/feature_{f.feature_id}"
    if f.label in ("stalled_vehicle", "parked_vehicle"):
        make_car(path, f.extent, f.label, f.position, f.yaw)
    elif f.label == "tree":
        make_tree(path, f.extent, f.position)
    elif f.label == "damaged_road":
        make_box(path, (f.extent[0], f.extent[1], 0.03), f.label, f.position, f.yaw, z_offset=0.02)
    else:
        make_box(path, f.extent, f.label, f.position, f.yaw)

drone_prims = {} if SWEEP_VIEWS is not None else {i: make_quad(f"/World/drone_{i}", start, 0.0) for i, start in enumerate(SCENE.drone_starts)}

# --- Cameras -----------------------------------------------------------------
# A USD camera looks down its local -Z with +Y up in the image, so the identity orientation is
# already nadir. Rotating by the drone's yaw keeps it nadir with image x along the heading.
NADIR_DROP = 0.1  # metres below the drone body, so the quad's own geometry is behind the lens
FOCAL_MM, APERTURE_MM = 16.0, 20.955


def nadir_pose(pos, yaw):
    return [pos[0], pos[1], pos[2] - NADIR_DROP], yaw_quat(yaw)


def configure_camera(path, focal_mm, aperture_mm, w, h):
    usd_cam = UsdGeom.Camera(stage_utils.get_current_stage().GetPrimAtPath(path))
    usd_cam.GetFocalLengthAttr().Set(focal_mm)  # USD attr is mm; the API setter scales by stage units
    usd_cam.GetHorizontalApertureAttr().Set(aperture_mm)
    usd_cam.GetVerticalApertureAttr().Set(aperture_mm * h / w)
    usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 3000.0))


nadir = {}
if args.score != "none" or args.record_nadir or SWEEP_VIEWS is not None:
    nadir_aperture = 2 * FOCAL_MM * np.tan(np.radians(fov_deg) / 2)  # aperture that gives the configured FOV
    nadir_w, nadir_h = args.nadir_res, int(args.nadir_res / aspect)
    starts = [(SCENE.width / 2, SCENE.height / 2, altitude)] if SWEEP_VIEWS is not None else list(SCENE.drone_starts)
    for i, start in enumerate(starts):
        path = f"/World/nadir_{i}"
        p0, q0 = nadir_pose(list(start), 0.0)
        cam = Camera(paths=path, positions=[p0], orientations=[q0])
        configure_camera(path, FOCAL_MM, nadir_aperture, nadir_w, nadir_h)
        rp = rep.create.render_product(path, (nadir_w, nadir_h))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        nadir[i] = (cam, annot)


def grab(annot):
    data = annot.get_data()
    if data is not None and getattr(data, "size", 0) > 0:
        return np.asarray(data)[:, :, :3]
    return None


def look_at_quat(pos, aim, up=(0.0, 0.0, 1.0)):
    f = aim - pos
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, dtype=float))
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.column_stack([r, u, -f])
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return [float(w), float(x), float(y), float(z)], R


CHASE_PATH = "/World/ChaseCam"
CHASE_OFFSET = np.array([-14.0, -10.0, 9.0])
chase_q0, _ = look_at_quat(np.asarray(SCENE.drone_starts[0]) + CHASE_OFFSET, np.asarray(SCENE.drone_starts[0]))
chase = Camera(paths=CHASE_PATH, positions=[(np.asarray(SCENE.drone_starts[0]) + CHASE_OFFSET).tolist()], orientations=[chase_q0])
configure_camera(CHASE_PATH, FOCAL_MM, APERTURE_MM, W, H)
chase_rp = rep.create.render_product(CHASE_PATH, (W, H))
chase_annot = rep.AnnotatorRegistry.get_annotator("rgb")
chase_annot.attach([chase_rp])

# --- Sweep mode: one frame per listed view, then exit ------------------------------
if SWEEP_VIEWS is not None:
    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    app_utils.play()
    for _ in range(10):
        simulation_app.update()
    out_dir = Path(args.record)
    (out_dir / "sweep").mkdir(parents=True, exist_ok=True)
    cam, annot = nadir[0]
    log(f"sweep: {len(SWEEP_VIEWS)} views -> {out_dir}")
    with open(out_dir / "manifest.jsonl", "w") as manifest:
        for k, v in enumerate(SWEEP_VIEWS):
            p, q = nadir_pose([v["x"], v["y"], v["z"]], v["yaw"])
            cam.set_world_poses(positions=[p], orientations=[q])
            # The annotator lags the camera move by a frame: without a forced render step the
            # grabbed image belongs to the previous pose. Settle, then step the renderer.
            for _ in range(args.settle):
                simulation_app.update()
            try:
                rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
            except Exception as exc:  # older orchestrators: fall back to extra updates
                log(f"WARNING orchestrator.step unavailable ({exc!r}); settling with updates only")
                for _ in range(args.settle):
                    simulation_app.update()
            frame = grab(annot)
            if frame is None:
                log(f"WARNING no frame for view {k}")
                continue
            name = f"sweep/{k:06d}.png"
            Image.fromarray(frame).save(out_dir / name)
            manifest.write(json.dumps({**v, "index": k, "file": name, "fov_deg": fov_deg, "aspect": aspect,
                                       "width": int(frame.shape[1]), "height": int(frame.shape[0])}) + "\n")
            if (k + 1) % 100 == 0:
                log(f"sweep: {k + 1}/{len(SWEEP_VIEWS)}")
    log(f"DONE sweep {len(SWEEP_VIEWS)} views")
    app_utils.stop()
    simulation_app.close()
    raise SystemExit(0)

# --- ROS2: /clock via action graph; poses in and importance out via rclpy ------
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: [
            ("OnTick", "omni.graph.action.OnPlaybackTick"),
            ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ],
        og.Controller.Keys.CONNECT: [
            ("OnTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("SimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("Context.outputs:context", "PublishClock.inputs:context"),
        ],
        og.Controller.Keys.SET_VALUES: [("PublishClock.inputs:topicName", "/clock")],
    },
)
simulation_app.update()
simulation_app.update()
SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
app_utils.play()
simulation_app.update()

rclpy.init()
ros_node = rclpy.create_node("vsc_scene")
importance_pub = ros_node.create_publisher(UInt8MultiArray, "/importance", 50)
latest_poses = {}
pose_seq = [0]


def on_drone_poses(msg):
    pose_seq[0] += 1
    data = json.loads(msg.data)
    for p in data["poses"]:
        latest_poses[p["drone_id"]] = (p["position"], p["orientation"], data.get("t", 0.0))


ros_node.create_subscription(String, "/drone_poses", on_drone_poses, 10)

# --- Scorer on this host --------------------------------------------------------
scorer = None
if args.score == "oracle":
    scorer = OracleScorer(rasterize_ground_truth(SCENE, GRID))
elif args.score == "vlm":
    scorer = VLMScorer(args.model, GRID)
obs_seq = {i: 0 for i in drone_prims}


nadir_manifest = None
if args.record and args.record_nadir:
    (Path(args.record) / "nadir").mkdir(parents=True, exist_ok=True)
    nadir_manifest = open(Path(args.record) / "nadir_manifest.jsonl", "w")


def score_and_publish(i, t):
    pos, quat, loop_t = latest_poses[i]
    yaw = 2.0 * np.arctan2(quat[3], quat[0])
    frame = grab(nadir[i][1]) if i in nadir else None
    if frame is not None and nadir_manifest is not None:
        name = f"nadir/d{i}_{obs_seq[i]:05d}.jpg"
        Image.fromarray(frame).save(Path(args.record) / name, quality=95)
        nadir_manifest.write(json.dumps({"drone": i, "seq": obs_seq[i], "t": loop_t, "x": pos[0], "y": pos[1], "z": pos[2],
                                         "yaw": yaw, "fov_deg": fov_deg, "aspect": aspect, "file": name}) + "\n")
    if scorer is None:
        obs_seq[i] += 1
        return
    view = View(i, t, pos[0], pos[1], pos[2], yaw, fov_deg, aspect, frame, SCENE.mission.text)
    obs = scorer.score(view)
    msg = UInt8MultiArray()
    msg.data = list(ObservationMessage.from_observation(obs, obs_seq[i]).to_bytes())
    importance_pub.publish(msg)
    obs_seq[i] += 1


# --- Loop, wall-clock paced ------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
record_dir = Path(args.record) if args.record else None
meta = None
if record_dir:
    (record_dir / "frames").mkdir(parents=True, exist_ok=True)
    meta = open(record_dir / "meta.jsonl", "w")
    json.dump({"W": W, "H": H, "focal_mm": FOCAL_MM, "aperture_mm": APERTURE_MM, "fps": args.fps,
               "chase_offset": CHASE_OFFSET.tolist()}, open(record_dir / "camera.json", "w"))
for _ in range(10):
    simulation_app.update()

log(f"scene up: {len(drone_prims)} drones, score={args.score}, record={record_dir}")
t0 = time.perf_counter()
t = 0.0
frame_idx = 0
next_save = 0.0
next_score = 0.0
while simulation_app.is_running() and t < args.duration:
    t = time.perf_counter() - t0
    for _ in range(20):
        before = pose_seq[0]
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        if pose_seq[0] == before:
            break
    if timeline.is_playing():
        for i, prim in drone_prims.items():
            if i in latest_poses:
                pos, quat, _ = latest_poses[i]
                prim.set_world_poses(positions=[pos], orientations=[quat])
                if i in nadir:
                    p, q = nadir_pose(pos, 2.0 * np.arctan2(quat[3], quat[0]))
                    nadir[i][0].set_world_poses(positions=[p], orientations=[q])
        if latest_poses:
            centre = np.mean([p[0] for p in latest_poses.values()], axis=0)
            centre[2] = altitude
            chase_pos = centre + CHASE_OFFSET
            chase_q, chase_R = look_at_quat(chase_pos, centre)
            chase.set_world_poses(positions=[chase_pos.tolist()], orientations=[chase_q])
    simulation_app.update()
    if timeline.is_playing() and (scorer is not None or nadir_manifest is not None) and latest_poses and t >= next_score:
        for i in latest_poses:
            score_and_publish(i, t)
        next_score += args.score_period
    if record_dir and timeline.is_playing() and t >= next_save:
        data = chase_annot.get_data()
        if data is not None and getattr(data, "size", 0) > 0:
            Image.fromarray(np.asarray(data)[:, :, :3]).save(record_dir / "frames" / f"f_{frame_idx:05d}.jpg", quality=94)
            meta.write(json.dumps({"frame": frame_idx, "t": t,
                                   "cam_pos": chase_pos.tolist() if latest_poses else None,
                                   "cam_R": chase_R.tolist() if latest_poses else None,
                                   "drones": {i: list(v[0]) for i, v in latest_poses.items()},
                                   "loop_t": {i: v[2] for i, v in latest_poses.items()}}) + "\n")
            frame_idx += 1
            if frame_idx % 150 == 0:
                log(f"t={t:6.1f}s frames={frame_idx} drones={sorted(latest_poses)}")
        next_save += 1.0 / args.fps
if meta:
    meta.close()
if nadir_manifest:
    nadir_manifest.close()
log(f"DONE t={t:.1f}s frames={frame_idx}")
ros_node.destroy_node()
rclpy.shutdown()
app_utils.stop()
simulation_app.close()
