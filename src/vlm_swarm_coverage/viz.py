"""Pictures of a run, computed offline from its event log (decision 061).

Nothing here is a metric. The visualiser draws what the log recorded: the answer key, each
drone's belief as it was shared, positions, footprints, and the Voronoi partition each drone
was acting on. `python -m vlm_swarm_coverage.viz runs/<run> --video out.mp4 --summary out.png`.

Requires matplotlib (the `[analysis]` or `[dev]` extra); ffmpeg for the mp4.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vlm_swarm_coverage.artifacts import RunDir
from vlm_swarm_coverage.control import voronoi_mask
from vlm_swarm_coverage.field import Grid, rasterize_ground_truth
from vlm_swarm_coverage.scene import TARGET_LABELS
from vlm_swarm_coverage.simulation import build_scene

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from vlm_swarm_coverage.scene import Scene

log = logging.getLogger(__name__)

TARGET_COLOR = "#ff9f1c"
DISTRACTOR_COLOR = "#748cab"
DRONE_COLORS = ("#f0ebd8", "#5bc0eb", "#9bc53d", "#e55934", "#fa7921")


@dataclass
class RunFrames:
    """The log, reshaped for drawing: one row per sync time, poses interpolated to those times."""

    grid: Grid
    scene: Scene
    ground_truth: NDArray[np.float64]
    times: NDArray[np.float64]
    positions: NDArray[np.float64]
    yaws: NDArray[np.float64]
    beliefs: NDArray[np.float64]
    fov_deg: float
    aspect: float
    altitude: float

    @property
    def n_drones(self) -> int:
        return self.positions.shape[1]


def load_run(run_dir: RunDir) -> RunFrames:
    cfg = run_dir.config
    scene = build_scene(cfg)
    grid = Grid(cfg.area.width, cfg.area.height, cfg.area.cell_size)
    gt = rasterize_ground_truth(scene, grid).values
    n = cfg.swarm.n_drones
    poses = {i: [] for i in range(n)}
    for e in run_dir.iter_events("pose"):
        poses[e["drone"]].append((e["t"], e["x"], e["y"], e["yaw"]))
    beliefs: dict[float, dict[int, list[float]]] = {}
    for e in run_dir.iter_events("belief"):
        beliefs.setdefault(e["t"], {})[e["drone"]] = e["values"]
    times = np.array(sorted(t for t, per in beliefs.items() if len(per) == n))
    if len(times) == 0:
        raise ValueError(f"{run_dir.path} has no complete belief snapshots; nothing to draw")
    pos = np.zeros((len(times), n, 2))
    yaw = np.zeros((len(times), n))
    for i in range(n):
        arr = np.asarray(poses[i])
        pos[:, i, 0] = np.interp(times, arr[:, 0], arr[:, 1])
        pos[:, i, 1] = np.interp(times, arr[:, 0], arr[:, 2])
        yaw[:, i] = np.interp(times, arr[:, 0], np.unwrap(arr[:, 3]))
    bel = np.array([[np.asarray(beliefs[t][i]).reshape(grid.shape) for i in range(n)] for t in times])
    return RunFrames(grid, scene, gt, times, pos, yaw, bel, cfg.swarm.camera_fov_deg, cfg.swarm.camera_aspect, cfg.swarm.altitude)


def _draw_scene(ax: Axes, scene: Scene) -> None:
    from matplotlib.patches import Polygon, Rectangle

    ax.add_patch(Rectangle((0, scene.road.y_center - scene.road.width / 2), scene.width, scene.road.width,
                           facecolor="none", edgecolor="#ffffff", linewidth=0.6, alpha=0.35))
    for f in scene.features:
        color = TARGET_COLOR if f.label in TARGET_LABELS else DISTRACTOR_COLOR
        ax.add_patch(Polygon(f.footprint, closed=True, facecolor="none", edgecolor=color, linewidth=1.2))


def _draw_footprint(ax: Axes, x: float, y: float, z: float, yaw: float, fov_deg: float, aspect: float, color: str) -> None:
    from matplotlib.patches import Polygon

    hw = z * math.tan(math.radians(fov_deg) / 2)
    hh = hw / aspect
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) @ np.array([[c, s], [-s, c]]) + [x, y]
    ax.add_patch(Polygon(corners, closed=True, facecolor=color, edgecolor=color, alpha=0.12, linewidth=0.8))


def _draw_partition(ax: Axes, frames: RunFrames, k: int) -> None:
    centres = frames.grid.cell_centers()
    owner = np.full(frames.grid.shape, -1)
    for i in range(frames.n_drones):
        peers = {j: frames.positions[k, j] for j in range(frames.n_drones) if j != i}
        owner[voronoi_mask(centres, frames.positions[k, i], peers, i)] = i
    cs = frames.grid.cell_size
    ax.contour(centres[..., 0], centres[..., 1], owner, levels=np.arange(frames.n_drones) + 0.5,
               colors="#ffffff", linewidths=0.7, alpha=0.6)
    ax.set_xlim(0, frames.grid.cols * cs)
    ax.set_ylim(0, frames.grid.rows * cs)


def draw_frame(ax: Axes, frames: RunFrames, k: int, drone: int | None = None, trail: int = 30) -> None:
    """One time step: heatmap of `drone`'s belief (or the mean of all), scene, footprints, tracks."""
    field = frames.beliefs[k, drone] if drone is not None else frames.beliefs[k].mean(axis=0)
    cs = frames.grid.cell_size
    vmax = max(float(frames.ground_truth.max()), 1e-6)
    ax.imshow(field, origin="lower", extent=(0, frames.grid.cols * cs, 0, frames.grid.rows * cs),
              cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    _draw_scene(ax, frames.scene)
    _draw_partition(ax, frames, k)
    for i in range(frames.n_drones):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        x, y = frames.positions[k, i]
        _draw_footprint(ax, x, y, frames.altitude, frames.yaws[k, i], frames.fov_deg, frames.aspect, color)
        lo = max(0, k - trail)
        ax.plot(frames.positions[lo : k + 1, i, 0], frames.positions[lo : k + 1, i, 1], color=color, linewidth=1.0, alpha=0.8)
        ax.plot(x, y, "o", color=color, markersize=6, markeredgecolor="#000000")
        ax.annotate(f"d{i}", (x, y), xytext=(5, 5), textcoords="offset points", color=color, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    label = f"drone {drone}" if drone is not None else "mean of drones"
    ax.set_title(f"belief ({label})   t = {frames.times[k]:6.1f} s", loc="left", fontsize=9, color="#f0ebd8")


def _figure(width_px: int, height_px: int):  # type: ignore[no-untyped-def]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100, facecolor="#0d1321")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.08, wspace=0.08)
    return fig, plt


def render_video(run_dir: RunDir, out: Path, fps: float = 10.0, drone: int | None = None, size: tuple[int, int] = (1280, 720)) -> Path:
    """Belief heatmap over time, one frame per belief snapshot, side by side with the answer key.

    Writes PNGs to a temporary directory and encodes with ffmpeg. Without ffmpeg the PNGs are
    moved next to `out` and the directory is returned instead.
    """
    frames = load_run(run_dir)
    fig, plt = _figure(*size)
    tmp = Path(tempfile.mkdtemp(prefix="vsc-viz-"))
    for k in range(len(frames.times)):
        fig.clf()
        fig.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.08, wspace=0.08)
        ax_b = fig.add_subplot(1, 2, 1)
        ax_g = fig.add_subplot(1, 2, 2)
        draw_frame(ax_b, frames, k, drone=drone)
        _draw_truth(ax_g, frames)
        fig.savefig(tmp / f"f_{k:05d}.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    if shutil.which("ffmpeg") is None:
        dest = out.with_suffix("")
        shutil.move(str(tmp), str(dest))
        log.warning("ffmpeg not found; frames left in %s", dest)
        return dest
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f_%05d.png"),
                    "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp)
    log.info("video written: %s (%d frames)", out, len(frames.times))
    return out


def _draw_truth(ax: Axes, frames: RunFrames) -> None:
    cs = frames.grid.cell_size
    ax.imshow(frames.ground_truth, origin="lower", extent=(0, frames.grid.cols * cs, 0, frames.grid.rows * cs),
              cmap="magma", vmin=0.0, vmax=max(float(frames.ground_truth.max()), 1e-6), interpolation="nearest")
    _draw_scene(ax, frames.scene)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("ground truth", loc="left", fontsize=9, color="#f0ebd8")


def summary_figure(run_dir: RunDir, out: Path) -> Path:
    """Full trajectories over the answer key, plus what the channel did over time."""
    frames = load_run(run_dir)
    fig, plt = _figure(1280, 720)
    ax = fig.add_subplot(1, 2, 1)
    _draw_truth(ax, frames)
    for i in range(frames.n_drones):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        ax.plot(frames.positions[:, i, 0], frames.positions[:, i, 1], color=color, linewidth=1.2, label=f"d{i}")
        ax.plot(*frames.positions[0, i], "s", color=color, markersize=5)
    ax.legend(loc="lower right", fontsize=7, facecolor="#0d1321", labelcolor="#f0ebd8")
    ax.set_title("trajectories over ground truth", loc="left", fontsize=9, color="#f0ebd8")
    ax2 = fig.add_subplot(1, 2, 2)
    sent = _cumulative(run_dir, "send", "bytes", per="receivers")
    recv = _cumulative(run_dir, "recv", "bytes")
    drop = _cumulative(run_dir, "drop", "bytes")
    for series, label, color in ((sent, "bytes offered", "#748cab"), (recv, "bytes delivered", "#5bc0eb"), (drop, "bytes dropped", "#e55934")):
        if len(series):
            ax2.step(series[:, 0], series[:, 1], where="post", label=label, color=color)
    ax2.set_xlabel("sim time (s)", color="#f0ebd8")
    ax2.set_ylabel("bytes (cumulative)", color="#f0ebd8")
    ax2.tick_params(colors="#f0ebd8")
    ax2.set_facecolor("#0d1321")
    ax2.legend(fontsize=7, facecolor="#0d1321", labelcolor="#f0ebd8")
    ax2.set_title("channel", loc="left", fontsize=9, color="#f0ebd8")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def _cumulative(run_dir: RunDir, kind: str, key: str, per: str | None = None) -> NDArray[np.float64]:
    """Cumulative sum of `key` over events of `kind`; `per` names a multiplier field (receivers)."""
    rows = [(e["t"], e[key] * (e[per] if per else 1)) for e in run_dir.iter_events(kind)]
    if not rows:
        return np.zeros((0, 2))
    arr = np.asarray(rows, dtype=float)
    arr[:, 1] = np.cumsum(arr[:, 1])
    return arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsc-viz", description="Draw a run from its event log.")
    parser.add_argument("run", type=Path)
    parser.add_argument("--video", type=Path, help="mp4 of the belief over time")
    parser.add_argument("--summary", type=Path, help="png of trajectories and channel use")
    parser.add_argument("--drone", type=int, default=None, help="Whose belief to draw (default: mean)")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_dir = RunDir.open(args.run)
    if args.video:
        print(render_video(run_dir, args.video, fps=args.fps, drone=args.drone))
    if args.summary:
        print(summary_figure(run_dir, args.summary))
    if not (args.video or args.summary):
        parser.error("nothing to do: pass --video and/or --summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
