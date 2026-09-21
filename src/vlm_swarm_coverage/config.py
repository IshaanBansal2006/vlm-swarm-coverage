"""Run configuration: one typed object that fully determines a run, together with its seed.

Reproducibility is a hard requirement for this project, so everything that can change a run's
outcome lives here and nowhere else. A run directory snapshots this object; re-running from the
snapshot must reproduce the run.

TOML is the on-disk format because Python 3.11 reads it from the standard library (`tomllib`;
`tomli` on 3.10), which keeps the core dependency set at numpy + pydantic. Snapshots are written as JSON because the
standard library cannot write TOML; the loader accepts either.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _toml_loader():  # type: ignore[no-untyped-def]
    """tomllib on 3.11+, tomli on 3.10 (the ROS 2 Humble interpreter). JSON configs need neither."""
    try:
        import tomllib

        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli

            return tomli
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "reading .toml on Python < 3.11 needs tomli: pip install tomli, or pass a .json config"
            ) from exc


class _Strict(BaseModel):
    """Reject unknown keys so a typo in a config file fails loudly instead of silently defaulting.

    Infinite floats serialise as the JSON constant `Infinity` so a config with `tau_s = inf`
    survives the snapshot round trip instead of becoming `null`."""

    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")


class AreaConfig(_Strict):
    width: float = Field(60.0, gt=0)
    height: float = Field(40.0, gt=0)
    cell_size: float = Field(2.0, gt=0, description="Edge of one importance-grid cell, metres.")

    @property
    def shape(self) -> tuple[int, int]:
        """Grid cells as (rows along y, columns along x); partial cells at the far edge round up."""
        import math

        return (math.ceil(self.height / self.cell_size), math.ceil(self.width / self.cell_size))


class SwarmConfig(_Strict):
    n_drones: int = Field(3, ge=1)
    altitude: float = Field(12.0, gt=0)
    max_speed: float = Field(3.0, gt=0, description="Speed limit per drone, m/s.")
    camera_fov_deg: float = Field(70.0, gt=0, lt=180, description="Full horizontal field of view.")
    camera_aspect: float = Field(4 / 3, gt=0, description="Image width / height.")


class SimConfig(_Strict):
    dt: float = Field(0.1, gt=0, description="Control timestep, seconds.")
    duration_s: float = Field(120.0, gt=0)
    seed: int = Field(0, ge=0)

    @property
    def n_steps(self) -> int:
        return int(round(self.duration_s / self.dt))


class ChannelConfig(_Strict):
    """The inter-drone channel. Defaults describe a perfect channel."""

    drop_rate: float = Field(0.0, ge=0, le=1, description="Probability a message is lost.")
    latency_s: float = Field(0.0, ge=0, description="Fixed delivery delay.")
    bytes_per_sync: int | None = Field(None, gt=0, description="Payload cap per sync, or unlimited.")
    sync_interval_s: float = Field(1.0, gt=0, description="How often a drone shares its belief.")


class ScorerConfig(_Strict):
    kind: Literal["oracle", "cached", "vlm"] = "oracle"
    period_s: float = Field(1.0, gt=0, description="How often a drone scores its view.")
    model: str | None = Field(None, description="Model identifier for kind='vlm'.")
    cache_path: Path | None = Field(None, description="Importance cache for kind='cached'.")
    prompt_set: Literal["mission", "null"] = Field("mission", description="Mission phrases, or one generic phrase.")
    contrast: bool = Field(True, description="Score phrases by softmax against background phrases (decision 016).")
    calibration: Path | None = Field(None, description="Affine score calibration JSON for kind='vlm'.")
    device: str | None = Field(None, description="torch device for kind='vlm'; default picks CUDA if present.")

    @model_validator(mode="after")
    def _kind_has_what_it_needs(self) -> ScorerConfig:
        if self.kind == "vlm" and not self.model:
            raise ValueError("scorer.kind='vlm' needs scorer.model (a model identifier)")
        if self.kind == "cached" and self.cache_path is None:
            raise ValueError("scorer.kind='cached' needs scorer.cache_path")
        return self


class MessageConfig(_Strict):
    """What a drone puts on the wire each sync besides its pose (decision 033)."""

    kind: Literal["dense", "topk", "quantised", "none"] = "dense"
    k: int | None = Field(None, ge=0, description="Cells per message for kind='topk'.")

    @model_validator(mode="after")
    def _topk_needs_k(self) -> MessageConfig:
        if self.kind == "topk" and self.k is None:
            raise ValueError("message.kind='topk' needs message.k (cells per message)")
        if self.kind != "topk" and self.k is not None:
            raise ValueError(f"message.k only applies to kind='topk', not {self.kind!r}")
        return self


class FusionConfig(_Strict):
    """How a drone folds received beliefs into its own (decision 022)."""

    kind: Literal["none", "age_weighted"] = "age_weighted"
    tau_s: float = Field(
        10.0, ge=0,
        description="Staleness time constant, seconds. 0 = freshest observation wins; inf = equal weights.",
    )


class ControllerConfig(_Strict):
    kind: Literal["lloyd", "hold", "replay"] = "lloyd"
    gain: float = Field(1.0, gt=0, description="Proportional gain toward the cell centroid.")
    replay_run: Path | None = Field(None, description="Run directory whose trajectory kind='replay' follows.")

    @model_validator(mode="after")
    def _replay_needs_a_run(self) -> ControllerConfig:
        if self.kind == "replay" and self.replay_run is None:
            raise ValueError("controller.kind='replay' needs controller.replay_run (a run directory)")
        if self.kind != "replay" and self.replay_run is not None:
            raise ValueError("controller.replay_run only applies to kind='replay'")
        return self


class PriorConfig(_Strict):
    """What every drone believes before its first observation (decision 023).

    `floor` is the mission floor everywhere. `truth` is the answer key. `shifted_truth` is the
    answer key displaced by `shift_m` (metres, x then y), floor where it leaves the area.
    `other_scene` is the answer key of a different random layout, `scene_seed`. With
    `observed_at` unset the prior is held but never transmitted (cells start unobserved); set to
    a time, it is stamped as an observation made then and shared like any other."""

    kind: Literal["floor", "truth", "shifted_truth", "other_scene"] = "floor"
    shift_m: tuple[float, float] = (0.0, 0.0)
    scene_seed: int | None = Field(None, ge=0)
    observed_at: float | None = None

    @model_validator(mode="after")
    def _kind_has_what_it_needs(self) -> PriorConfig:
        if self.kind == "other_scene" and self.scene_seed is None:
            raise ValueError("prior.kind='other_scene' needs prior.scene_seed")
        if self.kind != "shifted_truth" and self.shift_m != (0.0, 0.0):
            raise ValueError("prior.shift_m only applies to kind='shifted_truth'")
        if self.kind != "other_scene" and self.scene_seed is not None:
            raise ValueError("prior.scene_seed only applies to kind='other_scene'")
        return self


class SceneConfig(_Strict):
    kind: Literal["default", "random"] = "default"
    seed: int | None = Field(None, ge=0, description="Layout seed for kind='random'.")
    n_targets: int = Field(3, ge=1)
    n_distractors: int = Field(4, ge=0)

    @model_validator(mode="after")
    def _random_needs_seed(self) -> SceneConfig:
        if self.kind == "random" and self.seed is None:
            raise ValueError("scene.kind='random' needs scene.seed so the layout is reproducible")
        return self


class RunConfig(_Strict):
    name: str = Field("run", min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    area: AreaConfig = AreaConfig()
    swarm: SwarmConfig = SwarmConfig()
    sim: SimConfig = SimConfig()
    channel: ChannelConfig = ChannelConfig()
    scorer: ScorerConfig = ScorerConfig()
    message: MessageConfig = MessageConfig()
    fusion: FusionConfig = FusionConfig()
    controller: ControllerConfig = ControllerConfig()
    prior: PriorConfig = PriorConfig()
    scene: SceneConfig = SceneConfig()

    @classmethod
    def load(cls, path: Path | str) -> RunConfig:
        path = Path(path)
        if path.suffix == ".toml":
            with path.open("rb") as fh:
                return cls.model_validate(_toml_loader().load(fh))
        if path.suffix == ".json":
            return cls.model_validate_json(path.read_text())
        raise ValueError(f"config {path} must be .toml or .json, got {path.suffix!r}")

    def dump_json(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")
