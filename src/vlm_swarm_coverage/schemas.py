"""Typed messages between drones, with a binary wire format whose size is measurable.

Bytes on the channel are an experimental variable, so every message knows its own encoded size.
The wire format is a fixed little-endian header followed by a float32 payload; JSON is used only
for logging, never for transport, because JSON floats are ten bytes each and would make the byte
count a property of the encoder rather than of the message.

A `BeliefMessage` carries a drone's entire importance field (decision 030) and, per cell, the age
of the observation behind the value (decision 032): tenths of a second before the message time as
a uint16, with 65535 meaning the cell was never observed and still holds the sender's prior. Ages
rather than absolute stamps keep the message self-contained and the numbers small. A receiver can
therefore tell "observed at the floor" from "never seen", and weigh a value by how stale it is.

Two smaller belief messages exist for the bandwidth axis (decision 033): `SparseBeliefMessage`
carries only k chosen cells (index, value, age), and `QuantisedBeliefMessage` carries every cell at
one byte for the value and one for the age. All three decode to an `ImportanceField` with stamps,
so the fusion rule does not know which one arrived.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_swarm_coverage.field import ImportanceField
from vlm_swarm_coverage.scoring import Observation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from vlm_swarm_coverage.field import Grid

WIRE_DTYPE = np.dtype("<f4")
AGE_DTYPE = np.dtype("<u2")
AGE_UNIT_S = 0.1
AGE_NEVER = 65535
POSE_HEADER = struct.Struct("<IId4d")
BELIEF_HEADER = struct.Struct("<IIdII")
SPARSE_HEADER = struct.Struct("<IIdIII")
QUANT_HEADER = struct.Struct("<IIdIIf")
OBS_HEADER = struct.Struct("<IIdI")
CELL_DTYPE = np.dtype("<i2")
INDEX_DTYPE = np.dtype("<u2")
QUANT_DTYPE = np.dtype("<u1")
QUANT_LEVELS = 255
QUANT_AGE_NEVER = 255


def encode_ages(stamps: NDArray[np.float64] | None, t: float, shape: tuple[int, int]) -> NDArray[np.uint16]:
    """Observation stamps -> ages in tenths of a second before `t`; NaN or no stamps -> never."""
    if stamps is None:
        return np.full(shape, AGE_NEVER, dtype=AGE_DTYPE)
    with np.errstate(invalid="ignore"):
        age = np.rint((t - stamps) / AGE_UNIT_S)
    age = np.where(np.isnan(stamps), AGE_NEVER, np.clip(age, 0, AGE_NEVER - 1))
    return age.astype(AGE_DTYPE)


def decode_ages(ages: NDArray[np.integer], t: float) -> NDArray[np.float64]:
    """Ages -> observation stamps on the receiver's clock; never -> NaN."""
    ages = np.asarray(ages, dtype=np.int64)
    return np.where(ages == AGE_NEVER, np.nan, t - ages * AGE_UNIT_S)


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def nbytes(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        raise NotImplementedError


class PoseMessage(_Wire):
    """Where a drone is. The geometric-only content a coverage controller needs from a peer."""

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    x: float
    y: float
    z: float
    yaw: float = 0.0

    def to_bytes(self) -> bytes:
        return POSE_HEADER.pack(self.sender, self.seq, self.t, self.x, self.y, self.z, self.yaw)

    @classmethod
    def from_bytes(cls, buf: bytes) -> PoseMessage:
        if len(buf) != POSE_HEADER.size:
            raise ValueError(f"PoseMessage needs {POSE_HEADER.size} bytes, got {len(buf)}")
        sender, seq, t, x, y, z, yaw = POSE_HEADER.unpack(buf)
        return cls(sender=sender, seq=seq, t=t, x=x, y=y, z=z, yaw=yaw)


class BeliefMessage(_Wire):
    """A drone's whole importance field, dense, row-major: float32 values then uint16 ages."""

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    values: tuple[float, ...]
    ages: tuple[int, ...]

    @model_validator(mode="before")
    @classmethod
    def _ages_default_to_never(cls, data: object) -> object:
        if isinstance(data, dict) and "ages" not in data and "values" in data:
            data = {**data, "ages": (AGE_NEVER,) * len(data["values"])}
        return data

    @model_validator(mode="after")
    def _values_fill_the_grid(self) -> BeliefMessage:
        n = self.rows * self.cols
        if len(self.values) != n:
            raise ValueError(
                f"BeliefMessage has {len(self.values)} values for a {self.rows}x{self.cols} grid "
                f"({n} cells)"
            )
        if len(self.ages) != n:
            raise ValueError(f"BeliefMessage has {len(self.ages)} ages for {n} cells")
        if any(a < 0 or a > AGE_NEVER for a in self.ages):
            raise ValueError(f"BeliefMessage ages must lie in [0, {AGE_NEVER}]")
        return self

    @classmethod
    def from_field(cls, sender: int, seq: int, t: float, field: ImportanceField) -> BeliefMessage:
        rows, cols = field.grid.shape
        return cls(
            sender=sender, seq=seq, t=t, rows=rows, cols=cols,
            values=tuple(field.values.astype(WIRE_DTYPE).astype(np.float64).ravel().tolist()),
            ages=tuple(int(a) for a in encode_ages(field.stamps, t, (rows, cols)).ravel()),
        )

    def to_field(self, grid: Grid) -> ImportanceField:
        """The sender's field on the receiver's grid, stamps recovered from the ages."""
        if (self.rows, self.cols) != grid.shape:
            raise ValueError(
                f"message grid {self.rows}x{self.cols} does not match local grid "
                f"{grid.rows}x{grid.cols}; both drones must run the same area config"
            )
        values = np.asarray(self.values, dtype=np.float64).reshape(grid.shape)
        stamps = decode_ages(np.asarray(self.ages).reshape(grid.shape), self.t)
        return ImportanceField(grid, values, stamps)

    def to_bytes(self) -> bytes:
        header = BELIEF_HEADER.pack(self.sender, self.seq, self.t, self.rows, self.cols)
        return (header + np.asarray(self.values, dtype=WIRE_DTYPE).tobytes()
                + np.asarray(self.ages, dtype=AGE_DTYPE).tobytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> BeliefMessage:
        size = BELIEF_HEADER.size
        if len(buf) < size:
            raise ValueError(f"BeliefMessage needs at least {size} header bytes, got {len(buf)}")
        sender, seq, t, rows, cols = BELIEF_HEADER.unpack(buf[:size])
        n = rows * cols
        expected = size + n * (WIRE_DTYPE.itemsize + AGE_DTYPE.itemsize)
        if len(buf) != expected:
            raise ValueError(
                f"BeliefMessage for a {rows}x{cols} grid needs {expected} bytes, got {len(buf)}"
            )
        split = size + n * WIRE_DTYPE.itemsize
        values = np.frombuffer(buf[size:split], dtype=WIRE_DTYPE).astype(np.float64)
        ages = np.frombuffer(buf[split:], dtype=AGE_DTYPE)
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols,
                   values=tuple(values.tolist()), ages=tuple(int(a) for a in ages))


class SparseBeliefMessage(_Wire):
    """k cells of a belief: row-major cell index (uint16), value (float32), age (uint16).

    The sender chooses which cells; the wire does not care. Cells not present decode as never
    observed, so a receiver's fusion rule ignores them exactly as it ignores a peer's prior.
    """

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    cells: tuple[int, ...]
    values: tuple[float, ...]
    ages: tuple[int, ...]

    @model_validator(mode="after")
    def _consistent(self) -> SparseBeliefMessage:
        n = self.rows * self.cols
        if n > 65536:
            raise ValueError(f"sparse messages index cells as uint16; a {self.rows}x{self.cols} grid has {n} cells")
        if not (len(self.cells) == len(self.values) == len(self.ages)):
            raise ValueError(
                f"SparseBeliefMessage has {len(self.cells)} cells, {len(self.values)} values, {len(self.ages)} ages"
            )
        if len(set(self.cells)) != len(self.cells):
            raise ValueError("SparseBeliefMessage cells must be unique")
        if any(c < 0 or c >= n for c in self.cells):
            raise ValueError(f"SparseBeliefMessage cell indices must lie in [0, {n})")
        if any(a < 0 or a > AGE_NEVER for a in self.ages):
            raise ValueError(f"SparseBeliefMessage ages must lie in [0, {AGE_NEVER}]")
        return self

    @classmethod
    def top_k(cls, sender: int, seq: int, t: float, field: ImportanceField, k: int) -> SparseBeliefMessage:
        """The k observed cells with the highest values (ties broken by cell index). Fewer if the
        sender has observed fewer than k cells; none if it has observed none."""
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        rows, cols = field.grid.shape
        flat_values = field.values.astype(WIRE_DTYPE).astype(np.float64).ravel()
        observed = np.flatnonzero(field.observed.ravel())
        order = observed[np.lexsort((observed, -flat_values[observed]))][:k]
        ages = encode_ages(field.stamps, t, (rows, cols)).ravel()
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols,
                   cells=tuple(int(i) for i in order), values=tuple(float(flat_values[i]) for i in order),
                   ages=tuple(int(ages[i]) for i in order))

    def to_field(self, grid: Grid) -> ImportanceField:
        if (self.rows, self.cols) != grid.shape:
            raise ValueError(
                f"message grid {self.rows}x{self.cols} does not match local grid "
                f"{grid.rows}x{grid.cols}; both drones must run the same area config"
            )
        values = np.zeros(grid.n_cells)
        ages = np.full(grid.n_cells, AGE_NEVER, dtype=np.int64)
        idx = np.asarray(self.cells, dtype=np.int64)
        values[idx] = self.values
        ages[idx] = self.ages
        return ImportanceField(grid, values.reshape(grid.shape), decode_ages(ages, self.t).reshape(grid.shape))

    def to_bytes(self) -> bytes:
        header = SPARSE_HEADER.pack(self.sender, self.seq, self.t, self.rows, self.cols, len(self.cells))
        return (header + np.asarray(self.cells, dtype=INDEX_DTYPE).tobytes()
                + np.asarray(self.values, dtype=WIRE_DTYPE).tobytes()
                + np.asarray(self.ages, dtype=AGE_DTYPE).tobytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> SparseBeliefMessage:
        size = SPARSE_HEADER.size
        if len(buf) < size:
            raise ValueError(f"SparseBeliefMessage needs at least {size} header bytes, got {len(buf)}")
        sender, seq, t, rows, cols, k = SPARSE_HEADER.unpack(buf[:size])
        expected = size + k * (INDEX_DTYPE.itemsize + WIRE_DTYPE.itemsize + AGE_DTYPE.itemsize)
        if len(buf) != expected:
            raise ValueError(f"SparseBeliefMessage with {k} cells needs {expected} bytes, got {len(buf)}")
        a = size + k * INDEX_DTYPE.itemsize
        b = a + k * WIRE_DTYPE.itemsize
        cells = np.frombuffer(buf[size:a], dtype=INDEX_DTYPE)
        values = np.frombuffer(buf[a:b], dtype=WIRE_DTYPE).astype(np.float64)
        ages = np.frombuffer(buf[b:], dtype=AGE_DTYPE)
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols, cells=tuple(int(c) for c in cells),
                   values=tuple(values.tolist()), ages=tuple(int(x) for x in ages))


class QuantisedBeliefMessage(_Wire):
    """Every cell at two bytes: the value as a uint8 fraction of a per-message scale, the age as
    whole seconds in a uint8 (255 = never observed). Keeps the shape of the field at a quarter
    of the dense message's cost, at the price of value precision and age resolution."""

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    scale: float = Field(gt=0)
    levels: tuple[int, ...]
    ages: tuple[int, ...]

    @model_validator(mode="after")
    def _consistent(self) -> QuantisedBeliefMessage:
        n = self.rows * self.cols
        if len(self.levels) != n or len(self.ages) != n:
            raise ValueError(f"QuantisedBeliefMessage needs {n} levels and ages, got {len(self.levels)} and {len(self.ages)}")
        if any(v < 0 or v > QUANT_LEVELS for v in self.levels) or any(a < 0 or a > QUANT_AGE_NEVER for a in self.ages):
            raise ValueError("QuantisedBeliefMessage levels and ages must lie in [0, 255]")
        return self

    @classmethod
    def from_field(cls, sender: int, seq: int, t: float, field: ImportanceField) -> QuantisedBeliefMessage:
        rows, cols = field.grid.shape
        v = np.clip(field.values, 0.0, None)
        scale = float(v.max()) if v.max() > 0 else 1.0
        levels = np.rint(v / scale * QUANT_LEVELS).astype(np.int64).ravel()
        fine = encode_ages(field.stamps, t, (rows, cols)).ravel().astype(np.int64)
        ages = np.where(fine == AGE_NEVER, QUANT_AGE_NEVER, np.minimum(np.rint(fine * AGE_UNIT_S), QUANT_AGE_NEVER - 1))
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols, scale=scale,
                   levels=tuple(int(x) for x in levels), ages=tuple(int(x) for x in ages))

    def to_field(self, grid: Grid) -> ImportanceField:
        if (self.rows, self.cols) != grid.shape:
            raise ValueError(
                f"message grid {self.rows}x{self.cols} does not match local grid "
                f"{grid.rows}x{grid.cols}; both drones must run the same area config"
            )
        values = np.asarray(self.levels, dtype=np.float64).reshape(grid.shape) / QUANT_LEVELS * self.scale
        ages = np.asarray(self.ages, dtype=np.int64).reshape(grid.shape)
        stamps = np.where(ages == QUANT_AGE_NEVER, np.nan, self.t - ages.astype(np.float64))
        return ImportanceField(grid, values, stamps)

    def to_bytes(self) -> bytes:
        header = QUANT_HEADER.pack(self.sender, self.seq, self.t, self.rows, self.cols, self.scale)
        return (header + np.asarray(self.levels, dtype=QUANT_DTYPE).tobytes()
                + np.asarray(self.ages, dtype=QUANT_DTYPE).tobytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> QuantisedBeliefMessage:
        size = QUANT_HEADER.size
        if len(buf) < size:
            raise ValueError(f"QuantisedBeliefMessage needs at least {size} header bytes, got {len(buf)}")
        sender, seq, t, rows, cols, scale = QUANT_HEADER.unpack(buf[:size])
        n = rows * cols
        if len(buf) != size + 2 * n:
            raise ValueError(f"QuantisedBeliefMessage for a {rows}x{cols} grid needs {size + 2 * n} bytes, got {len(buf)}")
        levels = np.frombuffer(buf[size:size + n], dtype=QUANT_DTYPE)
        ages = np.frombuffer(buf[size + n:], dtype=QUANT_DTYPE)
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols, scale=scale,
                   levels=tuple(int(x) for x in levels), ages=tuple(int(x) for x in ages))


AnyBeliefMessage = BeliefMessage | SparseBeliefMessage | QuantisedBeliefMessage


class ObservationMessage(_Wire):
    """One scorer output crossing a host boundary: the cells seen and their values.

    Not a drone-to-drone message; it carries a scorer's result from wherever the camera and the
    model live to wherever the belief lives. Cells are int16 (row, col) pairs, values float32.
    """

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    cells: tuple[tuple[int, int], ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def _lengths_agree(self) -> ObservationMessage:
        if len(self.cells) != len(self.values):
            raise ValueError(f"ObservationMessage has {len(self.cells)} cells but {len(self.values)} values")
        return self

    @classmethod
    def from_observation(cls, obs: Observation, seq: int) -> ObservationMessage:
        cells = np.asarray(obs.cells, dtype=CELL_DTYPE)
        values = np.asarray(obs.values, dtype=WIRE_DTYPE).astype(np.float64)
        return cls(sender=obs.drone_id, seq=seq, t=obs.t,
                   cells=tuple((int(r), int(c)) for r, c in cells), values=tuple(values.tolist()))

    def to_observation(self) -> Observation:
        return Observation(self.sender, self.t, np.asarray(self.cells, dtype=np.int64).reshape(-1, 2),
                           np.asarray(self.values, dtype=np.float64))

    def to_bytes(self) -> bytes:
        header = OBS_HEADER.pack(self.sender, self.seq, self.t, len(self.cells))
        cells = np.asarray(self.cells, dtype=CELL_DTYPE).tobytes()
        return header + cells + np.asarray(self.values, dtype=WIRE_DTYPE).tobytes()

    @classmethod
    def from_bytes(cls, buf: bytes) -> ObservationMessage:
        size = OBS_HEADER.size
        if len(buf) < size:
            raise ValueError(f"ObservationMessage needs at least {size} header bytes, got {len(buf)}")
        sender, seq, t, k = OBS_HEADER.unpack(buf[:size])
        expected = size + k * (2 * CELL_DTYPE.itemsize + WIRE_DTYPE.itemsize)
        if len(buf) != expected:
            raise ValueError(f"ObservationMessage with {k} cells needs {expected} bytes, got {len(buf)}")
        cells = np.frombuffer(buf[size:size + 2 * k * CELL_DTYPE.itemsize], dtype=CELL_DTYPE).reshape(-1, 2)
        values = np.frombuffer(buf[size + 2 * k * CELL_DTYPE.itemsize:], dtype=WIRE_DTYPE).astype(np.float64)
        return cls(sender=sender, seq=seq, t=t, cells=tuple((int(r), int(c)) for r, c in cells), values=tuple(values.tolist()))
