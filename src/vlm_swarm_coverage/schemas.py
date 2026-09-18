"""Typed messages between drones, with a binary wire format whose size is measurable.

Bytes on the channel are an experimental variable, so every message knows its own encoded size.
The wire format is a fixed little-endian header followed by a float32 payload; JSON is used only
for logging, never for transport, because JSON floats are ten bytes each and would make the byte
count a property of the encoder rather than of the message.

A `BeliefMessage` carries a drone's entire importance field (decision 030). That is a belief, not
an observation: a cell the sender never saw carries the sender's prior, indistinguishable from a
cell observed at the floor. Any fusion rule that needs to tell those apart needs an added mask.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_swarm_coverage.field import ImportanceField

if TYPE_CHECKING:
    from vlm_swarm_coverage.field import Grid

WIRE_DTYPE = np.dtype("<f4")
POSE_HEADER = struct.Struct("<IId4d")
BELIEF_HEADER = struct.Struct("<IIdII")


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
    """A drone's whole importance field, dense, row-major, float32 on the wire."""

    sender: int = Field(ge=0)
    seq: int = Field(ge=0)
    t: float
    rows: int = Field(ge=1)
    cols: int = Field(ge=1)
    values: tuple[float, ...]

    @model_validator(mode="after")
    def _values_fill_the_grid(self) -> BeliefMessage:
        if len(self.values) != self.rows * self.cols:
            raise ValueError(
                f"BeliefMessage has {len(self.values)} values for a {self.rows}x{self.cols} grid "
                f"({self.rows * self.cols} cells)"
            )
        return self

    @classmethod
    def from_field(cls, sender: int, seq: int, t: float, field: ImportanceField) -> BeliefMessage:
        rows, cols = field.grid.shape
        return cls(
            sender=sender, seq=seq, t=t, rows=rows, cols=cols,
            values=tuple(field.values.astype(WIRE_DTYPE).astype(np.float64).ravel().tolist()),
        )

    def to_field(self, grid: Grid) -> ImportanceField:
        if (self.rows, self.cols) != grid.shape:
            raise ValueError(
                f"message grid {self.rows}x{self.cols} does not match local grid "
                f"{grid.rows}x{grid.cols}; both drones must run the same area config"
            )
        return ImportanceField(grid, np.asarray(self.values, dtype=np.float64).reshape(grid.shape))

    def to_bytes(self) -> bytes:
        header = BELIEF_HEADER.pack(self.sender, self.seq, self.t, self.rows, self.cols)
        return header + np.asarray(self.values, dtype=WIRE_DTYPE).tobytes()

    @classmethod
    def from_bytes(cls, buf: bytes) -> BeliefMessage:
        size = BELIEF_HEADER.size
        if len(buf) < size:
            raise ValueError(f"BeliefMessage needs at least {size} header bytes, got {len(buf)}")
        sender, seq, t, rows, cols = BELIEF_HEADER.unpack(buf[:size])
        expected = size + rows * cols * WIRE_DTYPE.itemsize
        if len(buf) != expected:
            raise ValueError(
                f"BeliefMessage for a {rows}x{cols} grid needs {expected} bytes, got {len(buf)}"
            )
        values = np.frombuffer(buf[size:], dtype=WIRE_DTYPE).astype(np.float64)
        return cls(sender=sender, seq=seq, t=t, rows=rows, cols=cols, values=tuple(values.tolist()))
