"""How a belief becomes a message: the sender-side encoders (decision 033).

The channel carries bytes and does not care what they mean; the fusion rule consumes fields and
does not care how they arrived. The encoder sits between: given the drone's belief, produce the
message to send, or `None` to send no belief at all (poses only). Four encoders:

- `DenseEncoder`: the whole field, float32 value and uint16 age per cell.
- `TopKEncoder(k)`: the k observed cells with the highest values, 8 bytes each. Keeps peaks.
- `QuantisedEncoder`: every cell at one byte of value and one of age. Keeps shape.
- `PosesOnly`: nothing. The drone shares where it is and not what it believes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from vlm_swarm_coverage.schemas import BeliefMessage, QuantisedBeliefMessage, SparseBeliefMessage

if TYPE_CHECKING:
    from vlm_swarm_coverage.config import MessageConfig
    from vlm_swarm_coverage.field import ImportanceField
    from vlm_swarm_coverage.schemas import AnyBeliefMessage


class BeliefEncoder(Protocol):
    def encode(self, sender: int, seq: int, t: float, field: ImportanceField) -> AnyBeliefMessage | None: ...


class DenseEncoder:
    def encode(self, sender: int, seq: int, t: float, field: ImportanceField) -> AnyBeliefMessage | None:
        return BeliefMessage.from_field(sender, seq, t, field)


class TopKEncoder:
    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError(f"TopKEncoder needs k >= 0, got {k}")
        self.k = k

    def encode(self, sender: int, seq: int, t: float, field: ImportanceField) -> AnyBeliefMessage | None:
        return SparseBeliefMessage.top_k(sender, seq, t, field, self.k)


class QuantisedEncoder:
    def encode(self, sender: int, seq: int, t: float, field: ImportanceField) -> AnyBeliefMessage | None:
        return QuantisedBeliefMessage.from_field(sender, seq, t, field)


class PosesOnly:
    def encode(self, sender: int, seq: int, t: float, field: ImportanceField) -> AnyBeliefMessage | None:
        return None


def build_encoder(cfg: MessageConfig) -> BeliefEncoder:
    if cfg.kind == "dense":
        return DenseEncoder()
    if cfg.kind == "topk":
        assert cfg.k is not None
        return TopKEncoder(cfg.k)
    if cfg.kind == "quantised":
        return QuantisedEncoder()
    return PosesOnly()
