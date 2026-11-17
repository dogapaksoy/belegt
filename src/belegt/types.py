"""Core data types: what an extraction claims, and what the verifier concluded."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ValueType(str, Enum):
    """How the claimed value is checked against the located source span."""

    DATE = "date"  # compared as calendar dates, any German or ISO format
    AMOUNT = "amount"  # compared as decimal money amounts
    ID = "id"  # reference numbers; compared alphanumerically
    TEXT = "text"  # free text; only the quote itself is checked


class Reason(str, Enum):
    OK = "ok"
    # Supported, but the value only matched after tolerating one OCR-dropped character.
    # Applications should show these to the user for a quick manual check.
    VALUE_APPROX = "value_approx"
    QUOTE_NOT_FOUND = "quote_not_found"
    VALUE_NOT_IN_SOURCE = "value_not_in_source"
    INVALID_VALUE = "invalid_value"
    # The quote was located, but a semantic (NLI) check found that the source says
    # something different, e.g. the claim inserted or dropped a negation.
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class Claim:
    """One fact extracted by a model, together with the source quote it cites."""

    field: str
    value: str
    quote: str
    type: ValueType = ValueType.TEXT

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        return cls(
            field=str(data["field"]),
            value=str(data["value"]),
            quote=str(data["quote"]),
            type=ValueType(data.get("type", ValueType.TEXT.value)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "quote": self.quote,
            "type": self.type.value,
        }


@dataclass(frozen=True)
class Span:
    """A region of the original document, with the match score (0-100)."""

    start: int
    end: int
    text: str
    score: float


@dataclass(frozen=True)
class Verification:
    claim: Claim
    supported: bool
    reason: Reason
    span: Span | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "claim": self.claim.to_dict(),
            "supported": self.supported,
            "reason": self.reason.value,
        }
        if self.span is not None:
            out["span"] = {
                "start": self.span.start,
                "end": self.span.end,
                "text": self.span.text,
                "score": round(self.span.score, 2),
            }
        return out
