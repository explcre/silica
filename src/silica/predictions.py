"""The release prediction format: the input of ``silica evaluate``.

A predictions file is JSON Lines with one object per evaluated row::

    {"id": "silica-...",            # the public row id
     "response_text": "...",        # final content string, or null
     "finish_reason": "stop",       # "stop", "length", or null
     "completion_tokens": 18,       # integer, or null when unknown
     "request_error": false,        # true if no response was obtained
     "reasoning_text": "..."}       # optional; recorded, never scored

``response_text`` is the final content only.  A reasoning trace returned
separately by the serving stack belongs in ``reasoning_text``.  A row is
scored as format-valid only if ``request_error`` is false, ``finish_reason``
is ``"stop"``, and ``response_text`` passes the strict parser.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import read_jsonl, write_jsonl

FINISH_REASONS = ("stop", "length", None)
_REQUIRED = frozenset(
    {"id", "response_text", "finish_reason", "completion_tokens", "request_error"}
)
_OPTIONAL = frozenset({"reasoning_text"})


@dataclass(frozen=True)
class Prediction:
    """One model response to one public row."""

    id: str
    response_text: str | None
    finish_reason: str | None
    completion_tokens: int | None
    request_error: bool
    reasoning_text: str | None = None

    def __post_init__(self) -> None:
        """Validate every field against the release format."""
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("prediction id must be a non-empty string")
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise ValueError(f"{self.id}: response_text must be a string or null")
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"{self.id}: finish_reason must be 'stop', 'length' or null")
        tokens = self.completion_tokens
        if tokens is not None and (
            not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0
        ):
            raise ValueError(f"{self.id}: completion_tokens must be a nonnegative integer or null")
        if not isinstance(self.request_error, bool):
            raise ValueError(f"{self.id}: request_error must be a Boolean")
        if self.request_error and self.response_text is not None:
            raise ValueError(f"{self.id}: a request error carries no response_text")
        if self.reasoning_text is not None and not isinstance(self.reasoning_text, str):
            raise ValueError(f"{self.id}: reasoning_text must be a string or null")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Prediction:
        """Parse one JSON object, rejecting missing or unknown fields."""
        keys = set(value)
        if not _REQUIRED <= keys or keys - _REQUIRED - _OPTIONAL:
            raise ValueError(
                f"prediction fields must be {sorted(_REQUIRED)} plus optional {sorted(_OPTIONAL)}; "
                f"found {sorted(keys)}"
            )
        return cls(**{key: value[key] for key in keys})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the prediction, omitting an absent reasoning trace."""
        value = {
            "id": self.id,
            "response_text": self.response_text,
            "finish_reason": self.finish_reason,
            "completion_tokens": self.completion_tokens,
            "request_error": self.request_error,
        }
        if self.reasoning_text is not None:
            value["reasoning_text"] = self.reasoning_text
        return value


def load_predictions(path: Path) -> list[Prediction]:
    """Read a predictions file, rejecting duplicate row ids."""
    predictions = [Prediction.from_dict(record) for record in read_jsonl(path)]
    identifiers = [item.id for item in predictions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate prediction ids")
    if not predictions:
        raise ValueError(f"{path}: no predictions")
    return predictions


def write_predictions(path: Path, predictions: Iterable[Prediction]) -> int:
    """Write predictions as canonical JSON lines and return the count."""
    return write_jsonl(path, (prediction.to_dict() for prediction in predictions))
