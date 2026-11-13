"""Canonical JSON, hashing, and strict JSON/JSONL input and output.

Every identifier and checksum in SILICA is the SHA-256 of one canonical JSON
serialization: sorted keys, no insignificant whitespace, UTF-8 without ASCII
escaping, and no NaN or infinity.  This module is the only place that defines
that serialization.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text used by every SILICA hash."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    """Return the hexadecimal SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def object_hash(value: Any) -> str:
    """Return the SHA-256 digest of a value's canonical JSON."""
    return sha256_text(canonical_json(value))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object, rejecting repeated keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    """Reject the non-standard JSON constants NaN, Infinity and -Infinity."""
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    """Parse a JSON number, rejecting values that overflow to infinity."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number: {value}")
    return parsed


def strict_loads(text: str) -> Any:
    """Parse standard JSON with unique keys and finite numbers only."""
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def read_json(path: Path) -> dict[str, Any]:
    """Read one strict JSON object from a file."""
    value = strict_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path, *, canonical: bool = False) -> list[dict[str, Any]]:
    """Read one strict JSON object per line, optionally requiring canonical lines."""
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            try:
                value = strict_loads(line)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if canonical and line != canonical_json(value) + "\n":
                raise ValueError(f"{path}:{line_number}: noncanonical JSONL record")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    """Write one value as canonical JSON followed by a newline."""
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write canonical JSON lines and return the number of records written."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(dict(record)) + "\n")
            count += 1
    return count
