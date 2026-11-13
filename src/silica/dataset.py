"""Loading the public and gold rows of a dataset directory.

A dataset directory holds ``public/*.jsonl`` (model-facing rows),
``gold/*.jsonl`` (evaluator-only rows joined by ``id``), and the manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .util import read_jsonl


def _shards(root: Path, view: str) -> list[Path]:
    """Return the sorted JSONL shards of one view."""
    paths = sorted((root / view).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no {view} JSONL shards under {root}")
    return paths


def load_public_rows(root: Path) -> list[dict[str, Any]]:
    """Load every public row in shard order."""
    return [row for path in _shards(root, "public") for row in read_jsonl(path)]


def load_gold_by_id(root: Path) -> dict[str, dict[str, Any]]:
    """Load every gold row keyed by row id, rejecting duplicates."""
    gold: dict[str, dict[str, Any]] = {}
    for path in _shards(root, "gold"):
        for row in read_jsonl(path):
            identifier = row.get("id")
            if not isinstance(identifier, str) or identifier in gold:
                raise ValueError(f"{path}: missing or duplicate gold id {identifier!r}")
            if row.get("gold", {}).get("decision") not in ("ANSWER", "ABSTAIN"):
                raise ValueError(f"{path}: invalid gold decision for {identifier}")
            gold[identifier] = row
    return gold
