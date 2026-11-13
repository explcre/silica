"""Shared fixtures: a small, deterministic slice of the release."""

from __future__ import annotations

from pathlib import Path

import pytest

from silica.generator import build_records, shard_name
from silica.util import write_jsonl

TEMPLATES = (
    "parity_guard-eval-00",
    "parity_guard-eval-01",
    "selector_multiplexer-eval-00",
    "selector_multiplexer-eval-01",
)


@pytest.fixture(scope="session")
def small_release() -> tuple[list[dict], dict[str, dict]]:
    """Return the public rows and the gold rows by id of two evaluation templates."""
    public, gold = build_records(template_ids=TEMPLATES)
    return public, {row["id"]: row for row in gold}


@pytest.fixture(scope="session")
def small_dataset(small_release, tmp_path_factory) -> Path:
    """Write the small release as a dataset directory."""
    public, gold_by_id = small_release
    root = tmp_path_factory.mktemp("dataset")
    for view in ("public", "gold"):
        (root / view).mkdir()
    for family in {row["mechanism"]["family"] for row in public}:
        rows = [row for row in public if row["mechanism"]["family"] == family]
        write_jsonl(root / "public" / shard_name(family), rows)
        write_jsonl(root / "gold" / shard_name(family), [gold_by_id[row["id"]] for row in rows])
    return root
