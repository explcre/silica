"""Checks against the downloaded release; skipped unless SILICA_DATASET points to it."""

import json
import os
from pathlib import Path

import pytest

from silica.cohort import COMMON_COHORT_ROWS, select_cohort
from silica.dataset import load_public_rows
from silica.evaluate import evaluate_predictions
from silica.prompts import prompt_ledger_identity, request_prompt_manifest
from silica.util import file_sha256

ROOT = Path(os.environ.get("SILICA_DATASET") or "missing-dataset")
pytestmark = pytest.mark.skipif(
    not (ROOT / "SHA256SUMS").is_file(),
    reason="set SILICA_DATASET to a downloaded SILICA dataset directory",
)
# Identities of the evaluated release, as printed in the paper.
COMMON_COHORT_HASH = "7956f0fb7270c983bb27a76ace8f2130c370bd9db64e37cbd191bfe7ead51fdc"
COMMON_REQUEST_MANIFEST = "7949317955743f2150e1e320235760832ba492e2f655a4d219215b1c2a11b605"
PROMPT_LEDGER = "cf36ed363351ffed8990c9f2ff14f0955b6ad3eca08afbcdca6b7c42a62efb98"


@pytest.fixture(scope="module")
def public_rows():
    """Load the released public rows once."""
    return load_public_rows(ROOT)


def test_checksums():
    """Every file listed in SHA256SUMS has the listed digest."""
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    for line in lines:
        digest, name = line.split("  ", 1)
        assert file_sha256(ROOT / name) == digest, name


def test_prompts_and_common_cohort(public_rows):
    """The renderer and the cohort selector reproduce the evaluated identities."""
    assert prompt_ledger_identity(public_rows) == PROMPT_LEDGER
    selection = select_cohort(public_rows)
    assert len(selection.items) == COMMON_COHORT_ROWS
    assert selection.cohort_hash == COMMON_COHORT_HASH
    assert request_prompt_manifest(selection.items) == COMMON_REQUEST_MANIFEST


def _conditions():
    """Return the released conditions, or skip if the predictions are absent."""
    path = ROOT / "predictions" / "conditions.json"
    if not path.is_file():
        pytest.skip("the dataset directory has no released predictions")
    return json.loads(path.read_text(encoding="utf-8"))["conditions"]


def test_released_predictions_match_their_checksums():
    """Every released predictions file has the SHA-256 listed in conditions.json."""
    for condition in _conditions():
        assert file_sha256(ROOT / condition["predictions_file"]) == condition["predictions_sha256"]


def test_a_released_condition_evaluates():
    """The always-answering Qwen2.5-7B panel condition scores as the paper reports."""
    if not any(item["condition_id"] == "qwen25-7b" for item in _conditions()):
        pytest.skip("qwen25-7b is not released here")
    report = evaluate_predictions(
        ROOT, ROOT / "predictions" / "qwen25-7b.jsonl", bootstrap_replicates=0
    )
    rows = report["rows"]
    assert report["cohort"]["cohort_hash"] == COMMON_COHORT_HASH
    assert rows["format_valid_rate"] == rows["answer_rate"] == 1.0
    assert rows["decision_signal"] == 0.0
