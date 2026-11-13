"""Metric identities of constant and exact policies."""

import json

import pytest

from silica.evaluate import evaluate_rows
from silica.predictions import Prediction


def _evidence_blind_answer(row):
    """Return the first declared value of the queried axis, ignoring the observations."""
    declaration, query = row["hypothesis_declaration"], row["query"]
    if row["task"] == "grammar_feature":
        axis = next(
            item for item in declaration["grammar_axes"] if item["name"] == query["feature"]
        )
        return axis["values"][0]
    axis = next(item for item in declaration["lexical_axes"] if query["form"] in item["forms"])
    return axis["possible_meanings"][0]


def _evaluate(small_release, respond):
    """Evaluate a policy given as a function from a row to a response object."""
    public, gold_by_id = small_release
    predictions = [
        Prediction(row["id"], json.dumps(respond(row, gold_by_id[row["id"]])), "stop", 1, False)
        for row in public
    ]
    return evaluate_rows(public, gold_by_id, predictions, bootstrap_replicates=0)


def test_always_abstain(small_release):
    """Abstaining everywhere scores exactly one half on BDA and BJA."""
    report = _evaluate(small_release, lambda row, gold: {"decision": "ABSTAIN", "answer": None})
    rows = report["rows"]
    assert rows["balanced_decision_accuracy"] == rows["balanced_strict_joint_accuracy"] == 0.5
    assert (
        rows["answerable_decision_recall"] == 0.0 and rows["conditional_payload_accuracy"] is None
    )


def test_always_answer_with_an_evidence_blind_payload(small_release):
    """Answering everywhere without reading the evidence gets one payload per unit right."""
    report = _evaluate(
        small_release,
        lambda row, gold: {"decision": "ANSWER", "answer": _evidence_blind_answer(row)},
    )
    rows = report["rows"]
    assert rows["answerable_decision_recall"] == rows["false_certainty_rate"] == 1.0
    assert rows["decision_signal"] == 0.0
    assert rows["answerable_exact_payload_accuracy"] == rows["conditional_payload_accuracy"] == 0.5
    assert rows["balanced_strict_joint_accuracy"] == pytest.approx(0.25, abs=0)
    assert report["paired_query_group_endpoints"]["paired_decision_pattern_exact_count"] == 0


def test_exact_oracle(small_release):
    """Answering with the gold decision and payload attains every endpoint."""
    report = _evaluate(
        small_release, lambda row, gold: {key: gold["gold"][key] for key in ("decision", "answer")}
    )
    assert report["rows"]["balanced_strict_joint_accuracy"] == 1.0
    assert report["rows"]["decision_signal"] == 1.0
    assert report["unit_counterfactual_consistency"]["unit_strict_joint_pattern_exact_rate"] == 1.0
    assert report["paired_query_group_endpoints"]["paired_strict_joint_pattern_exact_rate"] == 1.0


def test_invalid_responses_are_wrong_not_skipped(small_release):
    """A malformed response counts against the policy rather than being dropped."""
    public, gold_by_id = small_release
    predictions = [Prediction(row["id"], "ABSTAIN", "stop", 1, False) for row in public]
    rows = evaluate_rows(public, gold_by_id, predictions, bootstrap_replicates=0)["rows"]
    assert rows["n"] == len(public) and rows["format_valid_rate"] == 0.0
    assert rows["balanced_decision_accuracy"] == 0.0
