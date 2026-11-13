"""Round trip through the prediction format and ``silica evaluate``."""

import json

import pytest

from silica.cli import main
from silica.dataset import load_gold_by_id, load_public_rows
from silica.evaluate import evaluate_rows
from silica.predictions import Prediction, load_predictions, write_predictions


def _gold_predictions(public, gold_by_id):
    """Return exact predictions, with one request error and one truncated reply."""
    predictions = []
    for index, row in enumerate(public):
        gold = gold_by_id[row["id"]]["gold"]
        text = json.dumps({"decision": gold["decision"], "answer": gold["answer"]})
        if index == 0:
            predictions.append(Prediction(row["id"], None, None, None, True))
        elif index == 1:
            predictions.append(Prediction(row["id"], text, "length", 384, False))
        else:
            predictions.append(
                Prediction(row["id"], text, "stop", 12, False, reasoning_text="trace")
            )
    return predictions


def test_evaluate_round_trip(small_dataset, tmp_path, capsys):
    """Predictions written to disk and evaluated by the CLI match in-memory evaluation."""
    public, gold_by_id = load_public_rows(small_dataset), load_gold_by_id(small_dataset)
    predictions = _gold_predictions(public, gold_by_id)
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, predictions)
    assert load_predictions(path) == predictions
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--dataset",
                str(small_dataset),
                "--predictions",
                str(path),
                "--output",
                str(output),
                "--bootstrap-replicates",
                "20",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == report
    expected = evaluate_rows(public, gold_by_id, predictions, bootstrap_replicates=20)
    assert {key: report[key] for key in expected} == json.loads(json.dumps(expected))
    assert report["cohort"] == {"name": "full", "rows": len(public)}
    rows = report["rows"]
    assert (
        rows["request_error_count"],
        rows["format_error_count"],
        rows["format_valid_count"],
    ) == (1, 1, len(public) - 2)
    assert report["template_cluster_bootstrap"]["available"] is True


def test_predictions_must_cover_whole_units(small_dataset, tmp_path):
    """Dropping one row breaks a paired unit and is refused."""
    public, gold_by_id = load_public_rows(small_dataset), load_gold_by_id(small_dataset)
    with pytest.raises(ValueError):
        evaluate_rows(
            public, gold_by_id, _gold_predictions(public, gold_by_id)[1:], bootstrap_replicates=0
        )


@pytest.mark.parametrize(
    "record",
    [
        {"id": "x", "response_text": "", "finish_reason": "stop", "completion_tokens": 1},
        {
            "id": "x",
            "response_text": "",
            "finish_reason": "stop",
            "completion_tokens": 1,
            "request_error": False,
            "extra": 1,
        },
        {
            "id": "x",
            "response_text": "",
            "finish_reason": "eos",
            "completion_tokens": 1,
            "request_error": False,
        },
        {
            "id": "x",
            "response_text": "text",
            "finish_reason": None,
            "completion_tokens": None,
            "request_error": True,
        },
    ],
)
def test_prediction_format_is_strict(record):
    """Missing, unknown or inconsistent prediction fields are rejected."""
    with pytest.raises(ValueError):
        Prediction.from_dict(record)
