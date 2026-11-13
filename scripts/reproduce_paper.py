#!/usr/bin/env python3
"""Reproduce the quantitative tables and key numbers of the SILICA paper.

Every released model-response condition in a downloaded dataset tree is scored
with :func:`silica.evaluate.evaluate_predictions` (the partial Phi-3.5 sample
row by row with :func:`silica.evaluate.score_rows`).  Each value is formatted
the way the paper prints it, the tables are printed, and every value is
compared with ``paper_values.json``, the values printed in the camera-ready
paper.  The exit status is 1 if any value differs.

Usage::

    python scripts/reproduce_paper.py --hf silica-data

where ``silica-data`` holds the dataset files and ``predictions/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean
from typing import Any

from silica.cohort import select_cohort
from silica.dataset import load_gold_by_id, load_public_rows
from silica.evaluate import evaluate_predictions, evaluate_rows, score_rows
from silica.predictions import Prediction, load_predictions
from silica.util import file_sha256

VALUES = Path(__file__).with_name("paper_values.json")
PANEL = (
    "qwen25-05b",
    "qwen25-15b",
    "smollm2-17b",
    "qwen25-3b",
    "phi35-mini",
    "qwen25-7b",
    "qwen25-14b",
    "qwen25-32b",
)
MATCHED = (
    "qwen3-235b-trace-off",
    "qwen3-235b-trace-on",
    "qwen3-17b-trace-off",
    "qwen3-17b-trace-on",
    "qwen35-2b-greedy",
    "qwen35-2b-stochastic",
    "qwen25-14b-ablation-verbatim",
    "qwen25-14b-ablation-scratchpad",
    "qwen25-32b-ablation-verbatim",
    "qwen25-32b-ablation-scratchpad",
)
TRACE_ON = (
    "qwen35-08b-trace-on",
    "qwen35-2b-trace-on",
    "qwen3-17b-trace-on",
    "qwen3-235b-trace-on",
)
ABLATED = ("qwen25-14b", "qwen25-32b")
ARMS = ("verbatim", "scratchpad", "lean", "compact_rule")
# Metrics on which the ablation harness's verbatim arm is compared with the sealed panel row.
AGREEMENT_METRICS = (
    "format_valid_rate",
    "answerable_decision_recall",
    "false_certainty_rate",
    "underdetermined_abstention_recall",
    "balanced_strict_joint_accuracy",
)
# Conditions whose bootstrap interval the paper prints; the others skip the bootstrap.
WITH_INTERVALS = frozenset({*PANEL, "qwen25-14b-full", "qwen3-235b-trace-on"})
PARTIAL = "phi35-mini-multitoken-stop"
PHI_END_OF_TURN = "<|end|>"
FORMAT_GATE = 0.95

Report = Mapping[str, Any]
Results = Mapping[str, Mapping[str, Any]]


def three(value: float | None) -> str:
    """Format a rate as printed: three decimals, no leading zero, '---' if undefined."""
    if value is None:
        return "---"
    return f"{value:.3f}".removeprefix("0")


def signed(value: float) -> str:
    """Format a signed difference as printed, e.g. '+.008' or '-.010'."""
    return ("-" if value < 0 else "+") + three(abs(value))


def interval(bounds: Sequence[float]) -> str:
    """Format a 95% interval as printed."""
    return f"[{three(bounds[0])}, {three(bounds[1])}]"


def value_range(values: Sequence[float], formatter: Callable[[float], str] = three) -> str:
    """Format the smallest and largest of several values, e.g. '.085-.730'."""
    return f"{formatter(min(values))}-{formatter(max(values))}"


def signal_interval(values: Sequence[float]) -> str:
    """Format the range of several decision signals as a closed interval."""
    return f"[{signed(min(values))}, {signed(max(values))}]"


def fraction(numerator: int, denominator: int) -> str:
    """Format an exact count as printed, e.g. '11/192'."""
    return f"{numerator:,}/{denominator:,}"


def tokens(value: float) -> str:
    """Format a mean token count as printed: rounded half up, with thousands separators."""
    return f"{int(Decimal(repr(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP)):,}"


def percent(value: float, decimals: int = 1) -> str:
    """Format a proportion as a percentage; exactly 100% is printed without decimals."""
    return "100%" if value == 1 else f"{100 * value:.{decimals}f}%"


def two_significant(value: float) -> str:
    """Format a small difference with two significant digits, e.g. '.0052'."""
    return f"{value:.2g}".removeprefix("0")


def bja_interval(report: Report) -> list[float] | None:
    """Return the bootstrap interval of balanced strict joint accuracy, if computed."""
    metrics = report["template_cluster_bootstrap"]["metrics"]
    return metrics["balanced_strict_joint_accuracy"]["ci95"]


def complete_groups(*reports: Report) -> str:
    """Return the largest all-eight strict endpoint of one or more reports as a count."""
    paired = [report["paired_query_group_endpoints"] for report in reports]
    groups = {value["n_paired_query_groups"] for value in paired}
    if len(groups) != 1:
        raise ValueError("reports cover different numbers of route groups")
    best = max(value["paired_strict_joint_pattern_exact_count"] for value in paired)
    return fraction(best, groups.pop())


def blind_payload(row: Mapping[str, Any]) -> str:
    """Return a payload chosen without the observations: the queried axis's first value."""
    declaration, query = row["hypothesis_declaration"], row["query"]
    if row["task"] == "grammar_feature":
        values = {axis["name"]: axis["values"] for axis in declaration["grammar_axes"]}
        return values[query["feature"]][0]
    meanings = {
        form: axis["possible_meanings"]
        for axis in declaration["lexical_axes"]
        for form in axis["forms"]
    }
    return meanings[query["form"]][0]


def evaluate_condition(job: tuple[Path, str, int]) -> tuple[str, dict[str, Any]]:
    """Evaluate one released predictions file and summarize its generation statistics."""
    hf, condition_id, replicates = job
    path = hf / "predictions" / f"{condition_id}.jsonl"
    report = evaluate_predictions(hf, path, bootstrap_replicates=replicates)
    predictions = load_predictions(path)
    counts = [item.completion_tokens for item in predictions if item.completion_tokens is not None]
    return condition_id, {
        "report": report,
        "rows": len(predictions),
        "mean_tokens": mean(counts),
        "length_capped": sum(item.finish_reason == "length" for item in predictions),
        "texts": [item.response_text for item in predictions],
    }


def constant_policies(
    public_rows: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Report]:
    """Evaluate the always-ANSWER, always-ABSTAIN and exact-oracle references on the cohort."""
    cohort = list(select_cohort(public_rows).items)
    policies: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "always-answer": lambda row: {"decision": "ANSWER", "answer": blind_payload(row)},
        "always-abstain": lambda row: {"decision": "ABSTAIN", "answer": None},
        "exact-oracle": lambda row: {
            key: gold_by_id[row["id"]]["gold"][key] for key in ("decision", "answer")
        },
    }
    reports = {}
    for name, respond in policies.items():
        predictions = [
            Prediction(row["id"], json.dumps(respond(row)), "stop", None, False) for row in cohort
        ]
        reports[name] = evaluate_rows(cohort, gold_by_id, predictions, bootstrap_replicates=0)
    return reports


def panel_row(report: Report) -> list[str | None]:
    """Return one row of the sealed panel table (Table 9)."""
    rows, paired = report["rows"], report["paired_query_group_endpoints"]
    bounds = bja_interval(report)
    return [
        three(rows["format_valid_rate"]),
        three(rows["answerable_decision_recall"]),
        three(rows["false_certainty_rate"]),
        signed(rows["decision_signal"]),
        three(rows["underdetermined_abstention_recall"]),
        three(rows["conditional_payload_accuracy"]),
        three(rows["balanced_decision_accuracy"]),
        three(rows["balanced_strict_joint_accuracy"]),
        interval(bounds) if bounds else None,
        three(paired["paired_strict_joint_pattern_exact_rate"]),
    ]


def matched_row(result: Mapping[str, Any]) -> list[str | None]:
    """Return one row of the matched-controls table (Table 3)."""
    rows = result["report"]["rows"]
    return [
        three(rows["format_valid_rate"]),
        signed(rows["decision_signal"]),
        three(rows["conditional_payload_accuracy"]),
        three(rows["balanced_strict_joint_accuracy"]),
        complete_groups(result["report"]),
        tokens(result["mean_tokens"]),
    ]


def ablation_row(result: Mapping[str, Any]) -> list[str | None]:
    """Return one row of the prompt-ablation table (Table 8)."""
    rows = result["report"]["rows"]
    return [
        three(rows["format_valid_rate"]),
        three(rows["answerable_decision_recall"]),
        signed(rows["decision_signal"]),
        three(rows["conditional_payload_accuracy"]),
        three(rows["balanced_strict_joint_accuracy"]),
        tokens(result["mean_tokens"]),
    ]


def stopped_invalid_kind(text: str | None) -> str:
    """Classify a stopped but invalid response: code fence, bare object, or prose."""
    stripped = (text or "").strip()
    if "```" in stripped:
        return "code_fence"
    if stripped.startswith("{") and stripped.endswith("}"):
        return "bare_object"
    return "prose"


def partial_sample(
    hf: Path, public_rows: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """Return the counts of the partial Phi-3.5 multi-token-stop sample."""
    predictions = load_predictions(hf / "predictions" / f"{PARTIAL}.jsonl")
    by_id = {item.id: item for item in predictions}
    scored = score_rows([row for row in public_rows if row["id"] in by_id], gold_by_id, predictions)
    valid = sum(bool(row["format_valid"]) for row in scored)
    stopped = sum(item.finish_reason == "stop" for item in predictions)
    capped = sum(item.finish_reason == "length" for item in predictions)
    kinds = [
        stopped_invalid_kind(by_id[row["item_id"]].response_text)
        for row in scored
        if not row["format_valid"] and by_id[row["item_id"]].finish_reason == "stop"
    ]
    return {
        f"{PARTIAL}.format_valid": fraction(valid, len(predictions)),
        f"{PARTIAL}.length_capped": str(capped),
        f"{PARTIAL}.length_capped_share": percent(capped / len(predictions), 0),
        f"{PARTIAL}.stopped": str(stopped),
        f"{PARTIAL}.stopped_invalid": str(len(kinds)),
        f"{PARTIAL}.stopped_invalid_prose": str(kinds.count("prose")),
        f"{PARTIAL}.stopped_invalid_code_fence": str(kinds.count("code_fence")),
        f"{PARTIAL}.stopped_invalid_bare_object": str(kinds.count("bare_object")),
    }


def panel_numbers(results: Results) -> dict[str, str]:
    """Return the key numbers Section 6.2 states about the sealed panel."""
    rows = {name: results[name]["report"]["rows"] for name in PANEL}
    paired = {name: results[name]["report"]["paired_query_group_endpoints"] for name in PANEL}
    units = {name: results[name]["report"]["unit_counterfactual_consistency"] for name in PANEL}
    payload = [rows[name]["conditional_payload_accuracy"] for name in PANEL]
    usable = [name for name in PANEL if rows[name]["format_valid_rate"] >= FORMAT_GATE]

    def coin(name: str) -> float:
        """Return the same-state pair rate of answering independently at the model's rate."""
        rate = rows[name]["answer_rate"]
        return rate * (1 - rate)

    pairs = paired["qwen25-32b"]
    unit = units["qwen25-14b"]
    return {
        "panel.decision_signal_range": signal_interval(
            [rows[name]["decision_signal"] for name in PANEL]
        ),
        "panel.max_balanced_decision_accuracy": three(
            max(rows[name]["balanced_decision_accuracy"] for name in PANEL)
        ),
        "panel.conditional_payload_accuracy_range": value_range(
            [value for value in payload if value is not None]
        ),
        "panel.format_valid_conditions": str(len(usable)),
        "panel.format_valid_conditional_payload_accuracy_range": value_range(
            [rows[name]["conditional_payload_accuracy"] for name in usable]
        ),
        "panel.max_complete_route_groups": complete_groups(
            *(results[name]["report"] for name in PANEL)
        ),
        "panel.same_state_pairs_above_coin_level": ", ".join(
            name
            for name in PANEL
            if paired[name]["query_complement_edges_oriented_decision_correct_rate"] > coin(name)
        ),
        "qwen25-32b.same_state_pairs": fraction(
            pairs["query_complement_edges_oriented_decision_correct"],
            4 * pairs["n_paired_query_groups"],
        ),
        "qwen25-32b.same_state_pair_rate": three(
            pairs["query_complement_edges_oriented_decision_correct_rate"]
        ),
        "qwen25-32b.same_state_coin_level": three(coin("qwen25-32b")),
        "panel.conditions_with_four_cell_patterns": ", ".join(
            name for name in PANEL if units[name]["unit_decision_pattern_exact_count"]
        ),
        "qwen25-14b.four_cell_patterns": fraction(
            unit["unit_decision_pattern_exact_count"], unit["n_complete_units"]
        ),
        "qwen25-14b.four_cell_guessing_level": three(coin("qwen25-14b") ** 2),
    }


def trace_numbers(results: Results) -> dict[str, str]:
    """Return the key numbers Section 6.2 states about the trace-permitted conditions."""
    report = results["qwen3-235b-trace-on"]["report"]
    rows, paired = report["rows"], report["paired_query_group_endpoints"]
    groups = paired["n_paired_query_groups"]
    unsolved = groups - paired["paired_strict_joint_pattern_exact_count"]
    prefix = "qwen3-235b-trace-on"
    numbers = {
        f"{prefix}.request_success": three(rows["request_success_rate"]),
        f"{prefix}.format_valid": three(rows["format_valid_rate"]),
        f"{prefix}.decision_signal": three(rows["decision_signal"]),
        f"{prefix}.conditional_payload_accuracy": three(rows["conditional_payload_accuracy"]),
        f"{prefix}.balanced_strict_joint_accuracy": three(rows["balanced_strict_joint_accuracy"]),
        f"{prefix}.balanced_strict_joint_accuracy_ci": interval(bja_interval(report)),
        f"{prefix}.same_state_pair_decision": three(
            paired["query_complement_edges_oriented_decision_correct_rate"]
        ),
        f"{prefix}.same_state_independent_level": three(
            rows["answerable_decision_recall"] * rows["underdetermined_abstention_recall"]
        ),
        f"{prefix}.complete_route_groups": complete_groups(report),
        f"{prefix}.unsolved_route_groups": fraction(unsolved, groups),
        f"{prefix}.unsolved_share": percent(unsolved / groups, 0),
        f"{prefix}.mean_tokens": tokens(results[prefix]["mean_tokens"]),
    }
    for name in ("qwen3-235b-trace-off", "qwen3-17b-trace-on"):
        rows = results[name]["report"]["rows"]
        numbers[f"{name}.decision_signal"] = signed(rows["decision_signal"])
        numbers[f"{name}.conditional_payload_accuracy"] = three(
            rows["conditional_payload_accuracy"]
        )
        numbers[f"{name}.complete_route_groups"] = complete_groups(results[name]["report"])
        numbers[f"{name}.mean_tokens"] = tokens(results[name]["mean_tokens"])
    for name in TRACE_ON:
        result = results[name]
        numbers[f"{name}.non_termination"] = percent(result["length_capped"] / result["rows"])
        numbers[f"{name}.format_valid_percent"] = percent(
            result["report"]["rows"]["format_valid_rate"]
        )
    return numbers


def verdict_numbers(results: Results) -> dict[str, str]:
    """Return the key numbers of the full-artifact, replicate and Phi-3.5 conditions."""
    full = results["qwen25-14b-full"]["report"]
    rows = full["rows"]
    texts = results["phi35-mini"]["texts"]
    inside = [text for text in texts if text and PHI_END_OF_TURN in text]
    further = [text for text in inside if text.split(PHI_END_OF_TURN, 1)[1].strip()]
    same = (
        results["qwen25-15b"]["report"]["rows"] == results["qwen25-15b-replicate"]["report"]["rows"]
    )
    return {
        "qwen25-14b-full.request_errors": str(rows["request_error_count"]),
        "qwen25-14b-full.decision_signal": signed(rows["decision_signal"]),
        "qwen25-14b-full.conditional_payload_accuracy": three(rows["conditional_payload_accuracy"]),
        "qwen25-14b-full.balanced_decision_accuracy": three(rows["balanced_decision_accuracy"]),
        "qwen25-14b-full.balanced_strict_joint_accuracy": three(
            rows["balanced_strict_joint_accuracy"]
        ),
        "qwen25-14b-full.balanced_strict_joint_accuracy_ci": interval(bja_interval(full)),
        "qwen25-14b-full.complete_route_groups": complete_groups(full),
        "qwen25-15b-replicate.row_metrics_vs_primary": "identical" if same else "different",
        "phi35-mini.end_of_turn_inside_scored_span": fraction(len(inside), len(texts)),
        "phi35-mini.further_turn_after_end_of_turn": f"{len(further):,} of {len(texts):,}",
    }


def ablation_numbers(results: Results) -> dict[str, str]:
    """Return the key numbers Section 6.2 and Appendix J state about the prompt ablations."""
    numbers: dict[str, str] = {}
    signals: list[float] = []
    for model in ABLATED:
        arms = {arm: results[f"{model}-ablation-{arm}"] for arm in ARMS}
        rows = {arm: arms[arm]["report"]["rows"] for arm in ARMS}
        recall = [rows[arm]["answerable_decision_recall"] for arm in ARMS]
        model_signals = [rows[arm]["decision_signal"] for arm in ARMS]
        signals += model_signals
        sealed = results[model]["report"]["rows"]
        delta = max(abs(rows["verbatim"][key] - sealed[key]) for key in AGREEMENT_METRICS)
        joint = [rows[arm]["balanced_strict_joint_accuracy"] for arm in ("verbatim", "scratchpad")]
        prefix = f"ablation.{model}"
        numbers[f"{prefix}.answerable_decision_recall_range"] = value_range(recall)
        numbers[f"{prefix}.balanced_strict_joint_accuracy_verbatim_to_scratchpad"] = (
            f"{three(joint[0])} to {three(joint[1])}"
        )
        numbers[f"{prefix}.scratchpad_mean_tokens"] = tokens(arms["scratchpad"]["mean_tokens"])
        numbers[f"{prefix}.verbatim_vs_sealed_max_abs_diff"] = two_significant(delta)
        if model == "qwen25-14b":
            factor = max(recall) / min(recall)
            numbers[f"{prefix}.answerable_decision_recall_factor"] = f"{factor:.1f}"
            numbers[f"{prefix}.decision_signal_range"] = signal_interval(model_signals)
        else:
            numbers[f"{prefix}.answerable_decision_recall"] = ", ".join(map(three, recall))
            numbers[f"{prefix}.decision_signal"] = ", ".join(map(signed, model_signals))
    numbers["ablation.decision_signal_range"] = signal_interval(signals)
    numbers["ablation.max_complete_route_groups"] = complete_groups(
        *(results[f"{model}-ablation-{arm}"]["report"] for model in ABLATED for arm in ARMS)
    )
    return numbers


def compare(expected: Any, observed: Any, path: str, mismatches: list[str]) -> int:
    """Compare printed values recursively; return how many values were checked."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            mismatches.append(f"{path}: keys differ")
            return 0
        return sum(
            compare(expected[key], observed[key], f"{path}.{key}", mismatches) for key in expected
        )
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            mismatches.append(f"{path}: lengths differ")
            return 0
        return sum(
            compare(left, right, f"{path}[{index}]", mismatches)
            for index, (left, right) in enumerate(zip(expected, observed, strict=True))
        )
    if expected != observed:
        mismatches.append(f"{path}: paper {expected!r}, reproduced {observed!r}")
    return 1


def print_table(table: Mapping[str, Any]) -> None:
    """Print one table with right-aligned columns."""
    header = ["", *table["columns"]]
    body = [
        [entry["label"], *[value or "" for value in entry["values"]]]
        for entry in table["rows"].values()
    ]
    widths = [max(len(row[index]) for row in [header, *body]) for index in range(len(header))]
    print(f"\n{table['title']}")
    for row in [header, *body]:
        cells = (cell.rjust(width) for cell, width in zip(row, widths, strict=True))
        print("  ".join(cells).rstrip())


def reproduce(hf: Path, paper: Mapping[str, Any], jobs: int) -> tuple[dict[str, Any], list[str]]:
    """Evaluate every released condition and return the reproduced values and file errors."""
    conditions = json.loads((hf / "predictions" / "conditions.json").read_text())["conditions"]
    errors = [
        f"{item['predictions_file']}: SHA-256 differs from conditions.json"
        for item in conditions
        if file_sha256(hf / item["predictions_file"]) != item["predictions_sha256"]
    ]
    work = [
        (hf, item["condition_id"], 10_000 if item["condition_id"] in WITH_INTERVALS else 0)
        for item in conditions
        if item["condition_id"] != PARTIAL
    ]
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        results = dict(pool.map(evaluate_condition, work))
    public_rows, gold_by_id = load_public_rows(hf), load_gold_by_id(hf)
    references = constant_policies(public_rows, gold_by_id)
    builders: dict[str, tuple[Sequence[str], Callable[[str], list[str | None]]]] = {
        "model_panel": (
            [*PANEL, *references],
            lambda key: panel_row(references.get(key) or results[key]["report"]),
        ),
        "matched_controls": (MATCHED, lambda key: matched_row(results[key])),
        "prompt_ablations": (
            [f"qwen25-14b-ablation-{arm}" for arm in ARMS],
            lambda key: ablation_row(results[key]),
        ),
    }
    tables = {}
    for name, (keys, build) in builders.items():
        printed = paper["tables"][name]
        tables[name] = {
            "title": printed["title"],
            "columns": printed["columns"],
            "rows": {
                key: {"label": printed["rows"][key]["label"], "values": build(key)} for key in keys
            },
        }
    numbers = {
        **panel_numbers(results),
        **trace_numbers(results),
        **verdict_numbers(results),
        **ablation_numbers(results),
        **partial_sample(hf, public_rows, gold_by_id),
    }
    return {"tables": tables, "numbers": numbers}, errors


def main() -> int:
    """Evaluate every condition, print the tables, and compare with the paper."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", type=Path, required=True, help="downloaded dataset directory")
    parser.add_argument("--values", type=Path, default=VALUES, help="paper values to compare with")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, help="also write the reproduced values here")
    args = parser.parse_args()
    paper = json.loads(args.values.read_text(encoding="utf-8"))
    reproduced, mismatches = reproduce(args.hf, paper, args.jobs)
    for table in reproduced["tables"].values():
        print_table(table)
    print("\nKey numbers")
    width = max(map(len, reproduced["numbers"]))
    for key, value in sorted(reproduced["numbers"].items()):
        print(f"  {key.ljust(width)}  {value}")
    expected = {
        "tables": paper["tables"],
        "numbers": {key: item["value"] for key, item in paper["numbers"].items()},
    }
    checked = compare(expected, reproduced, "paper", mismatches)
    if args.output is not None:
        args.output.write_text(json.dumps(reproduced, indent=2) + "\n", encoding="utf-8")
    print(f"\n{checked} printed values checked; {len(mismatches)} differ from the paper")
    for line in mismatches:
        print(f"  MISMATCH {line}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
