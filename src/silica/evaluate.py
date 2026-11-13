"""Scoring predictions against the gold rows.

Row metrics follow the paper.  With G the gold action and Ĝ the predicted one:
answerable decision recall ``AR = P(Ĝ=ANSWER | G=ANSWER)``, false certainty
``FCR = P(Ĝ=ANSWER | G=ABSTAIN)``, underdetermined abstention recall
``UR = P(Ĝ=ABSTAIN | G=ABSTAIN)``, exact answerable recall ``EAR`` (ANSWER
with the canonical payload), balanced decision accuracy ``BDA = (AR+UR)/2``,
balanced joint accuracy ``BJA = (EAR+UR)/2``, decision signal ``AR-FCR`` and
conditional payload accuracy ``CPA = EAR/AR``.  An invalid or missing
response is wrong, never skipped.

Unit endpoints score each four-cell unit (two ANSWER and two ABSTAIN rows
asking about one feature) and each eight-row paired-query group.  Intervals
come from a percentile bootstrap that resamples whole evaluation templates
within each mechanism family, moving every row, unit and group of a template
together.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cohort import public_query_axis, select_cohort, validate_paired_cohort
from .dataset import load_gold_by_id, load_public_rows
from .parsing import score_response
from .predictions import Prediction, load_predictions
from .util import canonical_json, file_sha256

EVALUATION_SCHEMA = "silica-evaluation-v1"
BOOTSTRAP_METHOD = "family_stratified_template_cluster_percentile_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_METRICS = (
    "decision_accuracy",
    "balanced_decision_accuracy",
    "strict_joint_accuracy",
    "balanced_strict_joint_accuracy",
    "format_valid_rate",
    "abstention_rate",
    "unit_decision_pattern_exact_rate",
    "unit_strict_joint_pattern_exact_rate",
    "unit_factorial_consistency_rate",
)
PAIRED_BOOTSTRAP_METRICS = (
    "paired_decision_pattern_exact_rate",
    "paired_strict_joint_pattern_exact_rate",
    "all_eight_format_valid_rate",
    "both_four_cell_decision_orientations_exact_rate",
    "both_four_cell_strict_joint_orientations_exact_rate",
    "oriented_query_complement_decision_exact_rate",
    "oriented_query_complement_strict_joint_exact_rate",
)
_PAIRED_FLAGS = (
    "all_eight_format_valid",
    "predicted_four_answer_four_abstain",
    "paired_decision_pattern_exact",
    "paired_strict_joint_pattern_exact",
    "both_four_cell_decision_orientations_exact",
    "both_four_cell_strict_joint_orientations_exact",
    "oriented_query_complement_decision_exact",
    "oriented_query_complement_strict_joint_exact",
    "unoriented_query_complement_structural_consistency",
    "unoriented_paired_counterfactual_structural_consistency",
)
_PAIRED_EDGES = (
    "query_complement_edges_oriented_decision_correct",
    "query_complement_edges_strict_joint_correct",
    "query_complement_edges_unoriented_structurally_consistent",
)


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a ratio, or None when the denominator is zero."""
    return numerator / denominator if denominator else None


def _balanced(first: float | None, second: float | None) -> float | None:
    """Return the mean of two rates, or None if either is undefined."""
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def score_rows(
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    predictions: Sequence[Prediction],
) -> list[dict[str, Any]]:
    """Join predictions to gold rows and score each one."""
    by_id = {item.id: item for item in predictions}
    rows = []
    for item in sorted(public_rows, key=lambda row: str(row["id"])):
        identifier = str(item["id"])
        prediction = by_id[identifier]
        gold = gold_by_id[identifier]["gold"]
        parsed = score_response(
            prediction.response_text,
            finish_reason=prediction.finish_reason,
            request_error=prediction.request_error,
            task=str(item["task"]),
            query=item["query"],
        )
        decision = parsed.decision if parsed.valid else None
        if gold["decision"] == "ABSTAIN":
            joint_correct = parsed.valid and decision == "ABSTAIN"
        else:
            joint_correct = (
                parsed.valid
                and decision == "ANSWER"
                and canonical_json(parsed.answer) == canonical_json(gold.get("answer"))
            )
        rows.append(
            {
                "item_id": identifier,
                "complete_unit_id": item["complete_unit_id"],
                "paired_query_unit_id": item["paired_query_unit_id"],
                "paired_state_id": item["paired_state_id"],
                "family": item["mechanism"]["family"],
                "query_feature": public_query_axis(item),
                "template_id": item["mechanism"]["template_id"],
                "gold_decision": gold["decision"],
                "target_payload_key": canonical_json(gold["target_answer"]),
                "pred_decision": decision,
                "format_valid": parsed.valid,
                "format_error": not prediction.request_error and not parsed.valid,
                "request_error": prediction.request_error,
                "decision_correct": parsed.valid and decision == gold["decision"],
                "joint_correct": joint_correct,
            }
        )
    return rows


def row_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the selective row metrics of a row set."""
    n = len(rows)
    answerable = [row for row in rows if row["gold_decision"] == "ANSWER"]
    underdetermined = [row for row in rows if row["gold_decision"] == "ABSTAIN"]
    answer_decision = sum(bool(row["decision_correct"]) for row in answerable)
    abstain_decision = sum(bool(row["decision_correct"]) for row in underdetermined)
    answer_joint = sum(bool(row["joint_correct"]) for row in answerable)
    abstain_joint = sum(bool(row["joint_correct"]) for row in underdetermined)
    format_valid = sum(bool(row["format_valid"]) for row in rows)
    request_errors = sum(bool(row["request_error"]) for row in rows)
    format_errors = sum(bool(row["format_error"]) for row in rows)
    predicted_abstain = sum(row["pred_decision"] == "ABSTAIN" for row in rows)
    predicted_answer = sum(row["pred_decision"] == "ANSWER" for row in rows)
    false_certainty = sum(row["pred_decision"] == "ANSWER" for row in underdetermined)
    false_abstention = sum(row["pred_decision"] == "ABSTAIN" for row in answerable)
    answer_recall = _rate(answer_decision, len(answerable))
    abstain_recall = _rate(abstain_decision, len(underdetermined))
    exact_recall = _rate(answer_joint, len(answerable))
    false_certainty_rate = _rate(false_certainty, len(underdetermined))
    return {
        "n": n,
        "n_answerable": len(answerable),
        "n_underdetermined": len(underdetermined),
        "decision_correct": answer_decision + abstain_decision,
        "decision_accuracy": _rate(answer_decision + abstain_decision, n),
        "answerable_decision_recall": answer_recall,
        "underdetermined_abstention_recall": abstain_recall,
        "balanced_decision_accuracy": _balanced(answer_recall, abstain_recall),
        "strict_joint_correct": answer_joint + abstain_joint,
        "strict_joint_accuracy": _rate(answer_joint + abstain_joint, n),
        "answerable_exact_payload_correct": answer_joint,
        "answerable_exact_payload_accuracy": exact_recall,
        "balanced_strict_joint_accuracy": _balanced(
            exact_recall, _rate(abstain_joint, len(underdetermined))
        ),
        "decision_signal": (
            None
            if answer_recall is None or false_certainty_rate is None
            else answer_recall - false_certainty_rate
        ),
        "conditional_payload_accuracy": exact_recall / answer_recall if answer_recall else None,
        "format_valid_count": format_valid,
        "format_valid_rate": _rate(format_valid, n),
        "format_error_count": format_errors,
        "format_error_rate": _rate(format_errors, n),
        "request_error_count": request_errors,
        "request_error_rate": _rate(request_errors, n),
        "request_success_rate": _rate(n - request_errors, n),
        "invalid_or_missing_response_rate": _rate(n - format_valid, n),
        "predicted_answer_count": predicted_answer,
        "answer_rate": _rate(predicted_answer, n),
        "answer_rate_given_format_valid": _rate(predicted_answer, format_valid),
        "predicted_abstain_count": predicted_abstain,
        "abstention_rate": _rate(predicted_abstain, n),
        "abstention_rate_given_format_valid": _rate(predicted_abstain, format_valid),
        "false_certainty_count": false_certainty,
        "false_certainty_rate": false_certainty_rate,
        "false_abstention_count": false_abstention,
        "false_abstention_rate": _rate(false_abstention, len(answerable)),
    }


def _single(group: Sequence[Mapping[str, Any]], field: str, label: str) -> str:
    """Return the one value a field takes across a group, or raise."""
    values = {str(row[field]) for row in group}
    if len(values) != 1:
        raise ValueError(f"{label} changes {field}")
    return next(iter(values))


def unit_outcomes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score every four-cell unit as a decision-by-payload 2x2 factorial."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["complete_unit_id"])].append(row)
    outcomes = []
    for unit_id, group in sorted(grouped.items()):
        if len(group) != 4 or len({str(row["item_id"]) for row in group}) != 4:
            raise ValueError(f"predictions split unit {unit_id}")
        targets = sorted({str(row["target_payload_key"]) for row in group})
        cells = {(str(row["gold_decision"]), str(row["target_payload_key"])): row for row in group}
        expected = {(decision, target) for decision in ("ANSWER", "ABSTAIN") for target in targets}
        if len(targets) != 2 or len(cells) != 4 or set(cells) != expected:
            raise ValueError(f"unit {unit_id} is not a complete decision-by-payload 2x2")
        flips = 0
        for target in targets:
            left, right = (
                cells[("ANSWER", target)]["pred_decision"],
                cells[("ABSTAIN", target)]["pred_decision"],
            )
            flips += left is not None and right is not None and left != right
        invariances = 0
        for decision in ("ANSWER", "ABSTAIN"):
            left, right = (
                cells[(decision, targets[0])]["pred_decision"],
                cells[(decision, targets[1])]["pred_decision"],
            )
            invariances += left is not None and right is not None and left == right
        all_valid = all(bool(row["format_valid"]) for row in group)
        outcomes.append(
            {
                "complete_unit_id": unit_id,
                "family": _single(group, "family", f"unit {unit_id}"),
                "query_feature": _single(group, "query_feature", f"unit {unit_id}"),
                "template_id": _single(group, "template_id", f"unit {unit_id}"),
                "all_four_format_valid": all_valid,
                "predicted_two_answer_two_abstain": all_valid
                and Counter(str(row["pred_decision"]) for row in group)
                == Counter({"ANSWER": 2, "ABSTAIN": 2}),
                "decision_pattern_exact": all(bool(row["decision_correct"]) for row in group),
                "strict_joint_pattern_exact": all(bool(row["joint_correct"]) for row in group),
                "answer_payload_pair_exact": all(
                    bool(row["joint_correct"]) for row in group if row["gold_decision"] == "ANSWER"
                ),
                "factorial_consistent_unoriented": all_valid and flips == 2 and invariances == 2,
                "unoriented_answerability_flip_edges_consistent": flips,
                "unoriented_payload_invariance_edges_consistent": invariances,
            }
        )
    return outcomes


def unit_metrics(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate four-cell unit outcomes."""
    n = len(units)

    def count(field: str) -> int:
        """Count the units for which a flag holds."""
        return sum(bool(unit[field]) for unit in units)

    flips = sum(int(unit["unoriented_answerability_flip_edges_consistent"]) for unit in units)
    invariances = sum(int(unit["unoriented_payload_invariance_edges_consistent"]) for unit in units)
    result: dict[str, Any] = {"n_complete_units": n}
    for name, field in (
        ("all_four_format_valid", "all_four_format_valid"),
        ("predicted_two_answer_two_abstain", "predicted_two_answer_two_abstain"),
        ("unit_decision_pattern_exact", "decision_pattern_exact"),
        ("unit_strict_joint_pattern_exact", "strict_joint_pattern_exact"),
        ("unit_answer_payload_pair_exact", "answer_payload_pair_exact"),
        ("unit_factorial_consistency", "factorial_consistent_unoriented"),
    ):
        result[f"{name}_count"] = count(field)
        result[f"{name}_rate"] = _rate(count(field), n)
    result["unoriented_answerability_flip_edges_consistent"] = flips
    result["unoriented_answerability_flip_edge_structural_consistency_rate"] = _rate(flips, 2 * n)
    result["unoriented_payload_invariance_edges_consistent"] = invariances
    result["unoriented_payload_invariance_edge_structural_consistency_rate"] = _rate(
        invariances, 2 * n
    )
    return result


def paired_outcomes(
    rows: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Score every eight-row paired-query group and its four query complements."""
    unit_by_id = {str(unit["complete_unit_id"]): unit for unit in units}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["paired_query_unit_id"])].append(row)
    outcomes = []
    for paired_id, group in sorted(grouped.items()):
        label = f"paired group {paired_id}"
        query_features = sorted({str(row["query_feature"]) for row in group})
        complete_ids = sorted({str(row["complete_unit_id"]) for row in group})
        if (
            len(group) != 8
            or len({str(row["item_id"]) for row in group}) != 8
            or len(query_features) != 2
            or len(complete_ids) != 2
            or any(value not in unit_by_id for value in complete_ids)
        ):
            raise ValueError(f"{label} is split or changes its route")
        states: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            states[str(row["paired_state_id"])].append(row)
        if len(states) != 4 or {len(value) for value in states.values()} != {2}:
            raise ValueError(f"{label} must contain four query pairs")
        structural = oriented = strict = 0
        for state_id, state_rows in states.items():
            if (
                {str(row["query_feature"]) for row in state_rows} != set(query_features)
                or {str(row["complete_unit_id"]) for row in state_rows} != set(complete_ids)
                or Counter(str(row["gold_decision"]) for row in state_rows)
                != Counter({"ANSWER": 1, "ABSTAIN": 1})
            ):
                raise ValueError(f"{label}/{state_id} is not a query complement")
            decisions = [row["pred_decision"] for row in state_rows]
            structural += (
                all(bool(row["format_valid"]) for row in state_rows)
                and None not in decisions
                and decisions[0] != decisions[1]
            )
            oriented += all(bool(row["decision_correct"]) for row in state_rows)
            strict += all(bool(row["joint_correct"]) for row in state_rows)
        linked = [unit_by_id[value] for value in complete_ids]
        all_valid = all(bool(row["format_valid"]) for row in group)
        outcomes.append(
            {
                "paired_query_unit_id": paired_id,
                "family": _single(group, "family", label),
                "template_id": _single(group, "template_id", label),
                "query_route": "__".join(query_features),
                "all_eight_format_valid": all_valid,
                "predicted_four_answer_four_abstain": all_valid
                and Counter(str(row["pred_decision"]) for row in group)
                == Counter({"ANSWER": 4, "ABSTAIN": 4}),
                "paired_decision_pattern_exact": all(
                    bool(row["decision_correct"]) for row in group
                ),
                "paired_strict_joint_pattern_exact": all(
                    bool(row["joint_correct"]) for row in group
                ),
                "both_four_cell_decision_orientations_exact": all(
                    bool(unit["decision_pattern_exact"]) for unit in linked
                ),
                "both_four_cell_strict_joint_orientations_exact": all(
                    bool(unit["strict_joint_pattern_exact"]) for unit in linked
                ),
                "oriented_query_complement_decision_exact": oriented == 4,
                "oriented_query_complement_strict_joint_exact": strict == 4,
                "unoriented_query_complement_structural_consistency": structural == 4,
                "unoriented_paired_counterfactual_structural_consistency": structural == 4
                and all(bool(unit["factorial_consistent_unoriented"]) for unit in linked),
                "query_complement_edges_oriented_decision_correct": oriented,
                "query_complement_edges_strict_joint_correct": strict,
                "query_complement_edges_unoriented_structurally_consistent": structural,
            }
        )
    return outcomes


def paired_metrics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate paired-query group outcomes."""
    n = len(pairs)
    result: dict[str, Any] = {
        "available": bool(n),
        "n_paired_query_groups": n,
        "accuracy_endpoints_are_oriented_to_gold": True,
        "structural_consistency_is_not_accuracy": True,
    }
    for field in _PAIRED_FLAGS:
        value = sum(bool(pair[field]) for pair in pairs)
        result[f"{field}_count"] = value
        result[f"{field}_rate"] = _rate(value, n)
    for field in _PAIRED_EDGES:
        value = sum(int(pair[field]) for pair in pairs)
        result[field] = value
        result[f"{field}_rate"] = _rate(value, 4 * n)
    return result


def _groups(values: Sequence[Mapping[str, Any]], field: str) -> dict[str, list[Mapping[str, Any]]]:
    """Group records by the string value of one field, in sorted order."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for value in values:
        grouped[str(value[field])].append(value)
    return dict(sorted(grouped.items()))


def breakdown(
    rows: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Any]:
    """Return row and unit metrics per value of one field."""
    unit_groups = _groups(units, field)
    return {
        value: {"rows": row_metrics(group), "units": unit_metrics(unit_groups.get(value, []))}
        for value, group in _groups(rows, field).items()
    }


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated quantile."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of zero bootstrap values")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_values(
    rows: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Return the bootstrapped metrics of one (re)sample."""
    combined = {
        **row_metrics(rows),
        **unit_metrics(units),
        **(paired_metrics(pairs) if pairs else {}),
    }
    result = {}
    for metric in (*BOOTSTRAP_METRICS, *(PAIRED_BOOTSTRAP_METRICS if pairs else ())):
        value = combined.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"bootstrap metric {metric} is undefined")
        result[metric] = float(value)
    return result


def template_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return percentile intervals from resampling whole templates within each family."""
    if replicates < 0:
        raise ValueError("bootstrap replicates must be nonnegative")
    clusters: dict[str, dict[str, dict[str, list[Mapping[str, Any]]]]] = {}
    for kind, values in (("rows", rows), ("units", units), ("pairs", pairs)):
        clusters[kind] = defaultdict(lambda: defaultdict(list))
        for value in values:
            clusters[kind][str(value["family"])][str(value["template_id"])].append(value)
    layout = {family: set(templates) for family, templates in clusters["rows"].items()}
    for kind in ("units", "pairs") if pairs else ("units",):
        if {family: set(templates) for family, templates in clusters[kind].items()} != layout:
            raise ValueError(f"row and {kind} template clusters disagree")
    point = _bootstrap_values(rows, units, pairs)
    cluster_counts = {
        family: len(templates) for family, templates in sorted(clusters["rows"].items())
    }

    def size_histogram(kind: str) -> dict[str, int]:
        """Histogram the number of records per template cluster."""
        sizes = Counter(
            len(cluster) for templates in clusters[kind].values() for cluster in templates.values()
        )
        return {str(key): value for key, value in sorted(sizes.items())}

    metric_names = (*BOOTSTRAP_METRICS, *(PAIRED_BOOTSTRAP_METRICS if pairs else ()))
    base = {
        "method": BOOTSTRAP_METHOD,
        "confidence_level": 0.95,
        "sampling_unit": "mechanism.template_id",
        "stratified_by": "mechanism.family",
        "row_binomial_intervals": False,
        "estimand_label": "heldout_template_cluster_sensitivity",
        "interpretation": (
            "Sensitivity to resampling whole held-out Boolean templates; not an "
            "independent-row or deployment-population confidence interval."
        ),
        "selected_template_cluster_count": sum(cluster_counts.values()),
        "template_clusters_by_family": cluster_counts,
        "clusters_drawn_per_replicate_by_family": cluster_counts,
        "all_rows_and_units_move_with_template_cluster": True,
        "row_count_per_selected_template_histogram": size_histogram("rows"),
        "unit_count_per_selected_template_histogram": size_histogram("units"),
        "paired_group_count_per_selected_template_histogram": size_histogram("pairs"),
        "seed": seed,
        "replicates": replicates,
    }
    if replicates == 0:
        return {
            **base,
            "available": False,
            "reason": "disabled_by_zero_replicates",
            "metrics": {metric: {"point": point[metric], "ci95": None} for metric in metric_names},
        }
    if any(count < 2 for count in cluster_counts.values()):
        raise ValueError("the template-cluster bootstrap needs at least two templates per family")
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
    template_ids = {family: sorted(templates) for family, templates in clusters["rows"].items()}
    for _ in range(replicates):
        sampled: dict[str, list[Mapping[str, Any]]] = {"rows": [], "units": [], "pairs": []}
        for family in sorted(template_ids):
            candidates = template_ids[family]
            for _ in range(len(candidates)):
                template_id = candidates[rng.randrange(len(candidates))]
                for kind in sampled:
                    sampled[kind].extend(clusters[kind][family][template_id])
        for metric, value in _bootstrap_values(
            sampled["rows"], sampled["units"], sampled["pairs"]
        ).items():
            samples[metric].append(value)
    return {
        **base,
        "available": True,
        "reason": None,
        "metrics": {
            metric: {
                "point": point[metric],
                "ci95": [quantile(samples[metric], 0.025), quantile(samples[metric], 0.975)],
            }
            for metric in metric_names
        },
    }


def evaluate_rows(
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    predictions: Sequence[Prediction],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute every metric of predictions that cover exactly the given rows."""
    validate_paired_cohort(public_rows)
    predicted = {item.id for item in predictions}
    expected = {str(row["id"]) for row in public_rows}
    if predicted != expected:
        raise ValueError(
            f"predictions must cover exactly the evaluated rows: "
            f"missing={len(expected - predicted)}, extra={len(predicted - expected)}"
        )
    rows = score_rows(public_rows, gold_by_id, predictions)
    units = unit_outcomes(rows)
    pairs = paired_outcomes(rows, units)
    return {
        "rows": row_metrics(rows),
        "unit_counterfactual_consistency": unit_metrics(units),
        "paired_query_group_endpoints": paired_metrics(pairs),
        "paired_query_group_breakdowns": {
            "by_mechanism_family": {
                key: paired_metrics(value) for key, value in _groups(pairs, "family").items()
            },
            "by_query_route": {
                key: paired_metrics(value) for key, value in _groups(pairs, "query_route").items()
            },
        },
        "by_mechanism_family": breakdown(rows, units, "family"),
        "by_query_axis": breakdown(rows, units, "query_feature"),
        "template_cluster_bootstrap": template_cluster_bootstrap(
            rows, units, pairs, replicates=bootstrap_replicates, seed=bootstrap_seed
        ),
    }


def evaluate_predictions(
    dataset: Path,
    predictions_path: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate a predictions file covering the full dataset or its common cohort."""
    public_rows = load_public_rows(dataset)
    gold_by_id = load_gold_by_id(dataset)
    predictions = load_predictions(predictions_path)
    predicted = {item.id for item in predictions}
    cohort: dict[str, Any]
    if predicted == {str(row["id"]) for row in public_rows}:
        evaluated = public_rows
        cohort = {"name": "full", "rows": len(evaluated)}
    else:
        selection = select_cohort(public_rows)
        evaluated = list(selection.items)
        cohort = {"name": "common", "rows": len(evaluated), "cohort_hash": selection.cohort_hash}
    report = evaluate_rows(
        evaluated,
        gold_by_id,
        predictions,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    files = {
        path.relative_to(dataset).as_posix(): file_sha256(path)
        for view in ("public", "gold")
        for path in sorted((dataset / view).glob("*.jsonl"))
    }
    return {
        "schema_version": EVALUATION_SCHEMA,
        "inputs": {
            "dataset_files": files,
            "predictions_sha256": file_sha256(predictions_path),
            "predictions": len(predictions),
        },
        "cohort": cohort,
        **report,
    }
