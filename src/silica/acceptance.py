"""The acceptance audit recorded in the release manifest.

The audit recomputes, from the generated rows, every structural property the
benchmark claims: balanced decisions in every unit, identical posterior sizes
for answerable and unanswerable cells, complementary query pairs over
byte-identical raw state, balanced route cycling, an exact oracle, and eight
deterministic shortcut probes that must all score exactly one half.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._frozen import ACCEPTANCE_AUDIT_SCHEMA, ROUTE_COVERAGE_SCHEMA, ROUTE_CYCLING_ALGORITHM
from .certificates import certificate_answer
from .cohort import validate_paired_cohort
from .generator import query_masked_state
from .hypotheses import FEATURE_NAMES
from .routes import QUERY_AXIS_PAIRS, evaluation_semantic_counts, route_registry
from .templates import (
    MECHANISM_FAMILIES,
    STRATA,
    balanced_strata,
    observed_state_index,
    template_registry,
)
from .util import canonical_json, object_hash

SHORTCUT_PROBES = (
    "posterior_cardinality",
    "declared_class_size",
    "structural_metrics",
    "identifier_routing",
    "row_order",
    "query_masked_raw_program_evidence",
    "evidence_identities_values_masked",
    "joint_masked",
)


def _histogram(values: Iterable[int]) -> dict[str, int]:
    """Return a string-keyed histogram sorted by value."""
    return {str(key): count for key, count in sorted(Counter(values).items())}


def _masked_evidence(public: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the observations with their observed forms masked."""
    masked = []
    for raw in public["evidence"]:
        if set(raw) != {"id", "channel", "category", "semantic", "form"}:
            raise ValueError("dictionary evidence schema drifted")
        masked.append(
            {
                "channel": raw["channel"],
                "category": raw["category"],
                "semantic": raw["semantic"],
                "observed_form": "__MASKED__",
            }
        )
    return masked


def _identifier_shape(value: Any) -> Any:
    """Return the prefix and length of an identifier, hiding its digits."""
    if isinstance(value, str):
        return {"prefix": value.split("-", 1)[0], "length": len(value)}
    return {"type": type(value).__name__}


def _probe_signature(public: Mapping[str, Any], probe: str, unit_position: int) -> Any:
    """Return the query-blind view of a row that one shortcut probe may read."""
    mechanism = public["mechanism"]
    declaration = public["hypothesis_declaration"]
    if probe == "posterior_cardinality":
        return {"full_evidence_posterior_size": 16}
    if probe == "declared_class_size":
        return {"class_size": declaration["class_size"]}
    if probe == "structural_metrics":
        return public["structural_complexity"]
    if probe == "identifier_routing":
        return {
            "complete_unit_id": public["complete_unit_id"],
            "query_id": public["query_id"],
            "language_id": public["language_id"],
            "row_id_shape": _identifier_shape(public["id"]),
            "cell_id_shape": _identifier_shape(public["counterfactual_cell_id"]),
        }
    if probe == "row_order":
        return {"position_within_complete_unit": unit_position}
    if probe == "query_masked_raw_program_evidence":
        return query_masked_state(public)
    if probe == "evidence_identities_values_masked":
        return _masked_evidence(public)
    if probe == "joint_masked":
        return {
            "complete_unit_id": public["complete_unit_id"],
            "family": mechanism["family"],
            "template_id": mechanism["template_id"],
            "query": public["query"],
            "declaration_program_hash": declaration["boolean_program_hash"],
            "structural_complexity": public["structural_complexity"],
            "evidence": _masked_evidence(public),
        }
    raise ValueError(f"unknown shortcut probe {probe!r}")


def _majority_counts(grouped: Mapping[str, Counter[str]]) -> tuple[int, int]:
    """Return the correct and total counts of the best constant guess per signature."""
    correct = sum(max(counts.values()) for counts in grouped.values())
    total = sum(sum(counts.values()) for counts in grouped.values())
    return correct, total


def signature_probe(
    public_records: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    probe: str,
) -> dict[str, Any]:
    """Score the best predictor that maps each probe signature to one decision."""
    positions: dict[str, int] = defaultdict(int)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    family_groups: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for public in public_records:
        unit = str(public["complete_unit_id"])
        signature = object_hash(_probe_signature(public, probe, positions[unit]))
        positions[unit] += 1
        decision = str(gold_by_id[str(public["id"])]["gold"]["decision"])
        groups[signature][decision] += 1
        family_groups[str(public["mechanism"]["family"])][signature][decision] += 1
    correct, total = _majority_counts(groups)
    per_family = {}
    for family, signatures in sorted(family_groups.items()):
        family_correct, family_total = _majority_counts(signatures)
        per_family[family] = {
            "correct": family_correct,
            "total": family_total,
            "balanced_decision_accuracy": family_correct / family_total,
            "exactly_one_half": 2 * family_correct == family_total,
        }
    return {
        "signature_groups": len(groups),
        "correct": correct,
        "total": total,
        "balanced_decision_accuracy": correct / total,
        "exactly_one_half": 2 * correct == total,
        "all_signatures_exactly_balanced": all(
            counts["ANSWER"] == counts["ABSTAIN"] and counts["ANSWER"] > 0
            for counts in groups.values()
        ),
        "per_family": per_family,
    }


def shortcut_probes(
    public_records: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Score all eight deterministic query-blind shortcut probes."""
    return {name: signature_probe(public_records, gold_by_id, name) for name in SHORTCUT_PROBES}


def _posterior_matching(
    public_records: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare answerable and unanswerable posterior-size histograms in every stratum."""
    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for public in public_records:
        private = gold_by_id[str(public["id"])]
        decision = str(private["gold"]["decision"])
        posterior = int(private["certificate"]["hypothesis_count"])
        family = public["mechanism"]["family"]
        template = public["mechanism"]["template_id"]
        query = str(private["oracle"]["query_feature"])
        for key in (
            "global",
            f"split={public['split']}",
            f"family={family}",
            f"query={query}",
            f"template={template}",
            f"family_query={family}|{query}",
            f"template_query={template}|{query}",
        ):
            strata[key][decision].append(posterior)
    serialized = {}
    for key, by_decision in sorted(strata.items()):
        answer = _histogram(by_decision["ANSWER"])
        abstain = _histogram(by_decision["ABSTAIN"])
        serialized[key] = {
            "ANSWER": answer,
            "ABSTAIN": abstain,
            "histograms_identical": answer == abstain,
            "only_posterior_size_16": set(answer) == {"16"} and set(abstain) == {"16"},
        }
    return {
        "strata_checked": len(serialized),
        "all_answer_abstain_histograms_identical": all(
            item["histograms_identical"] for item in serialized.values()
        ),
        "all_cells_posterior_size_16": all(
            item["only_posterior_size_16"] for item in serialized.values()
        ),
        "strata": serialized,
    }


def _route_state_coverage(
    public_records: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Verify balanced route cycling independently of the generated row order."""
    template_by_id = {item.template_id: item for item in template_registry().evaluation}
    pair_index_by_features = {
        frozenset((lexical, grammar)): index
        for index, (_, lexical, grammar) in enumerate(QUERY_AXIS_PAIRS)
    }
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in public_records:
        paired[str(row["paired_query_unit_id"])].append(row)
    template_routes: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    route_details = []
    for paired_id, rows in sorted(paired.items()):
        template_ids = {str(row["mechanism"]["template_id"]) for row in rows}
        query_features = {
            str(gold_by_id[str(row["id"])]["oracle"]["query_feature"]) for row in rows
        }
        if len(template_ids) != 1 or frozenset(query_features) not in pair_index_by_features:
            raise ValueError(f"{paired_id}: cannot identify one template/query route")
        template_id = next(iter(template_ids))
        pair_index = pair_index_by_features[frozenset(query_features)]
        states = {}
        for row in rows:
            oracle = gold_by_id[str(row["id"])]["oracle"]
            ordered = tuple(str(value) for value in oracle["ordered_evidence_features"])
            bits = tuple(int(oracle["observed_feature_bits"][feature]) for feature in ordered)
            if bool(oracle["guard_active"]):
                stratum = f"active_{int(oracle['role_zero_payload_value'])}"
            else:
                stratum = f"inactive_{int(oracle['role_one_payload_value'])}"
            value = (stratum, observed_state_index(bits))
            if states.setdefault(str(row["paired_state_id"]), value) != value:
                raise ValueError(f"{paired_id}: paired rows disagree on their state")
        if len(states) != 4:
            raise ValueError(f"{paired_id}: route must expose exactly four states")
        selected = dict(states.values())
        if set(selected) != set(STRATA):
            raise ValueError(f"{paired_id}: route does not cover the four strata")
        if len(set(selected.values())) != 4:
            raise ValueError(f"{paired_id}: route reuses an observed X state")
        if pair_index in template_routes[template_id]:
            raise ValueError(f"{template_id}: duplicate route index {pair_index}")
        template_routes[template_id][pair_index] = selected
        route_details.append((template_id, pair_index, tuple(sorted(selected.items()))))

    diagnostics = []
    stratum_sizes: Counter[int] = Counter()
    spreads: Counter[int] = Counter()
    union_sizes: Counter[int] = Counter()
    for template_id, routes in sorted(template_routes.items()):
        if template_id not in template_by_id or set(routes) != set(range(8)):
            raise ValueError(f"{template_id}: expected all eight route indices")
        template = template_by_id[template_id]
        strata = balanced_strata(template.guard, template.query_payload, template.nuisance_payload)
        if strata is None:
            raise ValueError(f"{template_id}: balanced strata disappeared")
        counts: dict[str, Counter[int]] = {stratum: Counter() for stratum in strata}
        observed_union = set()
        for pair_index in range(8):
            for stratum, members in strata.items():
                indices = tuple(observed_state_index(bits) for bits in members)
                expected = indices[pair_index % len(indices)]
                if routes[pair_index][stratum] != expected:
                    raise ValueError(f"{template_id}/route-{pair_index}: route cycling drifted")
                counts[stratum][expected] += 1
                observed_union.add(expected)
        if observed_union != set(range(8)):
            raise ValueError(f"{template_id}: routes do not cover all eight X states")
        union_sizes[len(observed_union)] += 1
        stratum_diagnostics = {}
        for stratum, members in strata.items():
            multiplicities = [counts[stratum][observed_state_index(bits)] for bits in members]
            if sum(multiplicities) != 8 or max(multiplicities) - min(multiplicities) > 1:
                raise ValueError(f"{template_id}/{stratum}: imbalanced route cycling")
            stratum_sizes[len(members)] += 1
            spreads[max(multiplicities) - min(multiplicities)] += 1
            stratum_diagnostics[stratum] = {
                "member_count": len(members),
                "minimum_selection_multiplicity": min(multiplicities),
                "maximum_selection_multiplicity": max(multiplicities),
                "selection_count": sum(multiplicities),
            }
        diagnostics.append(
            {
                "template_id": template_id,
                "family": template.family,
                "route_count": len(routes),
                "observed_state_union": sorted(observed_union),
                "strata": stratum_diagnostics,
            }
        )
    if set(template_routes) != set(template_by_id):
        raise ValueError("route coverage does not include every evaluation template")
    return {
        "schema_version": ROUTE_COVERAGE_SCHEMA,
        "algorithm_version": ROUTE_CYCLING_ALGORITHM,
        "status": "pass",
        "evaluation_templates": len(template_routes),
        "paired_routes": len(paired),
        "states_per_route": 4,
        "route_indices_per_template": 8,
        "all_observed_states_covered_per_template": True,
        "all_strata_selected_eight_times_per_template": True,
        "per_member_multiplicity_floor_or_ceiling": True,
        "observed_state_union_size_histogram": _histogram(union_sizes.elements()),
        "stratum_member_count_histogram": _histogram(stratum_sizes.elements()),
        "per_member_selection_spread_histogram": _histogram(spreads.elements()),
        "route_selection_hash": object_hash(route_details),
        "template_diagnostics_hash": object_hash(diagnostics),
    }


def _decision(gold_by_id: Mapping[str, Mapping[str, Any]], row: Mapping[str, Any]) -> str:
    """Return the gold decision of a public row."""
    return str(gold_by_id[str(row["id"])]["gold"]["decision"])


def _unit_balance(
    evaluator_units: Mapping[str, list[Mapping[str, Any]]],
    paired_units: Mapping[str, list[Mapping[str, Any]]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify every unit's decision balance, query complements and payload balance."""
    for unit_id, rows in evaluator_units.items():
        oracles = [gold_by_id[str(row["id"])]["oracle"] for row in rows]
        if (
            len(rows) != 4
            or Counter(_decision(gold_by_id, row) for row in rows)
            != Counter({"ANSWER": 2, "ABSTAIN": 2})
            or len({str(row["paired_query_unit_id"]) for row in rows}) != 1
            or len({str(oracle["query_feature"]) for oracle in oracles}) != 1
        ):
            raise ValueError(f"{unit_id}: invalid four-cell evaluator unit")
    complementary_pairs = 0
    answer_balanced = 0
    abstain_balanced = 0
    for unit_id, rows in paired_units.items():
        if len(rows) != 8 or Counter(_decision(gold_by_id, row) for row in rows) != Counter(
            {"ANSWER": 4, "ABSTAIN": 4}
        ):
            raise ValueError(f"{unit_id}: incomplete paired eight-cell unit")
        states: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            states[str(row["paired_state_id"])].append(row)
        if len(states) != 4:
            raise ValueError(f"{unit_id}: expected four paired states")
        for state_id, state_rows in states.items():
            if (
                len(state_rows) != 2
                or Counter(_decision(gold_by_id, row) for row in state_rows)
                != Counter({"ANSWER": 1, "ABSTAIN": 1})
                or len(
                    {
                        str(gold_by_id[str(row["id"])]["oracle"]["query_feature"])
                        for row in state_rows
                    }
                )
                != 2
                or len({object_hash(query_masked_state(row)) for row in state_rows}) != 1
            ):
                raise ValueError(
                    f"{unit_id}/{state_id}: query pair is not complementary "
                    "over byte-identical raw state"
                )
            complementary_pairs += 1
        answer_bits: dict[int, set[int]] = defaultdict(set)
        abstain_bits: dict[int, set[int]] = defaultdict(set)
        for row in rows:
            private = gold_by_id[str(row["id"])]
            role = int(private["oracle"]["hidden_role_index"])
            bit = int(
                private["oracle"]["role_zero_feature_bit" if role == 0 else "role_one_feature_bit"]
            )
            (answer_bits if private["gold"]["decision"] == "ANSWER" else abstain_bits)[role].add(
                bit
            )
        answer_balanced += answer_bits == {0: {0, 1}, 1: {0, 1}}
        abstain_balanced += abstain_bits == {0: {0, 1}, 1: {0, 1}}
    return {
        "complete_two_answer_two_abstain_evaluator_units": len(evaluator_units),
        "complete_four_answer_four_abstain_paired_groups": len(paired_units),
        "complementary_same_state_query_pairs": complementary_pairs,
        "expected_complementary_same_state_query_pairs": 4 * len(paired_units),
        "both_roles_answer_payload_0_1_units": answer_balanced,
        "both_roles_abstain_target_0_1_units": abstain_balanced,
        "four_cell_evaluator_units": len(evaluator_units),
        "paired_counterfactual_groups": len(paired_units),
        "all_pass": (
            answer_balanced == abstain_balanced == len(paired_units)
            and complementary_pairs == 4 * len(paired_units)
        ),
    }


def acceptance_audit(
    public_records: Sequence[Mapping[str, Any]], gold_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Recompute the full acceptance audit of a public/gold row set."""
    public_ids = [str(item["id"]) for item in public_records]
    gold_ids = [str(item["id"]) for item in gold_records]
    if (
        len(set(public_ids)) != len(public_ids)
        or len(set(gold_ids)) != len(gold_ids)
        or set(public_ids) != set(gold_ids)
    ):
        raise ValueError("the audit requires unique joined public/gold IDs")
    gold_by_id = {str(item["id"]): item for item in gold_records}
    evaluator_units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    paired_units: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for public in public_records:
        evaluator_units[str(public["complete_unit_id"])].append(public)
        paired_units[str(public["paired_query_unit_id"])].append(public)
    balance = _unit_balance(evaluator_units, paired_units, gold_by_id)
    oracle_correct = sum(
        gold_by_id[identifier]["certificate"]["decision"]
        == gold_by_id[identifier]["gold"]["decision"]
        and canonical_json(certificate_answer(gold_by_id[identifier]["certificate"]))
        == canonical_json(gold_by_id[identifier]["gold"]["answer"])
        for identifier in public_ids
    )
    probes = shortcut_probes(public_records, gold_by_id)
    if not all(
        item["exactly_one_half"] and item["all_signatures_exactly_balanced"]
        for item in probes.values()
    ):
        raise ValueError("a deterministic query-blind probe exceeds chance")
    posterior_matching = _posterior_matching(public_records, gold_by_id)
    if not (
        posterior_matching["all_answer_abstain_histograms_identical"]
        and posterior_matching["all_cells_posterior_size_16"]
    ):
        raise ValueError("posterior matching failed")
    templates = template_registry().evaluation
    evaluation_routes = route_registry().partition("evaluation")
    route_counts = evaluation_semantic_counts(evaluation_routes)
    return {
        "schema_version": ACCEPTANCE_AUDIT_SCHEMA,
        "scale_counts": {
            "rows": len(public_records),
            "paired_counterfactual_groups": len(paired_units),
            "four_cell_evaluator_units": len(evaluator_units),
            "primary_semantic_template_clusters": len(templates),
            "evaluation_templates": len(templates),
            "mechanism_families": len(MECHANISM_FAMILIES),
            "query_axes": len(FEATURE_NAMES),
            "query_axis_pairs": len(QUERY_AXIS_PAIRS),
            "query_tasks": len(evaluator_units),
        },
        "decision_counts": dict(
            sorted(Counter(str(item["gold"]["decision"]) for item in gold_records).items())
        ),
        "declared_class_size_histogram": _histogram(
            int(item["hypothesis_declaration"]["class_size"]) for item in public_records
        ),
        "posterior_size_histogram": _histogram(
            int(item["certificate"]["hypothesis_count"]) for item in gold_records
        ),
        "matched_unit_balance": balance,
        "posterior_matching": posterior_matching,
        "paired_cohort_gate": {
            **validate_paired_cohort(public_records),
            "subsampling_unit": "paired_query_unit_id",
            "four_cell_complete_unit_selector_alone_is_safe": False,
            "full_dataset_evaluation_is_safe": True,
            "status": "pass",
        },
        "template_semantic_packing": template_registry().packing_audit,
        "route_state_coverage": _route_state_coverage(public_records, gold_by_id),
        "shortcut_probes": probes,
        "exact_oracle": {
            "correct": oracle_correct,
            "total": len(public_records),
            "joint_accuracy": oracle_correct / len(public_records),
            "exactly_one": oracle_correct == len(public_records),
        },
        "semantic_scale": {
            "evaluation_templates": len(templates),
            "unique_evaluation_template_full_semantics": len(
                {item.full_semantic_hash for item in templates}
            ),
            "unique_evaluation_template_paired_query_behaviors": len(
                {item.paired_behavior_hash for item in templates}
            ),
            "evaluation_paired_counterfactual_groups": len(evaluation_routes),
            "primary_generalization_clusters": len(templates),
            "evaluation_query_tasks": 2 * len(evaluation_routes),
            "unique_evaluation_full_class_semantics": route_counts["full_class_semantics"],
            "unique_evaluation_logical_unit_semantics": route_counts["logical_unit_semantics"],
            "unique_evaluation_paired_query_behaviors": route_counts["paired_query_behaviors"],
            "unique_evaluation_query_tasks": route_counts["query_tasks"],
            "nuisance_only_variants_counted_as_query_tasks": 0,
        },
        "permutation_invariance": {
            "join_key": "id",
            "audit_recomputed_after_reverse_order": False,
            "status": "checked_by_independent_audit",
        },
    }
