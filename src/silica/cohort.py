"""Public cohort selection that never splits a paired-query unit.

The evaluated models saw either the full release (12,288 rows) or its common
cohort of 1,536 rows: one paired-query unit (eight rows) for each of the 192
evaluation templates.  The selector reads public rows only.  It ranks query
routes and templates by seeded hashes and assigns each template its route
cyclically, so every family and every query-axis pair is represented equally.
The cohort hash commits to the complete selection.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._frozen import ADAPTER_PROTOCOL, ADAPTER_VERSION, COHORT_METHOD, COHORT_SEED, TRACK
from .certificates import TASKS
from .util import canonical_json, object_hash

ROWS_PER_UNIT = 4
ROWS_PER_PAIRED_UNIT = 8
TEMPLATE_COUNT = 192
FAMILY_COUNT = 4
TEMPLATES_PER_FAMILY = 48
ROUTES_PER_TEMPLATE = 8
COMMON_COHORT_ROWS = TEMPLATE_COUNT * ROWS_PER_PAIRED_UNIT
SELECTION_MODE = "paired_query_unit"
ADAPTER_PROTOCOL_HASH = object_hash(ADAPTER_PROTOCOL)


def _rank(seed: str, *parts: str) -> str:
    """Return a seeded ordering key."""
    return hashlib.sha256((seed + "\x1f" + "\x1f".join(parts)).encode("utf-8")).hexdigest()


def _text(value: Any, label: str) -> str:
    """Return a nonempty string or raise."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_paired_cohort(public_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Fail unless a row set consists of whole paired-query units."""
    if not public_records:
        raise ValueError("a paired cohort must be nonempty")
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in public_records:
        identifiers = (
            row.get("paired_query_unit_id"),
            row.get("complete_unit_id"),
            row.get("paired_state_id"),
        )
        if not all(isinstance(value, str) and value for value in identifiers):
            raise ValueError("a cohort row lacks paired routing IDs")
        paired[str(row["paired_query_unit_id"])].append(row)
    evaluator_units = 0
    for paired_id, rows in paired.items():
        complete_counts = Counter(str(row["complete_unit_id"]) for row in rows)
        state_counts = Counter(str(row["paired_state_id"]) for row in rows)
        if (
            len(rows) != ROWS_PER_PAIRED_UNIT
            or sorted(complete_counts.values()) != [4, 4]
            or sorted(state_counts.values()) != [2, 2, 2, 2]
        ):
            raise ValueError(f"{paired_id}: subsampling broke a paired-query group")
        evaluator_units += len(complete_counts)
    return {
        "rows": len(public_records),
        "paired_query_groups": len(paired),
        "four_cell_evaluator_units": evaluator_units,
    }


def public_query_axis(item: Mapping[str, Any]) -> str:
    """Return the role-free axis label of a row's query, e.g. ``lexicon:noun_pair_01``."""
    query = item.get("query")
    declaration = item.get("hypothesis_declaration")
    label = f"row {item.get('id')!r}"
    if not isinstance(query, Mapping) or not isinstance(declaration, Mapping):
        raise ValueError(f"{label} lacks query/declaration data")
    if item.get("task") == "grammar_feature":
        feature = _text(query.get("feature"), f"{label} query.feature")
        axes = declaration.get("grammar_axes")
        if not isinstance(axes, list) or feature not in {
            axis.get("name") for axis in axes if isinstance(axis, Mapping)
        }:
            raise ValueError(f"{label} queries an undeclared grammar axis")
        return f"grammar:{feature}"
    if item.get("task") == "lexicon":
        category = _text(query.get("category"), f"{label} query.category")
        form = _text(query.get("form"), f"{label} query.form")
        axes = declaration.get("lexical_axes")
        matches = [
            _text(axis.get("name"), f"{label} lexical axis name")
            for axis in (axes if isinstance(axes, list) else [])
            if isinstance(axis, Mapping)
            and axis.get("category") == category
            and isinstance(axis.get("forms"), list)
            and form in axis["forms"]
        ]
        if len(matches) != 1:
            raise ValueError(f"{label} query does not identify one declared lexical axis")
        return f"lexicon:{matches[0]}"
    raise ValueError(f"{label} has an unsupported task")


def _experiment_signature(item: Mapping[str, Any]) -> tuple[tuple[str, str, str], ...]:
    """Project observations to their outcome-blind experiment identities."""
    evidence = item.get("evidence")
    label = f"row {item.get('id')!r}"
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{label} evidence must be a non-empty array")
    signature = []
    for index, observation in enumerate(evidence):
        if not isinstance(observation, Mapping):
            raise ValueError(f"{label} evidence[{index}] is not an object")
        signature.append(
            tuple(
                _text(observation.get(name), f"{label} evidence[{index}].{name}")
                for name in ("channel", "category", "semantic")
            )
        )
    if len(set(signature)) != len(signature):
        raise ValueError(f"{label} repeats a public experiment identity")
    return tuple(signature)


def _unit_metadata(unit_id: str, group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate one four-row unit and return its public metadata."""
    if len(group) != ROWS_PER_UNIT:
        raise ValueError(f"unit {unit_id!r} has {len(group)} rows, expected 4")
    cell_ids = {
        _text(item.get("counterfactual_cell_id"), f"unit {unit_id!r} cell id") for item in group
    }
    if len(cell_ids) != ROWS_PER_UNIT:
        raise ValueError(f"unit {unit_id!r} repeats a cell identity")
    for name in ("dataset_version", "track", "split", "language_id", "query_id", "policy", "task"):
        if len({canonical_json(item.get(name)) for item in group}) != 1:
            raise ValueError(f"unit {unit_id!r} changes public {name} identity")
    if group[0]["track"] != TRACK:
        raise ValueError(f"unit {unit_id!r} is not on the release track")
    if group[0]["task"] not in TASKS:
        raise ValueError(f"unit {unit_id!r} has an unsupported task")
    declarations = {object_hash(item["hypothesis_declaration"]) for item in group}
    queries = {object_hash(item["query"]) for item in group}
    budgets = {object_hash(item["budget"]) for item in group}
    budget_ids = {
        _text(
            item["budget"].get("id") if isinstance(item.get("budget"), Mapping) else None,
            "budget.id",
        )
        for item in group
    }
    experiments = {object_hash(_experiment_signature(item)) for item in group}
    observations = {object_hash(item["evidence"]) for item in group}
    mechanisms = {object_hash(item.get("mechanism")) for item in group}
    if len(declarations) != 1 or len(queries) != 1 or len(budgets) != 1 or len(budget_ids) != 1:
        raise ValueError(f"unit {unit_id!r} changes its declaration, query or budget")
    if len(experiments) != 1 or len(mechanisms) != 1:
        raise ValueError(f"unit {unit_id!r} changes its experiments or mechanism")
    if len(observations) != ROWS_PER_UNIT:
        raise ValueError(f"unit {unit_id!r} lacks four distinct observations")
    mechanism = group[0].get("mechanism")
    if not isinstance(mechanism, Mapping):
        raise ValueError(f"unit {unit_id!r} lacks public mechanism data")
    paired_ids = {_text(item.get("paired_query_unit_id"), "paired_query_unit_id") for item in group}
    paired_states = [_text(item.get("paired_state_id"), "paired_state_id") for item in group]
    if len(paired_ids) != 1 or len(set(paired_states)) != ROWS_PER_UNIT:
        raise ValueError(f"unit {unit_id!r} changes its paired-query identity")
    return {
        "complete_unit_id": unit_id,
        "family": _text(mechanism.get("family"), "mechanism.family"),
        "query_feature": public_query_axis(group[0]),
        "task": str(group[0]["task"]),
        "declaration_hash": next(iter(declarations)),
        "query_hash": next(iter(queries)),
        "budget_hash": next(iter(budgets)),
        "experiment_identity_hash": next(iter(experiments)),
        "observation_hashes": sorted(observations),
        "item_ids": sorted(str(item["id"]) for item in group),
        "counterfactual_cell_ids": sorted(cell_ids),
        "paired_query_unit_id": next(iter(paired_ids)),
        "paired_state_ids": sorted(paired_states),
        "template_id": _text(mechanism.get("template_id"), "mechanism.template_id"),
    }


def _paired_metadata(
    pair_id: str, group: Sequence[Mapping[str, Any]], units: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate one eight-row paired-query unit and return its public metadata."""
    if len(group) != ROWS_PER_PAIRED_UNIT:
        raise ValueError(f"paired unit {pair_id!r} has {len(group)} rows, expected 8")
    complete_counts = Counter(
        _text(item.get("complete_unit_id"), "complete_unit_id") for item in group
    )
    if len(complete_counts) != 2 or set(complete_counts.values()) != {ROWS_PER_UNIT}:
        raise ValueError(f"paired unit {pair_id!r} must contain two complete four-row units")
    complete = [units[unit_id] for unit_id in sorted(complete_counts)]
    if any(unit.get("paired_query_unit_id") != pair_id for unit in complete):
        raise ValueError(f"paired unit {pair_id!r} changes parent identity")
    for name in (
        "family",
        "template_id",
        "declaration_hash",
        "budget_hash",
        "experiment_identity_hash",
    ):
        if len({canonical_json(unit.get(name)) for unit in complete}) != 1:
            raise ValueError(f"paired unit {pair_id!r} changes public {name}")
    query_features = sorted(str(unit["query_feature"]) for unit in complete)
    query_hashes = sorted(str(unit["query_hash"]) for unit in complete)
    if len(set(query_features)) != 2 or len(set(query_hashes)) != 2:
        raise ValueError(f"paired unit {pair_id!r} lacks two distinct query orientations")
    if len({canonical_json(unit["observation_hashes"]) for unit in complete}) != 1:
        raise ValueError(f"paired unit {pair_id!r} changes its four evidence states")
    states: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in group:
        states[_text(item.get("paired_state_id"), "paired_state_id")].append(item)
    if len(states) != ROWS_PER_UNIT or any(len(rows) != 2 for rows in states.values()):
        raise ValueError(f"paired unit {pair_id!r} must contain four states with two queries each")
    for state_id, rows in states.items():
        if {str(row["complete_unit_id"]) for row in rows} != set(complete_counts):
            raise ValueError(f"paired state {state_id!r} does not span both complete units")
        for name in (
            "evidence",
            "dataset_version",
            "track",
            "split",
            "language_id",
            "policy",
            "budget",
            "hypothesis_declaration",
            "mechanism",
        ):
            if len({canonical_json(row.get(name)) for row in rows}) != 1:
                raise ValueError(f"paired state {state_id!r} changes public {name}")
    return {
        "paired_query_unit_id": pair_id,
        "family": str(complete[0]["family"]),
        "template_id": str(complete[0]["template_id"]),
        "query_features": query_features,
        "query_pair_hash": object_hash(query_features),
        "query_hashes": query_hashes,
        "declaration_hash": str(complete[0]["declaration_hash"]),
        "budget_hash": str(complete[0]["budget_hash"]),
        "experiment_identity_hash": str(complete[0]["experiment_identity_hash"]),
        "complete_unit_ids": sorted(complete_counts),
        "paired_state_ids": sorted(states),
        "item_ids": sorted(str(item["id"]) for item in group),
    }


def _select_balanced(
    candidates: Sequence[dict[str, Any]], *, groups_per_template: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Give every template the same number of routes, cycling through a seeded route order."""
    by_template: dict[str, dict[str, dict[str, Any]]] = {}
    template_family: dict[str, str] = {}
    for candidate in candidates:
        template_id = str(candidate["template_id"])
        if template_family.setdefault(template_id, str(candidate["family"])) != candidate["family"]:
            raise ValueError(f"template {template_id!r} changes mechanism family")
        routes = by_template.setdefault(template_id, {})
        if candidate["query_pair_hash"] in routes:
            raise ValueError(f"template {template_id!r} repeats a query-axis-pair route")
        routes[str(candidate["query_pair_hash"])] = candidate
    family_templates: dict[str, list[str]] = defaultdict(list)
    for template_id, family in template_family.items():
        family_templates[family].append(template_id)
    route_sets = {frozenset(routes) for routes in by_template.values()}
    if (
        len(by_template) != TEMPLATE_COUNT
        or len(family_templates) != FAMILY_COUNT
        or {len(values) for values in family_templates.values()} != {TEMPLATES_PER_FAMILY}
        or len(route_sets) != 1
        or len(next(iter(route_sets))) != ROUTES_PER_TEMPLATE
    ):
        raise ValueError("cohort selection requires the complete 192-template, eight-route release")
    routes = next(iter(route_sets))
    route_order = sorted(routes, key=lambda route: (_rank(seed, "route", route), route))
    selected: list[dict[str, Any]] = []
    for family in sorted(family_templates):
        template_order = sorted(
            family_templates[family],
            key=lambda template_id: (_rank(seed, "template", family, template_id), template_id),
        )
        for template_index, template_id in enumerate(template_order):
            chosen = {
                route_order[(template_index + offset) % ROUTES_PER_TEMPLATE]
                for offset in range(groups_per_template)
            }
            selected.extend(by_template[template_id][route] for route in sorted(chosen))
    route_counts = Counter(str(item["query_pair_hash"]) for item in selected)
    family_counts = Counter(str(item["family"]) for item in selected)
    per_route = TEMPLATE_COUNT * groups_per_template // ROUTES_PER_TEMPLATE
    per_family_route = TEMPLATES_PER_FAMILY * groups_per_template // ROUTES_PER_TEMPLATE
    if (
        set(Counter(str(item["template_id"]) for item in selected).values())
        != {groups_per_template}
        or set(route_counts.values()) != {per_route}
        or set(family_counts.values()) != {TEMPLATES_PER_FAMILY * groups_per_template}
        or set(
            Counter(
                (str(item["family"]), str(item["query_pair_hash"])) for item in selected
            ).values()
        )
        != {per_family_route}
    ):
        raise AssertionError("paired cohort balancing schedule drifted")
    selected.sort(key=lambda value: str(value["paired_query_unit_id"]))
    return selected, {
        "strategy": "equal_template_seeded_cyclic_query_route_v1",
        "semantic_templates": TEMPLATE_COUNT,
        "families": len(family_counts),
        "templates_per_family": TEMPLATES_PER_FAMILY,
        "available_query_axis_pair_routes_per_template": ROUTES_PER_TEMPLATE,
        "selected_groups_per_template": groups_per_template,
        "selected_groups_per_query_axis_pair_route": per_route,
        "selected_groups_per_family_query_axis_pair_route": per_family_route,
        "template_group_count_histogram": {str(groups_per_template): TEMPLATE_COUNT},
        "query_axis_pair_route_counts": dict(sorted(route_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


@dataclass(frozen=True)
class CohortSelection:
    """A selected public cohort, its units, and its committing hash."""

    items: tuple[dict[str, Any], ...]
    units: tuple[dict[str, Any], ...]
    sampling_units: tuple[dict[str, Any], ...]
    selection_balance: dict[str, Any]
    cohort_hash: str
    seed: str
    eligible_count: int

    def metadata(self) -> dict[str, Any]:
        """Return a compact description of the selection."""
        return {
            "method": COHORT_METHOD,
            "adapter_version": ADAPTER_VERSION,
            "adapter_protocol_hash": ADAPTER_PROTOCOL_HASH,
            "seed": self.seed,
            "eligible_count": self.eligible_count,
            "selected_count": len(self.items),
            "selected_complete_unit_count": len(self.units),
            "selected_sampling_unit_count": len(self.sampling_units),
            "selection_mode": SELECTION_MODE,
            "selection_balance": dict(self.selection_balance),
            "cohort_hash": self.cohort_hash,
        }


def select_cohort(
    items: Sequence[Mapping[str, Any]], *, rows: int = COMMON_COHORT_ROWS, seed: str = COHORT_SEED
) -> CohortSelection:
    """Select whole paired-query units: ``rows`` must be a multiple of 1,536 up to 12,288."""
    if not items:
        raise ValueError("no public rows to select from")
    if rows <= 0 or rows % COMMON_COHORT_ROWS or rows // COMMON_COHORT_ROWS > ROUTES_PER_TEMPLATE:
        raise ValueError(
            f"cohort size must be a positive multiple of {COMMON_COHORT_ROWS} up to 12,288"
        )
    if rows > len(items):
        raise ValueError("cohort size exceeds the eligible public rows")
    public_items = [dict(original) for original in items]
    identifiers = [_text(item.get("id"), "public row id") for item in public_items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("public rows contain duplicate ids")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in public_items:
        if item.get("track") != TRACK:
            raise ValueError("the selector accepts only public rows of the release track")
        grouped[_text(item.get("complete_unit_id"), "complete_unit_id")].append(item)
        pair_groups[_text(item.get("paired_query_unit_id"), "paired_query_unit_id")].append(item)
    units = [_unit_metadata(unit_id, grouped[unit_id]) for unit_id in sorted(grouped)]
    units_by_id = {str(unit["complete_unit_id"]): unit for unit in units}
    candidates = [
        _paired_metadata(pair_id, pair_groups[pair_id], units_by_id)
        for pair_id in sorted(pair_groups)
    ]
    selected, balance = _select_balanced(
        candidates, groups_per_template=rows // COMMON_COHORT_ROWS, seed=seed
    )
    selected_pairs = {str(candidate["paired_query_unit_id"]) for candidate in selected}
    selected_units = {
        str(unit_id) for candidate in selected for unit_id in candidate["complete_unit_ids"]
    }
    public_units = sorted(
        (unit for unit in units if str(unit["complete_unit_id"]) in selected_units),
        key=lambda value: str(value["complete_unit_id"]),
    )
    selected_items = sorted(
        (item for item in public_items if str(item["paired_query_unit_id"]) in selected_pairs),
        key=lambda item: (
            str(item["paired_query_unit_id"]),
            str(item["paired_state_id"]),
            str(item["complete_unit_id"]),
            str(item["counterfactual_cell_id"]),
            str(item["id"]),
        ),
    )
    payload = {
        "method": COHORT_METHOD,
        "adapter_version": ADAPTER_VERSION,
        "adapter_protocol_hash": ADAPTER_PROTOCOL_HASH,
        "seed": seed,
        "selection_mode": SELECTION_MODE,
        "selection_balance": balance,
        "complete_units": public_units,
        "sampling_units": selected,
        "selected_ids": [str(item["id"]) for item in selected_items],
    }
    return CohortSelection(
        items=tuple(selected_items),
        units=tuple(public_units),
        sampling_units=tuple(selected),
        selection_balance=balance,
        cohort_hash=object_hash(payload),
        seed=seed,
        eligible_count=len(items),
    )
