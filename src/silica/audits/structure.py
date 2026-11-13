"""Census of the decision structure of every published program, and two guard probes.

Each published program's decision is a function of the three observed bits:
which of the two query-facing coordinates the survivors pin.  The census
recovers that guard function behaviourally, by enumerating the declared
universe, counts how many distinct guards the release contains, and compares
two costs: the stated enumerate-and-filter procedure, and the smallest formula
that computes the guard from the three bits.

The probes ask whether the guard is a shortcut.  A gradient-boosted classifier
sees every evidence-side and query-side feature except the guard value, under
held-out-template and held-out-family splits; a second one is also given the
guard value.  The first must stay at chance; the second shows the decision is
trivial once the program has been evaluated on the evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from ..dsl import accepted_indices

Record = tuple[Mapping[str, Any], Mapping[str, Any]]


class GuardStructureError(RuntimeError):
    """Raised when a program violates a property the census assumes."""


def guard_table(
    public: Mapping[str, Any], observed_features: Sequence[str], role_features: Sequence[str]
) -> tuple[int, ...]:
    """Return, for each of the eight observed states, which role the survivors pin."""
    if len(observed_features) != 3 or len(role_features) != 2:
        raise GuardStructureError("expected three observed features and two roles")
    declaration = public["hypothesis_declaration"]
    names = [item["name"] for item in declaration["feature_variables_lsb_to_msb"]]
    admitted = [
        {name: (index >> offset) & 1 for offset, name in enumerate(names)}
        for index in accepted_indices(declaration["boolean_program"])
    ]
    table = []
    for state_index in range(8):
        pinned = {
            name: (state_index >> position) & 1 for position, name in enumerate(observed_features)
        }
        survivors = [
            bits for bits in admitted if all(bits[name] == value for name, value in pinned.items())
        ]
        if not survivors:
            raise GuardStructureError("an observable state has no surviving hypothesis")
        constant = [
            index
            for index, role in enumerate(role_features)
            if len({bits[role] for bits in survivors}) == 1
        ]
        if len(constant) != 1:
            raise GuardStructureError(
                f"state {state_index} leaves {len(constant)} constant role coordinates"
            )
        table.append(constant[0])
    return tuple(table)


@lru_cache(maxsize=1)
def _formula_costs() -> dict[int, int]:
    """Return the exact minimal formula size of every Boolean function of three variables.

    Cost counts not/and/or/xor/eq applications with variables and constants
    free, relaxed to a fixed point over all 256 truth tables.  Formula size
    forbids shared subexpressions, so it can only overstate the work needed.
    """
    mask = 0xFF
    cost = {0: 0, mask: 0}
    for position in range(3):
        cost[sum(((state >> position) & 1) << state for state in range(8))] = 0
    changed = True
    while changed:
        changed = False
        known = list(cost.items())
        for left, left_cost in known:
            if cost.get((~left) & mask, 1 << 30) > left_cost + 1:
                cost[(~left) & mask] = left_cost + 1
                changed = True
            for right, right_cost in known:
                total = left_cost + right_cost + 1
                for value in (left & right, left | right, left ^ right, (~(left ^ right)) & mask):
                    if cost.get(value, 1 << 30) > total:
                        cost[value] = total
                        changed = True
    if len(cost) != 256:
        raise GuardStructureError("the DSL basis does not span all three-bit functions")
    return cost


def minimal_formula_size(table: Sequence[int]) -> int:
    """Return the minimal formula size of one guard function."""
    return _formula_costs()[sum(1 << index for index, value in enumerate(table) if value)]


def _roles(gold: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, str]]:
    """Return a row's sorted observed features and its two role features."""
    oracle = gold["oracle"]
    observed = tuple(sorted(oracle["ordered_evidence_features"]))
    return observed, (oracle["role_zero_feature"], oracle["role_one_feature"])


def _spread(values: Sequence[int]) -> dict[str, int]:
    """Return the minimum, upper median and maximum of a list."""
    ordered = sorted(values)
    return {"min": ordered[0], "median": ordered[len(ordered) // 2], "max": ordered[-1]}


def census(records: Sequence[Record]) -> dict[str, Any]:
    """Census the guard functions and both cost accounts over every published program."""
    by_program: dict[str, dict[str, Any]] = {}
    for public, gold in records:
        program_hash = public["hypothesis_declaration"]["boolean_program_hash"]
        observed, roles = _roles(gold)
        entry = by_program.get(program_hash)
        if entry is None:
            entry = by_program[program_hash] = {
                "family": public["mechanism"]["family"],
                "template": public["mechanism"]["template_hash"],
                "observed_features": observed,
                "role_features": roles,
                "guard_table": guard_table(public, observed, roles),
                "operator_count": public["structural_complexity"]["operator_count"],
                "maximum_dependency_depth": public["structural_complexity"][
                    "maximum_dependency_depth"
                ],
                "rows": 0,
            }
        elif entry["observed_features"] != observed or entry["role_features"] != roles:
            raise GuardStructureError(
                f"program {program_hash} changes its observed features or roles"
            )
        entry["rows"] += 1
    entries = list(by_program.values())
    table_counts = Counter(entry["guard_table"] for entry in entries)
    by_family: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    for entry in entries:
        by_family[entry["family"]].add(entry["guard_table"])
    minimal = [minimal_formula_size(entry["guard_table"]) for entry in entries]
    operator_counts = [entry["operator_count"] for entry in entries]
    return {
        "template_count": len({entry["template"] for entry in entries}),
        "program_count": len(entries),
        "row_count": sum(entry["rows"] for entry in entries),
        "rows_per_program": sorted({entry["rows"] for entry in entries}),
        "distinct_guard_functions": len(table_counts),
        "distinct_guard_functions_by_family": {
            family: len(values) for family, values in sorted(by_family.items())
        },
        "guard_function_multiplicity": {
            "max_programs_sharing_one_guard": max(table_counts.values()),
            "programs_with_a_unique_guard": sum(1 for count in table_counts.values() if count == 1),
            "histogram": dict(sorted(Counter(table_counts.values()).items())),
        },
        "guard_truth_tables": {
            "".join(str(bit) for bit in table): count
            for table, count in sorted(table_counts.items())
        },
        "stated_procedure_operator_evaluations": {
            "definition": (
                "256 declared-universe assignments x operator_count of the "
                "published program, before the 16 survivor query evaluations"
            ),
            **_spread([256 * count for count in operator_counts]),
        },
        "published_program_operator_count": _spread(operator_counts),
        "published_program_dependency_depth": _spread(
            [entry["maximum_dependency_depth"] for entry in entries]
        ),
        "minimal_decision_formula_size": {
            "definition": (
                "fewest not/and/or/xor/eq operator applications that compute the "
                "answerable-role indicator from the three observed feature bits"
            ),
            **_spread(minimal),
            "histogram": dict(sorted(Counter(minimal).items())),
        },
    }


def _probe_examples(
    records: Sequence[Record],
) -> tuple[list[dict[str, Any]], list[int], list[str], list[str]]:
    """Build the feature rows both probes are scored on."""
    guards: dict[str, tuple[int, ...]] = {}
    rows, labels, templates, families = [], [], [], []
    for public, gold in records:
        oracle = gold["oracle"]
        program_hash = public["hypothesis_declaration"]["boolean_program_hash"]
        observed, roles = _roles(gold)
        if program_hash not in guards:
            guards[program_hash] = guard_table(public, observed, roles)
        state = tuple(oracle["observed_feature_bits"][name] for name in observed)
        declaration = public["hypothesis_declaration"]
        rows.append(
            {
                "bit0": state[0],
                "bit1": state[1],
                "bit2": state[2],
                "query_role": roles.index(oracle["query_feature"]),
                "family": public["mechanism"]["family"],
                "task": public["task"],
                "class_size": declaration["class_size"],
                "base_class_size": declaration["base_class_size"],
                "guard_value": guards[program_hash][
                    sum(bit << position for position, bit in enumerate(state))
                ],
            }
        )
        labels.append(1 if gold["gold"]["decision"] == "ANSWER" else 0)
        templates.append(public["mechanism"]["template_hash"])
        families.append(public["mechanism"]["family"])
    return rows, labels, templates, families


def _fit_probe(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    use_guard: bool,
    folds: int,
) -> dict[str, Any]:
    """Return the grouped cross-validated balanced accuracy of one probe."""
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    keys = ["bit0", "bit1", "bit2", "query_role", "class_size", "base_class_size"]
    if use_guard:
        keys.append("guard_value")
    family_values = sorted({row["family"] for row in rows})
    task_values = sorted({row["task"] for row in rows})
    matrix = np.array(
        [
            [float(row[key]) for key in keys]
            + [1.0 if row["family"] == value else 0.0 for value in family_values]
            + [1.0 if row["task"] == value else 0.0 for value in task_values]
            for row in rows
        ],
        dtype=float,
    )
    target = np.array(labels, dtype=int)
    correct_positive = correct_negative = total_positive = total_negative = 0
    for train, test in GroupKFold(n_splits=folds).split(matrix, target, np.array(groups)):
        model = GradientBoostingClassifier(random_state=0)
        model.fit(matrix[train], target[train])
        predicted = model.predict(matrix[test])
        truth = target[test]
        total_positive += int((truth == 1).sum())
        total_negative += int((truth == 0).sum())
        correct_positive += int(((truth == 1) & (predicted == 1)).sum())
        correct_negative += int(((truth == 0) & (predicted == 0)).sum())
    recall_positive = correct_positive / total_positive if total_positive else 0.0
    recall_negative = correct_negative / total_negative if total_negative else 0.0
    return {
        "balanced_decision_accuracy": (recall_positive + recall_negative) / 2,
        "answerable_recall": recall_positive,
        "underdetermined_recall": recall_negative,
        "rows": len(rows),
        "folds": folds,
    }


def guard_probes(records: Sequence[Record]) -> dict[str, Any]:
    """Score the guard-blind and guard-given probes under both grouped splits."""
    rows, labels, templates, families = _probe_examples(records)
    return {
        "interpretation": (
            "The blind probe is a shortcut gate: it sees every evidence-side and "
            "query-side feature except the value of the guard, which no reader can "
            "obtain without evaluating the published program on the evidence.  The "
            "guard-given probe is a placement test, not a competitor: it shows that "
            "the decision is a two-line rule once the program has been evaluated, "
            "which is the claim the benchmark makes about its own task."
        ),
        **{
            name: {
                "held_out_template": _fit_probe(
                    rows, labels, templates, use_guard=use_guard, folds=8
                ),
                "held_out_family": _fit_probe(rows, labels, families, use_guard=use_guard, folds=4),
            }
            for name, use_guard in (("guard_blind", False), ("guard_given", True))
        },
    }
