"""Mechanism templates: guarded switches over five placeholder variables.

Every template is a Boolean program over placeholders ``Q, N, X0, X1, X2``.
Its guard, a function of the observed placeholders ``X0..X2``, decides which
of the two query-facing placeholders the program pins: when the guard holds,
``Q`` equals a payload expression and ``N`` is free; otherwise ``N`` equals a
nuisance payload and ``Q`` is free.  Every program therefore admits exactly
16 of the 32 placeholder assignments, so a full observation state always
leaves the same posterior size while only one of the two queries is
answerable.

The registry is built deterministically from four guard families.  Candidates
are deduplicated by behaviour, twelve support templates are fixed by hash, and
48 evaluation templates per family are packed so that every pair of selected
templates differs in at least four paired query-behaviour cells.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from typing import Any

from ._frozen import TEMPLATE_PACKING_ALGORITHM, TEMPLATE_PACKING_SCHEMA
from .dsl import (
    DSL_VERSION,
    canonicalize_expression,
    canonicalize_program,
    evaluate_expression,
    index_assignment,
    program_accepts,
)
from .util import canonical_json, object_hash

PLACEHOLDER_VARIABLES = ("Q", "N", "X0", "X1", "X2")
OBSERVED_PLACEHOLDERS = ("X0", "X1", "X2")
MECHANISM_FAMILIES = (
    "parity_guard",
    "majority_guard",
    "equality_chain",
    "selector_multiplexer",
)
HELDOUT_MECHANISM_FAMILY = "selector_multiplexer"
EVALUATION_TEMPLATES_PER_FAMILY = 48
PACKING_MINIMUM_DISTANCE = 4
PAIRED_BEHAVIOR_LENGTH = 16
EXPECTED_CANDIDATE_COUNTS = {
    "parity_guard": 914,
    "majority_guard": 1112,
    "equality_chain": 268,
    "selector_multiplexer": 350,
}
EXPECTED_UNIQUE_PAIRED_BEHAVIORS = 2411
EXPECTED_SUPPORT_FILTERED_COUNTS = {
    "parity_guard": 854,
    "majority_guard": 1021,
    "equality_chain": 229,
    "selector_multiplexer": 340,
}
FIXED_SUPPORT_TEMPLATE_HASHES = {
    "parity_guard": (
        "95798887b83f753ac5ebf4f48257f4370abdf5a4107196294a0a1384de45c451",
        "662b40eeb1f81d818354561e27f332c4612defe84a9dd1318e8fbb3fd9c4bdfa",
        "0c07880e5d58c1903d65b1fd90cfe4e3c6a8bd87283711e1dd690c9cb8dfd25d",
        "2590115e8665cf1e0c26b4c1b4543bb7be7c1f4640da08d2af0f3fecb93bc038",
    ),
    "majority_guard": (
        "f7be8a6b470afc653697fd8afa9f7c37560bbc8b18fbb57a7a25aeb4c6b3573b",
        "011a3d28b023c388ad731272cf26558f5d2fb97435e774045197129c05d09128",
        "c40c06ae7fddec78f779370960fcac6ef3d93284cadcf570d800e7d3687458c3",
        "92a05449bad43007556506d3da29a332c71638dc87ca0df4de179993f16bcb48",
    ),
    "equality_chain": (
        "9bdffe1730c1b95afbac279f76e1c5813c5900ccb79f76e3ed79fba22c545a3a",
        "1eacabdaf42b389f61de918c5437c842eea94967fe78826519a3e14ed3913b5e",
        "6d08fb3d6b0143fb7e06861242d5bb6bb59c91749033d47d3caa180799ac4493",
        "239fe81aa3a2a4e64b4d869fd06cfa046403deb985205abb36c069ee87aec629",
    ),
    "selector_multiplexer": (),
}
PLACEHOLDER_FULL_SEMANTIC_HASH_DEFINITION = {
    "hash": "sha256_of_canonical_json",
    "payload": "accepted_assignment_indices",
    "variables_lsb_to_msb": list(PLACEHOLDER_VARIABLES),
    "assignment_indices": "0_through_31_inclusive",
}
QUERY_BEHAVIOR_HASH_DEFINITION = {
    "hash": "sha256_of_canonical_json",
    "payload": (
        "eight X0/X1/X2 cells in binary-index order; each cell is ABSTAIN, ANSWER:0, or ANSWER:1"
    ),
}
STRATA = ("active_0", "inactive_0", "active_1", "inactive_1")

Expression = dict[str, Any]


def var(name: str) -> Expression:
    """Return a variable expression."""
    return {"op": "var", "name": name}


def const(value: bool) -> Expression:
    """Return a constant expression."""
    return {"op": "const", "value": value}


def negate(argument: Mapping[str, Any]) -> Expression:
    """Return the negation of an expression."""
    return {"op": "not", "arg": dict(argument)}


def apply(operation: str, *arguments: Mapping[str, Any]) -> Expression:
    """Return an n-ary operator expression."""
    return {"op": operation, "args": [dict(item) for item in arguments]}


def _literal(name: str, negated: bool = False) -> Expression:
    """Return a possibly negated variable."""
    expression = var(name)
    return negate(expression) if negated else expression


def _majority(arguments: Sequence[Mapping[str, Any]]) -> Expression:
    """Return the three-input majority of three expressions."""
    first, second, third = arguments
    return apply(
        "or",
        apply("and", first, second),
        apply("and", first, third),
        apply("and", second, third),
    )


def x_assignment(index: int) -> dict[str, bool]:
    """Assign the observed placeholders from a state index, with Q and N false."""
    return {"Q": False, "N": False, **index_assignment(index, OBSERVED_PLACEHOLDERS)}


def observed_state_index(bits: Sequence[int]) -> int:
    """Return the binary index of three observed placeholder bits."""
    if len(bits) != 3 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("an observed state must contain exactly three binary bits")
    return int(bits[0]) + 2 * int(bits[1]) + 4 * int(bits[2])


def expression_truth(expression: Mapping[str, Any]) -> tuple[int, ...]:
    """Return an expression's truth table over the eight observed states."""
    canonical = canonicalize_expression(expression, allowed_variables=PLACEHOLDER_VARIABLES)
    return tuple(int(evaluate_expression(canonical, x_assignment(index))) for index in range(8))


def _deduplicate(expressions: Iterable[Mapping[str, Any]]) -> tuple[Expression, ...]:
    """Keep the canonically smallest expression of every truth table."""
    by_truth: dict[tuple[int, ...], Expression] = {}
    for raw in expressions:
        expression = canonicalize_expression(raw, allowed_variables=PLACEHOLDER_VARIABLES)
        truth = expression_truth(expression)
        previous = by_truth.get(truth)
        if previous is None or canonical_json(expression) < canonical_json(previous):
            by_truth[truth] = expression
    return tuple(by_truth[key] for key in sorted(by_truth))


def _payload_expressions() -> tuple[Expression, ...]:
    """Enumerate the payload expressions over the observed placeholders."""
    variables = OBSERVED_PLACEHOLDERS
    values: list[Expression] = []
    values.extend(_literal(name, negated) for name in variables for negated in (False, True))
    for left, right in permutations(variables, 2):
        values.extend(
            (
                apply("xor", var(left), var(right)),
                apply("eq", var(left), var(right)),
                apply("and", var(left), var(right)),
                apply("or", var(left), var(right)),
                apply("implies", var(left), var(right)),
            )
        )
    for mask in product((False, True), repeat=3):
        literals = [_literal(name, negated) for name, negated in zip(variables, mask, strict=True)]
        values.extend(
            (
                apply("xor", *literals),
                _majority(literals),
                apply("ite", literals[0], literals[1], literals[2]),
            )
        )
    for selector, left, right in permutations(variables, 3):
        for left_negated, right_negated in product((False, True), repeat=2):
            values.append(
                apply(
                    "ite",
                    var(selector),
                    _literal(left, left_negated),
                    _literal(right, right_negated),
                )
            )
    return _deduplicate(values)


@lru_cache(maxsize=1)
def payload_catalog() -> tuple[tuple[Expression, tuple[int, ...]], ...]:
    """Return every payload expression with its truth table, in hash order."""
    return tuple(
        sorted(
            ((expression, expression_truth(expression)) for expression in _payload_expressions()),
            key=lambda item: object_hash({"expression": item[0], "truth": list(item[1])}),
        )
    )


def _parity_guards() -> tuple[Expression, ...]:
    """Enumerate parity guards and their pairwise combinations."""
    variables = OBSERVED_PLACEHOLDERS
    atoms: list[Expression] = []
    for left, right in combinations(variables, 2):
        atoms.extend((apply("xor", var(left), var(right)), apply("eq", var(left), var(right))))
    triple = apply("xor", *(var(name) for name in variables))
    atoms.extend((triple, negate(triple)))
    guards = list(atoms)
    for left, right in combinations(atoms, 2):
        guards.extend(
            (
                apply("and", left, right),
                apply("or", left, right),
                apply("xor", left, right),
                apply("implies", left, right),
                apply("implies", right, left),
            )
        )
    return _deduplicate(guards)


def _majority_guards() -> tuple[Expression, ...]:
    """Enumerate majority guards and their combinations with one literal."""
    variables = OBSERVED_PLACEHOLDERS
    majorities = [
        _majority([_literal(name, negated) for name, negated in zip(variables, mask, strict=True)])
        for mask in product((False, True), repeat=3)
    ]
    literals = tuple(_literal(name, negated) for name in variables for negated in (False, True))
    guards = list(majorities)
    for majority, literal in product(majorities, literals):
        guards.extend(
            (
                apply("and", majority, literal),
                apply("or", majority, literal),
                apply("xor", majority, literal),
                apply("implies", majority, literal),
            )
        )
    return _deduplicate(guards)


def _chain_guards() -> tuple[Expression, ...]:
    """Enumerate guards built from two chained equality or parity links."""
    guards: list[Expression] = []
    for first, second, third in permutations(OBSERVED_PLACEHOLDERS):
        for left_mode, right_mode in product(("eq", "xor"), repeat=2):
            left = apply(left_mode, var(first), var(second))
            right = apply(right_mode, var(second), var(third))
            guards.extend(
                (
                    apply("and", left, right),
                    apply("or", left, right),
                    apply("xor", left, right),
                    apply("eq", left, right),
                    apply("implies", left, right),
                    apply("implies", right, left),
                )
            )
    return _deduplicate(guards)


def _selector_guards() -> tuple[Expression, ...]:
    """Enumerate multiplexer guards selected by one observed placeholder."""
    guards: list[Expression] = []
    for selector, left, right in permutations(OBSERVED_PLACEHOLDERS):
        branches = (
            (var(left), var(right)),
            (negate(var(left)), var(right)),
            (var(left), negate(var(right))),
            (negate(var(left)), negate(var(right))),
            (apply("xor", var(left), var(right)), apply("eq", var(left), var(right))),
        )
        for when_true, when_false in branches:
            guards.append(apply("ite", var(selector), when_true, when_false))
    return _deduplicate(guards)


_FAMILY_GUARDS = {
    "parity_guard": _parity_guards,
    "majority_guard": _majority_guards,
    "equality_chain": _chain_guards,
    "selector_multiplexer": _selector_guards,
}


def query_behavior_signature(
    guard: Mapping[str, Any], payload: Mapping[str, Any]
) -> tuple[int, ...]:
    """Return the Q-query behaviour per state: -1 to abstain, else the answer bit."""
    values = []
    for index in range(8):
        assignment = x_assignment(index)
        values.append(
            int(evaluate_expression(payload, assignment))
            if evaluate_expression(guard, assignment)
            else -1
        )
    return tuple(values)


def placeholder_program(
    family: str,
    guard: Mapping[str, Any],
    payload: Mapping[str, Any],
    nuisance_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical placeholder program of one guarded switch."""
    query_equality = apply("eq", var("Q"), payload)
    nuisance_equality = apply("eq", var("N"), nuisance_payload)
    if family == "parity_guard":
        constraint = apply(
            "and",
            apply("implies", guard, query_equality),
            apply("implies", negate(guard), nuisance_equality),
        )
    elif family == "majority_guard":
        constraint = apply(
            "and",
            apply("or", negate(guard), query_equality),
            apply("or", guard, nuisance_equality),
        )
    elif family == "equality_chain":
        constraint = apply(
            "and",
            apply("eq", apply("and", guard, var("Q")), apply("and", guard, payload)),
            apply(
                "eq",
                apply("and", negate(guard), var("N")),
                apply("and", negate(guard), nuisance_payload),
            ),
        )
    elif family == "selector_multiplexer":
        constraint = apply("ite", guard, query_equality, nuisance_equality)
    else:
        raise ValueError(f"unknown mechanism family {family!r}")
    return canonicalize_program(
        {
            "dsl_version": DSL_VERSION,
            "variables": list(PLACEHOLDER_VARIABLES),
            "constraints": [constraint],
        },
        variables=PLACEHOLDER_VARIABLES,
    )


def _placeholder_full_signature(program: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the accepted placeholder-assignment indices of a program."""
    return tuple(
        index
        for index in range(32)
        if program_accepts(program, index_assignment(index, PLACEHOLDER_VARIABLES))
    )


def balanced_strata(
    guard: Mapping[str, Any],
    payload: Mapping[str, Any],
    nuisance_payload: Mapping[str, Any],
) -> dict[str, tuple[tuple[int, int, int], ...]] | None:
    """Partition the eight states by guard and pinned value, or None if a stratum is empty."""
    active: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    inactive: dict[int, list[tuple[int, int, int]]] = {0: [], 1: []}
    for bits in product((0, 1), repeat=3):
        assignment = x_assignment(observed_state_index(bits))
        if evaluate_expression(guard, assignment):
            active[int(evaluate_expression(payload, assignment))].append(bits)
        else:
            inactive[int(evaluate_expression(nuisance_payload, assignment))].append(bits)
    if any(not active[value] or not inactive[value] for value in (0, 1)):
        return None
    strata = {
        "active_0": tuple(sorted(active[0], key=observed_state_index)),
        "inactive_0": tuple(sorted(inactive[0], key=observed_state_index)),
        "active_1": tuple(sorted(active[1], key=observed_state_index)),
        "inactive_1": tuple(sorted(inactive[1], key=observed_state_index)),
    }
    flattened = [bits for values in strata.values() for bits in values]
    if (
        len(flattened) != 8
        or len(set(flattened)) != 8
        or {observed_state_index(bits) for bits in flattened} != set(range(8))
        or max(map(len, strata.values())) > 5
    ):
        raise AssertionError("balanced strata must partition all eight X states")
    return strata


def balanced_quartet(
    guard: Mapping[str, Any],
    payload: Mapping[str, Any],
    nuisance_payload: Mapping[str, Any],
    *,
    route_index: int = 0,
) -> tuple[dict[str, Any], ...] | None:
    """Return one state per stratum for a route: active/inactive with values 0/1."""
    strata = balanced_strata(guard, payload, nuisance_payload)
    if strata is None:
        return None
    chosen = {name: members[route_index % len(members)] for name, members in strata.items()}

    def value(expression: Mapping[str, Any], bits: Sequence[int]) -> int:
        """Evaluate a payload at one observed state."""
        return int(evaluate_expression(expression, x_assignment(observed_state_index(bits))))

    return (
        {
            "observed_bits": chosen["active_0"],
            "role_zero_bit": 0,
            "role_one_bit": 0,
            "guard_active": True,
            "role_zero_payload": 0,
            "role_one_payload": value(nuisance_payload, chosen["active_0"]),
        },
        {
            "observed_bits": chosen["inactive_0"],
            "role_zero_bit": 0,
            "role_one_bit": 0,
            "guard_active": False,
            "role_zero_payload": value(payload, chosen["inactive_0"]),
            "role_one_payload": 0,
        },
        {
            "observed_bits": chosen["active_1"],
            "role_zero_bit": 1,
            "role_one_bit": 1,
            "guard_active": True,
            "role_zero_payload": 1,
            "role_one_payload": value(nuisance_payload, chosen["active_1"]),
        },
        {
            "observed_bits": chosen["inactive_1"],
            "role_zero_bit": 1,
            "role_one_bit": 1,
            "guard_active": False,
            "role_zero_payload": value(payload, chosen["inactive_1"]),
            "role_one_payload": 1,
        },
    )


def paired_behavior_signature(
    query_behavior: Sequence[int], nuisance_payload: Mapping[str, Any]
) -> tuple[int, ...]:
    """Return the 16-cell behaviour of both query roles over the eight states."""
    if len(query_behavior) != 8:
        raise ValueError("paired behavior requires exactly eight X cells per role")
    nuisance_truth = expression_truth(nuisance_payload)
    role_one = tuple(
        -1 if value >= 0 else nuisance_truth[index] for index, value in enumerate(query_behavior)
    )
    paired = tuple(query_behavior) + role_one
    if len(paired) != PAIRED_BEHAVIOR_LENGTH:
        raise AssertionError("paired behavior vector length drifted")
    return paired


def paired_behavior_hash(paired: Sequence[int]) -> str:
    """Hash a 16-cell paired behaviour as two abstract role vectors."""
    return object_hash({"abstract_role_0": list(paired[:8]), "abstract_role_1": list(paired[8:])})


@dataclass(frozen=True)
class MechanismTemplate:
    """One semantically unique five-placeholder mechanism."""

    template_id: str
    family: str
    partition: str
    route: str
    guard: dict[str, Any]
    query_payload: dict[str, Any]
    nuisance_payload: dict[str, Any]
    placeholder_program: dict[str, Any]
    template_hash: str
    full_semantic_hash: str
    query_behavior_hash: str
    query_behavior_signature: tuple[int, ...]
    ordinal: int

    @property
    def paired_behavior_hash(self) -> str:
        """Hash the template's paired behaviour over both query roles."""
        return paired_behavior_hash(
            paired_behavior_signature(self.query_behavior_signature, self.nuisance_payload)
        )

    def public_metadata(self) -> dict[str, Any]:
        """Return the role-free template metadata published in the manifests."""
        return {
            "template_id": self.template_id,
            "mechanism_family": self.family,
            "partition": self.partition,
            "route": self.route,
            "template_hash": self.template_hash,
            "placeholder_full_semantic_hash": self.full_semantic_hash,
            "paired_query_behavior_semantic_hash": self.paired_behavior_hash,
            "role_metadata_released": False,
        }

    def private_metadata(self) -> dict[str, Any]:
        """Return the evaluator-only template metadata, including the program."""
        return {
            **self.public_metadata(),
            "placeholder_variables": list(PLACEHOLDER_VARIABLES),
            "placeholder_program": self.placeholder_program,
            "role_zero_query_behavior_semantic_hash": self.query_behavior_hash,
        }


@dataclass(frozen=True)
class _Candidate:
    """One deduplicated template candidate with its packing signatures."""

    family: str
    guard: dict[str, Any]
    query_payload: dict[str, Any]
    nuisance_payload: dict[str, Any]
    placeholder_program: dict[str, Any]
    query_behavior_signature: tuple[int, ...]
    paired_behavior_signature: tuple[int, ...]
    full_signature: tuple[int, ...]
    template_hash: str
    full_semantic_hash: str
    candidate_hash: str


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of positions at which two equal-length vectors differ."""
    if len(left) != len(right):
        raise ValueError("Hamming vectors must have equal length")
    return sum(a != b for a, b in zip(left, right, strict=True))


def _family_candidates(family: str) -> Iterable[_Candidate]:
    """Yield the behaviour-unique, quartet-balanced candidates of one family."""
    seen_behavior: set[tuple[int, ...]] = set()
    guards = sorted(
        _FAMILY_GUARDS[family](),
        key=lambda item: object_hash(
            {"family": family, "guard": item, "truth": list(expression_truth(item))}
        ),
    )
    for guard in guards:
        if not 2 <= sum(expression_truth(guard)) <= 6:
            continue
        for payload, _ in payload_catalog():
            behavior = query_behavior_signature(guard, payload)
            if behavior in seen_behavior or {value for value in behavior if value >= 0} != {0, 1}:
                continue
            nuisance_options = [
                (expression, truth)
                for expression, truth in payload_catalog()
                if {truth[index] for index, value in enumerate(behavior) if value == -1} == {0, 1}
            ]
            if not nuisance_options:
                continue
            nuisance_payload = min(
                nuisance_options,
                key=lambda item: object_hash(
                    {"family": family, "behavior": list(behavior), "nuisance": item[0]}
                ),
            )[0]
            program = placeholder_program(family, guard, payload, nuisance_payload)
            full_signature = _placeholder_full_signature(program)
            if len(full_signature) != 16:
                raise AssertionError("a placeholder program must accept 16/32 assignments")
            if balanced_quartet(guard, payload, nuisance_payload) is None:
                continue
            seen_behavior.add(behavior)
            template_hash = object_hash(program)
            full_semantic_hash = object_hash(list(full_signature))
            paired = paired_behavior_signature(behavior, nuisance_payload)
            yield _Candidate(
                family=family,
                guard=guard,
                query_payload=payload,
                nuisance_payload=nuisance_payload,
                placeholder_program=program,
                query_behavior_signature=behavior,
                paired_behavior_signature=paired,
                full_signature=full_signature,
                template_hash=template_hash,
                full_semantic_hash=full_semantic_hash,
                candidate_hash=object_hash(
                    {
                        "family": family,
                        "template_hash": template_hash,
                        "placeholder_full_semantic_hash": full_semantic_hash,
                        "paired_query_behavior_semantic_hash": paired_behavior_hash(paired),
                    }
                ),
            )


def median(values: Sequence[int]) -> int | float:
    """Return the median, as an integer whenever the middle pair sums evenly."""
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    total = ordered[middle - 1] + ordered[middle]
    return total // 2 if total % 2 == 0 else total / 2


def _histogram(values: Iterable[int]) -> dict[str, int]:
    """Return a string-keyed histogram sorted by value."""
    return {str(key): count for key, count in sorted(Counter(values).items())}


def distance_summary(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]] | None = None
) -> dict[str, Any]:
    """Summarize Hamming distances within one set or across two sets."""
    if right is None:
        distances = [
            hamming_distance(left[index], left[other])
            for index in range(len(left))
            for other in range(index + 1, len(left))
        ]
        nearest = [
            min(
                hamming_distance(left[index], left[other])
                for other in range(len(left))
                if other != index
            )
            for index in range(len(left))
        ]
        comparison = "unordered_within_set"
    else:
        distances = [hamming_distance(a, b) for a in left for b in right]
        nearest = [min(hamming_distance(vector, other) for other in right) for vector in left]
        comparison = "all_cross_pairs_left_to_right"
    if not distances or not nearest:
        raise ValueError("distance summary requires nonempty comparison sets")
    return {
        "comparison": comparison,
        "pair_count": len(distances),
        "minimum": min(distances),
        "median": median(distances),
        "histogram": _histogram(distances),
        "nearest_neighbor_minimum": min(nearest),
        "nearest_neighbor_median": median(nearest),
        "nearest_neighbor_histogram": _histogram(nearest),
    }


def _full_vector(signature: Sequence[int]) -> tuple[int, ...]:
    """Return the 32-cell indicator vector of an accepted-index signature."""
    accepted = set(signature)
    return tuple(int(index in accepted) for index in range(32))


def _fixed_support(pools: Mapping[str, Sequence[_Candidate]]) -> tuple[_Candidate, ...]:
    """Return the twelve support candidates fixed by template hash."""
    support = []
    for family in MECHANISM_FAMILIES:
        by_hash = {candidate.template_hash: candidate for candidate in pools[family]}
        expected = FIXED_SUPPORT_TEMPLATE_HASHES[family]
        if any(value not in by_hash for value in expected):
            raise AssertionError(f"{family}: fixed support candidate disappeared")
        support.extend(by_hash[value] for value in expected)
    if len(support) != 12 or len({item.candidate_hash for item in support}) != 12:
        raise AssertionError("fixed support registry is not exactly 12 candidates")
    return tuple(support)


def _support_filtered(
    pools: Mapping[str, Sequence[_Candidate]], support: Sequence[_Candidate]
) -> dict[str, tuple[_Candidate, ...]]:
    """Drop candidates equal to, or within the packing distance of, a support template."""
    support_hashes = {item.candidate_hash for item in support}
    support_full = {item.full_signature for item in support}
    allowed = {}
    for family in MECHANISM_FAMILIES:
        kept = [
            candidate
            for candidate in pools[family]
            if candidate.candidate_hash not in support_hashes
            and candidate.full_signature not in support_full
            and all(
                hamming_distance(
                    candidate.paired_behavior_signature, item.paired_behavior_signature
                )
                >= PACKING_MINIMUM_DISTANCE
                for item in support
            )
        ]
        allowed[family] = tuple(sorted(kept, key=lambda item: item.candidate_hash))
    return allowed


def _conflicts(pools: Mapping[str, Sequence[_Candidate]]) -> tuple[dict[str, frozenset[str]], int]:
    """Return each candidate's too-close neighbours and the number of conflict edges."""
    candidates = sorted(
        (item for family in MECHANISM_FAMILIES for item in pools[family]),
        key=lambda item: item.candidate_hash,
    )
    conflicts: dict[str, set[str]] = {item.candidate_hash: set() for item in candidates}
    edges = 0
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            distance = hamming_distance(
                left.paired_behavior_signature, right.paired_behavior_signature
            )
            if distance < PACKING_MINIMUM_DISTANCE:
                conflicts[left.candidate_hash].add(right.candidate_hash)
                conflicts[right.candidate_hash].add(left.candidate_hash)
                edges += 1
    return {key: frozenset(values) for key, values in conflicts.items()}, edges


def _greedy_pack(
    pools: Mapping[str, Sequence[_Candidate]],
    conflicts: Mapping[str, frozenset[str]],
    *,
    family_order: Sequence[str],
    score_variant: str,
) -> tuple[tuple[_Candidate, ...], ...]:
    """Select whole one-per-family cycles greedily under a static score order."""
    by_hash = {item.candidate_hash: item for family in MECHANISM_FAMILIES for item in pools[family]}

    def score(candidate: _Candidate) -> tuple[Any, ...]:
        """Order candidates by conflict degree, then by hash."""
        neighbours = conflicts[candidate.candidate_hash]
        if score_variant == "global_degree_then_hash":
            return (len(neighbours), candidate.candidate_hash)
        if score_variant == "max_family_then_total_degree_then_hash":
            per_family = Counter(by_hash[value].family for value in neighbours)
            return (max(per_family.values(), default=0), len(neighbours), candidate.candidate_hash)
        raise ValueError(f"unknown packing score variant {score_variant!r}")

    ordered = {family: tuple(sorted(pools[family], key=score)) for family in MECHANISM_FAMILIES}
    selected_hashes: set[str] = set()
    selected_full: set[tuple[int, ...]] = set()
    cycles: list[tuple[_Candidate, ...]] = []
    while True:
        cycle: list[_Candidate] = []
        cycle_hashes: set[str] = set()
        cycle_full: set[tuple[int, ...]] = set()
        for family in family_order:
            blocked = selected_hashes | cycle_hashes
            chosen = next(
                (
                    candidate
                    for candidate in ordered[family]
                    if candidate.candidate_hash not in blocked
                    and candidate.full_signature not in (selected_full | cycle_full)
                    and not conflicts[candidate.candidate_hash].intersection(blocked)
                ),
                None,
            )
            if chosen is None:
                return tuple(cycles)
            cycle.append(chosen)
            cycle_hashes.add(chosen.candidate_hash)
            cycle_full.add(chosen.full_signature)
        cycles.append(tuple(cycle))
        selected_hashes.update(cycle_hashes)
        selected_full.update(cycle_full)


def _pack_evaluation(
    pools: Mapping[str, Sequence[_Candidate]],
) -> tuple[tuple[_Candidate, ...], tuple[_Candidate, ...], dict[str, Any]]:
    """Choose the support and evaluation candidates and return the packing audit."""
    pool_counts = {family: len(pools[family]) for family in MECHANISM_FAMILIES}
    if pool_counts != EXPECTED_CANDIDATE_COUNTS:
        raise AssertionError(f"template candidate pool counts drifted: {pool_counts}")
    all_candidates = [item for family in MECHANISM_FAMILIES for item in pools[family]]
    unique_behaviors = len({item.paired_behavior_signature for item in all_candidates})
    if unique_behaviors != EXPECTED_UNIQUE_PAIRED_BEHAVIORS:
        raise AssertionError("global paired-behavior candidate count drifted")
    support = _fixed_support(pools)
    allowed = _support_filtered(pools, support)
    allowed_counts = {family: len(allowed[family]) for family in MECHANISM_FAMILIES}
    if allowed_counts != EXPECTED_SUPPORT_FILTERED_COUNTS:
        raise AssertionError(f"support-filtered candidate counts drifted: {allowed_counts}")
    conflicts, conflict_edges = _conflicts(allowed)
    score_variants = ("global_degree_then_hash", "max_family_then_total_degree_then_hash")
    searches = []
    for score_variant in score_variants:
        for family_order in permutations(MECHANISM_FAMILIES):
            cycles = _greedy_pack(
                allowed, conflicts, family_order=family_order, score_variant=score_variant
            )
            searches.append(
                {
                    "score_variant": score_variant,
                    "family_order": tuple(family_order),
                    "cycles": cycles,
                    "complete_cycle_count": len(cycles),
                    "selected_hashes": tuple(
                        sorted(candidate.candidate_hash for cycle in cycles for candidate in cycle)
                    ),
                }
            )
    best = min(
        searches,
        key=lambda item: (
            -int(item["complete_cycle_count"]),
            item["selected_hashes"],
            item["score_variant"],
            item["family_order"],
        ),
    )
    achieved = int(best["complete_cycle_count"])
    if achieved < EVALUATION_TEMPLATES_PER_FAMILY:
        raise AssertionError(
            f"semantic packing achieved only {achieved} templates per family; "
            f"required {EVALUATION_TEMPLATES_PER_FAMILY}"
        )
    selected_by_family: dict[str, list[_Candidate]] = {family: [] for family in MECHANISM_FAMILIES}
    for cycle in best["cycles"]:
        for candidate in cycle:
            selected_by_family[candidate.family].append(candidate)
    evaluation = tuple(
        candidate
        for family in MECHANISM_FAMILIES
        for candidate in selected_by_family[family][:EVALUATION_TEMPLATES_PER_FAMILY]
    )
    evaluation_vectors = [item.paired_behavior_signature for item in evaluation]
    support_vectors = [item.paired_behavior_signature for item in support]
    evaluation_distance = distance_summary(evaluation_vectors)
    support_distance = distance_summary(evaluation_vectors, support_vectors)
    evaluation_full = [_full_vector(item.full_signature) for item in evaluation]
    support_full = [_full_vector(item.full_signature) for item in support]
    full_evaluation_distance = distance_summary(evaluation_full)
    full_support_distance = distance_summary(evaluation_full, support_full)
    if (
        min(
            evaluation_distance["minimum"],
            support_distance["minimum"],
            full_evaluation_distance["minimum"],
            full_support_distance["minimum"],
        )
        < PACKING_MINIMUM_DISTANCE
    ):
        raise AssertionError("selected template packing violates distance gates")
    if len({item.full_signature for item in (*support, *evaluation)}) != len(support) + len(
        evaluation
    ):
        raise AssertionError("selected full-class semantics are not unique")
    by_family = {}
    for family in MECHANISM_FAMILIES:
        family_vectors = [
            item.paired_behavior_signature for item in evaluation if item.family == family
        ]
        by_family[family] = {
            "evaluation_evaluation": distance_summary(family_vectors),
            "evaluation_support": distance_summary(family_vectors, support_vectors),
            "selected_candidate_hash": object_hash(
                [item.candidate_hash for item in evaluation if item.family == family]
            ),
        }
    audit = {
        "schema_version": TEMPLATE_PACKING_SCHEMA,
        "algorithm": {
            "version": TEMPLATE_PACKING_ALGORITHM,
            "dependency": "python_standard_library_only",
            "minimum_paired_behavior_hamming_distance": PACKING_MINIMUM_DISTANCE,
            "paired_behavior_vector_length": PAIRED_BEHAVIOR_LENGTH,
            "fixed_support": True,
            "family_orders_exhaustively_checked": 24,
            "static_score_variants": list(score_variants),
            "complete_cycles_only": True,
            "tie_break": "lexicographic_selected_candidate_hashes",
        },
        "candidate_pool": {
            "per_family": pool_counts,
            "family_specific_total": len(all_candidates),
            "globally_unique_paired_behaviors": unique_behaviors,
            "support_filtered_per_family": allowed_counts,
            "support_filtered_total": sum(allowed_counts.values()),
            "global_conflict_edges_below_threshold": conflict_edges,
            "pool_hash": object_hash(sorted(item.candidate_hash for item in all_candidates)),
            "support_filtered_pool_hash": object_hash(
                sorted(
                    item.candidate_hash for family in MECHANISM_FAMILIES for item in allowed[family]
                )
            ),
        },
        "search": {
            "runs": len(searches),
            "complete_cycle_count_histogram": _histogram(
                int(item["complete_cycle_count"]) for item in searches
            ),
            "chosen_score_variant": best["score_variant"],
            "chosen_family_order": list(best["family_order"]),
            "achieved_balanced_capacity_per_family": achieved,
            "required_selected_per_family": EVALUATION_TEMPLATES_PER_FAMILY,
            "unused_achieved_headroom_per_family": achieved - EVALUATION_TEMPLATES_PER_FAMILY,
            "balanced_solution_hash": object_hash(list(best["selected_hashes"])),
        },
        "selection": {
            "support_count": len(support),
            "evaluation_count": len(evaluation),
            "evaluation_per_family": {
                family: sum(item.family == family for item in evaluation)
                for family in MECHANISM_FAMILIES
            },
            "support_candidate_hash": object_hash([item.candidate_hash for item in support]),
            "evaluation_candidate_hash": object_hash([item.candidate_hash for item in evaluation]),
        },
        "paired_behavior_distances": {
            "evaluation_evaluation": evaluation_distance,
            "evaluation_support": support_distance,
            "per_evaluation_family": by_family,
        },
        "full_class_distances": {
            "derivation": (
                "paired-behavior distance >=4 implies placeholder-class "
                "truth-table Hamming distance >=4 for this guarded-switch family"
            ),
            "evaluation_evaluation": full_evaluation_distance,
            "evaluation_support": full_support_distance,
            "exact_semantics_unique": True,
        },
    }
    return support, evaluation, audit


@dataclass(frozen=True)
class TemplateRegistry:
    """The fixed support templates, the packed evaluation templates, and the packing audit."""

    support: tuple[MechanismTemplate, ...]
    evaluation: tuple[MechanismTemplate, ...]
    packing_audit: dict[str, Any]

    @property
    def all(self) -> tuple[MechanismTemplate, ...]:
        """Return support templates followed by evaluation templates."""
        return self.support + self.evaluation


@lru_cache(maxsize=1)
def template_registry() -> TemplateRegistry:
    """Build the deterministic template registry once per process."""
    pools = {family: tuple(_family_candidates(family)) for family in MECHANISM_FAMILIES}
    packed_support, packed_evaluation, packing_audit = _pack_evaluation(pools)
    templates: dict[str, list[MechanismTemplate]] = {"support": [], "evaluation": []}
    ordinal = 0
    for partition, candidates in (("support", packed_support), ("evaluation", packed_evaluation)):
        family_ordinals: Counter[str] = Counter()
        for candidate in candidates:
            family_index = family_ordinals[candidate.family]
            family_ordinals[candidate.family] += 1
            if partition == "support":
                route, label = "support", "support"
            elif candidate.family == HELDOUT_MECHANISM_FAMILY:
                route, label = "heldout_mechanism_family", "eval"
            else:
                route, label = "heldout_template", "eval"
            templates[partition].append(
                MechanismTemplate(
                    template_id=f"{candidate.family}-{label}-{family_index:02d}",
                    family=candidate.family,
                    partition=partition,
                    route=route,
                    guard=candidate.guard,
                    query_payload=candidate.query_payload,
                    nuisance_payload=candidate.nuisance_payload,
                    placeholder_program=candidate.placeholder_program,
                    template_hash=candidate.template_hash,
                    full_semantic_hash=candidate.full_semantic_hash,
                    query_behavior_hash=object_hash(list(candidate.query_behavior_signature)),
                    query_behavior_signature=candidate.query_behavior_signature,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    registry = TemplateRegistry(
        tuple(templates["support"]), tuple(templates["evaluation"]), packing_audit
    )
    expected = Counter({family: EVALUATION_TEMPLATES_PER_FAMILY for family in MECHANISM_FAMILIES})
    if Counter(item.family for item in registry.evaluation) != expected:
        raise AssertionError("evaluation template family counts are wrong")
    if len({item.full_semantic_hash for item in registry.all}) != len(registry.all):
        raise AssertionError("full placeholder semantics are not globally unique")
    if len({item.paired_behavior_hash for item in registry.all}) != len(registry.all):
        raise AssertionError("paired template behaviors are not globally unique")
    return registry
