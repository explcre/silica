#!/usr/bin/env python3
"""Independent clean-room oracle for the released SILICA dataset.

This program uses only the Python standard library and imports no SILICA
code.  From the released files alone it reparses the Boolean DSL, enumerates
every published program over the 2^8 declared universe, reconstructs the 16
survivors of every row and the support partition of its answers, and checks
each stored certificate, gold decision and payload by row ID.  It also
rebuilds the private template registry, its semantic-packing distances and
the route-state schedule, so agreement is not a second call to the
generator's oracle.

Usage: python audits/cleanroom_oracle.py --dataset DATASET [--output SUMMARY.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Frozen identifiers of the evaluated release, repeated here because this
# auditor must not import the package that defines them.
ROUTE_COVERAGE_SCHEMA = "silica-hard-v2-route-state-coverage-audit-v1"
ROUTE_CYCLING_ALGORITHM = "silica-hard-v2-balanced-strata-route-cycling-v1"

RELEASE_ROWS = 12288
FORBIDDEN_PUBLIC_KEYS = {
    "seed",
    "target_hypothesis_id",
    "surface_inventory",
    "gold",
    "certificate",
    "support",
    "oracle",
}
PLACEHOLDER_VARIABLES = ("Q", "N", "X0", "X1", "X2")
MECHANISM_FAMILIES = (
    "parity_guard",
    "majority_guard",
    "equality_chain",
    "selector_multiplexer",
)
EXPECTED_SUPPORT_TEMPLATE_HASHES = {
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
PACKING_MINIMUM_DISTANCE = 4
AXIS_TO_FEATURE = {
    "noun_pair_01": "noun_swap_01",
    "noun_pair_23": "noun_swap_23",
    "verb_pair_01": "verb_swap_01",
    "modifier_pair_01": "modifier_swap_01",
}
QUERY_FIELDS = {
    "id",
    "complete_unit_id",
    "task",
    "item_index",
    "query_id",
    "counterfactual_cell_id",
    "query",
}


def canonical_json(value: Any) -> str:
    """Return sorted-key, compact, UTF-8 JSON without NaN."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def object_hash(value: Any) -> str:
    """Return the SHA-256 of a value's canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read canonical JSON lines, rejecting any noncanonical line."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict) or line != canonical_json(value) + "\n":
                raise ValueError(f"{path}:{line_number}: noncanonical JSONL")
            rows.append(value)
    return rows


def forbidden_paths(value: Any, path: str = "$") -> list[str]:
    """Return the paths of evaluator-only keys inside a public value."""
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in FORBIDDEN_PUBLIC_KEYS:
                found.append(child_path)
            found.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{path}[{index}]"))
    return found


def evaluate(expression: Mapping[str, Any], assignment: Mapping[str, bool]) -> bool:
    """Evaluate one Boolean-DSL expression."""
    operation = expression.get("op")
    if operation == "var":
        return bool(assignment[str(expression["name"])])
    if operation == "const":
        return bool(expression["value"])
    if operation == "not":
        return not evaluate(expression["arg"], assignment)
    arguments = [evaluate(item, assignment) for item in expression["args"]]
    if operation == "and":
        return all(arguments)
    if operation == "or":
        return any(arguments)
    if operation == "xor":
        return sum(arguments) % 2 == 1
    if operation == "eq":
        return arguments[0] == arguments[1]
    if operation == "implies":
        return (not arguments[0]) or arguments[1]
    if operation == "ite":
        return arguments[1] if arguments[0] else arguments[2]
    raise ValueError(f"unknown Boolean operation {operation!r}")


def program_accepts(program: Mapping[str, Any], assignment: Mapping[str, bool]) -> bool:
    """Return whether every constraint of a program holds."""
    return all(evaluate(constraint, assignment) for constraint in program["constraints"])


def assignment(index: int, features: Sequence[str]) -> dict[str, bool]:
    """Assign named bits from an index, least significant feature first."""
    return {feature: bool((index >> offset) & 1) for offset, feature in enumerate(features)}


def hamming_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the number of differing positions of two equal-length vectors."""
    if len(left) != len(right):
        raise ValueError("Hamming vectors must have equal length")
    return sum(a != b for a, b in zip(left, right, strict=True))


def median(values: Sequence[int]) -> int | float:
    """Return the median, integral whenever the middle pair sums evenly."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("median requires values")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    total = ordered[middle - 1] + ordered[middle]
    return total // 2 if total % 2 == 0 else total / 2


def histogram(values: Sequence[int]) -> dict[str, int]:
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
        nearest = [min(hamming_distance(a, b) for b in right) for a in left]
        comparison = "all_cross_pairs_left_to_right"
    return {
        "comparison": comparison,
        "pair_count": len(distances),
        "minimum": min(distances),
        "median": median(distances),
        "histogram": histogram(distances),
        "nearest_neighbor_minimum": min(nearest),
        "nearest_neighbor_median": median(nearest),
        "nearest_neighbor_histogram": histogram(nearest),
    }


def abstract_roles(accepted: Sequence[int]) -> dict[int, list[int]]:
    """Return each role's pinned value per observed state, or -1 where it is free."""
    accepted_set = set(accepted)
    abstract: dict[int, list[int]] = {0: [], 1: []}
    for x_index in range(8):
        values_by_role: dict[int, set[int]] = {0: set(), 1: set()}
        for role_zero in (0, 1):
            for role_one in (0, 1):
                if role_zero | (role_one << 1) | (x_index << 2) in accepted_set:
                    values_by_role[0].add(role_zero)
                    values_by_role[1].add(role_one)
        for role in (0, 1):
            values = values_by_role[role]
            abstract[role].append(next(iter(values)) if len(values) == 1 else -1)
    return abstract


def placeholder_semantics(program: Mapping[str, Any]) -> dict[str, Any]:
    """Return the accepted assignments and role behaviours of a placeholder program."""
    accepted = [
        index
        for index in range(32)
        if program_accepts(program, assignment(index, PLACEHOLDER_VARIABLES))
    ]
    abstract = abstract_roles(accepted)
    return {
        "accepted": accepted,
        "full_vector": tuple(int(index in set(accepted)) for index in range(32)),
        "paired_vector": tuple(abstract[0] + abstract[1]),
        "paired_hash": object_hash(
            {"abstract_role_0": abstract[0], "abstract_role_1": abstract[1]}
        ),
        "role_zero_hash": object_hash(abstract[0]),
    }


def audit_template_packing(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the private template registry and its semantic-packing distances."""
    registry = manifest.get("private_template_registry")
    if not isinstance(registry, list) or len(registry) != 204:
        raise ValueError("the private template registry must hold 12 support + 192 templates")
    reconstructed = []
    for item in registry:
        program = item["placeholder_program"]
        semantic = placeholder_semantics(program)
        family = str(item["mechanism_family"])
        if (
            object_hash(program) != item["template_hash"]
            or len(semantic["accepted"]) != 16
            or object_hash(semantic["accepted"]) != item["placeholder_full_semantic_hash"]
            or semantic["paired_hash"] != item["paired_query_behavior_semantic_hash"]
            or semantic["role_zero_hash"] != item["role_zero_query_behavior_semantic_hash"]
            or family not in MECHANISM_FAMILIES
        ):
            raise ValueError(f"{item.get('template_id')}: private template semantic mismatch")
        candidate_hash = object_hash(
            {
                "family": family,
                "template_hash": item["template_hash"],
                "placeholder_full_semantic_hash": item["placeholder_full_semantic_hash"],
                "paired_query_behavior_semantic_hash": item["paired_query_behavior_semantic_hash"],
            }
        )
        reconstructed.append({**item, **semantic, "candidate_hash": candidate_hash})
    support = [item for item in reconstructed if item["partition"] == "support"]
    evaluation = [item for item in reconstructed if item["partition"] == "evaluation"]
    if len(support) != 12 or len(evaluation) != 192:
        raise ValueError("template partition scale mismatch")
    family_counts = Counter(str(item["mechanism_family"]) for item in evaluation)
    if family_counts != Counter({family: 48 for family in MECHANISM_FAMILIES}):
        raise ValueError("evaluation template family balance mismatch")
    support_hashes = {
        family: tuple(
            str(item["template_hash"]) for item in support if item["mechanism_family"] == family
        )
        for family in MECHANISM_FAMILIES
    }
    if support_hashes != EXPECTED_SUPPORT_TEMPLATE_HASHES:
        raise ValueError("fixed support registry drifted")
    if (
        len({tuple(item["paired_vector"]) for item in reconstructed}) != 204
        or len({tuple(item["accepted"]) for item in reconstructed}) != 204
    ):
        raise ValueError("selected template semantics are not exactly unique")
    paired_eval = [item["paired_vector"] for item in evaluation]
    paired_support = [item["paired_vector"] for item in support]
    full_eval = [item["full_vector"] for item in evaluation]
    full_support = [item["full_vector"] for item in support]
    paired_eval_distance = distance_summary(paired_eval)
    paired_support_distance = distance_summary(paired_eval, paired_support)
    full_eval_distance = distance_summary(full_eval)
    full_support_distance = distance_summary(full_eval, full_support)
    if (
        min(
            paired_eval_distance["minimum"],
            paired_support_distance["minimum"],
            full_eval_distance["minimum"],
            full_support_distance["minimum"],
        )
        < PACKING_MINIMUM_DISTANCE
    ):
        raise ValueError("template semantic-packing distance gate failed")
    declared = manifest["acceptance_audit"]["template_semantic_packing"]
    if (
        declared["paired_behavior_distances"]["evaluation_evaluation"] != paired_eval_distance
        or declared["paired_behavior_distances"]["evaluation_support"] != paired_support_distance
        or declared["full_class_distances"]["evaluation_evaluation"] != full_eval_distance
        or declared["full_class_distances"]["evaluation_support"] != full_support_distance
        or declared["selection"]["support_candidate_hash"]
        != object_hash([item["candidate_hash"] for item in support])
        or declared["selection"]["evaluation_candidate_hash"]
        != object_hash([item["candidate_hash"] for item in evaluation])
    ):
        raise ValueError("the declared template-packing audit does not reconstruct")
    return {
        "support_templates": len(support),
        "evaluation_templates": len(evaluation),
        "evaluation_per_family": dict(sorted(family_counts.items())),
        "minimum_paired_evaluation_distance": paired_eval_distance["minimum"],
        "minimum_paired_support_distance": paired_support_distance["minimum"],
        "minimum_full_evaluation_distance": full_eval_distance["minimum"],
        "minimum_full_support_distance": full_support_distance["minimum"],
        "evaluation_candidate_hash": object_hash([item["candidate_hash"] for item in evaluation]),
    }


def canonicalize_expression(expression: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical form of an expression: commutative arguments sorted."""
    operation = str(expression["op"])
    if operation == "var":
        return {"op": "var", "name": str(expression["name"])}
    if operation == "const":
        return {"op": "const", "value": bool(expression["value"])}
    if operation == "not":
        return {"op": "not", "arg": canonicalize_expression(expression["arg"])}
    arguments = [canonicalize_expression(item) for item in expression["args"]]
    if operation in {"and", "or", "xor", "eq"}:
        arguments.sort(key=canonical_json)
    return {"op": operation, "args": arguments}


def inverse_placeholder_program(
    program: Mapping[str, Any], route: Mapping[str, Any]
) -> dict[str, Any]:
    """Map a published program back onto the five placeholders of its route."""
    mapping = {
        str(route["role_zero_feature"]): "Q",
        str(route["role_one_feature"]): "N",
        **{
            str(feature): f"X{index}"
            for index, feature in enumerate(route["ordered_evidence_features"])
        },
    }

    def substitute(expression: Mapping[str, Any]) -> dict[str, Any]:
        """Rename the variables of one expression."""
        if expression["op"] == "var":
            return {"op": "var", "name": mapping[str(expression["name"])]}
        if expression["op"] == "const":
            return {"op": "const", "value": bool(expression["value"])}
        if expression["op"] == "not":
            return {"op": "not", "arg": substitute(expression["arg"])}
        return {
            "op": str(expression["op"]),
            "args": [substitute(item) for item in expression["args"]],
        }

    constraints = [canonicalize_expression(substitute(item)) for item in program["constraints"]]
    constraints.sort(key=canonical_json)
    return {
        "dsl_version": program["dsl_version"],
        "variables": list(PLACEHOLDER_VARIABLES),
        "constraints": constraints,
    }


def feature_values(index: int, declaration: Mapping[str, Any]) -> dict[str, str]:
    """Return the value labels of a hypothesis index, including word order."""
    values = {
        str(item["name"]): str(item["values"][(index >> offset) & 1])
        for offset, item in enumerate(declaration["feature_variables_lsb_to_msb"])
    }
    values["word_order"] = (("SVO", "SOV"), ("VSO", "VOS"))[(index >> 4) & 1][(index >> 5) & 1]
    return values


def hypothesis_dict(index: int, declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize one hypothesis exactly as the declared class hash commits to it."""
    return {
        "id": f"h{index:03d}",
        "bits": [(index >> offset) & 1 for offset in range(8)],
        "features": feature_values(index, declaration),
    }


def observed_bits(
    evidence: Sequence[Mapping[str, Any]], declaration: Mapping[str, Any]
) -> dict[str, int]:
    """Return the swap bit each observation reveals, keyed by feature."""
    result = {}
    for item in evidence:
        if set(item) != {"id", "channel", "category", "semantic", "form"}:
            raise ValueError("unexpected evidence schema")
        payload = {key: item[key] for key in ("channel", "category", "semantic", "form")}
        if item["id"] != "e-d-" + object_hash(payload)[:16]:
            raise ValueError("evidence ID mismatch")
        candidates = [
            axis
            for axis in declaration["lexical_axes"]
            if axis["category"] == item["category"]
            and item["semantic"] in axis["possible_meanings"]
        ]
        if len(candidates) != 1:
            raise ValueError("evidence does not identify one lexical axis")
        axis = candidates[0]
        bit = axis["possible_meanings"].index(item["semantic"]) ^ axis["forms"].index(item["form"])
        result[AXIS_TO_FEATURE[str(axis["name"])]] = bit
    if len(result) != 3:
        raise ValueError("expected three distinct observed features")
    return result


def query_answer(index: int, public: Mapping[str, Any], declaration: Mapping[str, Any]) -> Any:
    """Return the canonical answer of a row's query in one hypothesis."""
    names = [str(item["name"]) for item in declaration["feature_variables_lsb_to_msb"]]
    if public["task"] == "grammar_feature":
        feature = str(public["query"]["feature"])
        offset = names.index(feature)
        return declaration["feature_variables_lsb_to_msb"][offset]["values"][(index >> offset) & 1]
    if public["task"] == "lexicon":
        category, form = public["query"]["category"], public["query"]["form"]
        axes = [
            axis
            for axis in declaration["lexical_axes"]
            if axis["category"] == category and form in axis["forms"]
        ]
        if len(axes) != 1:
            raise ValueError("lexicon query does not identify one axis")
        bit = (index >> names.index(AXIS_TO_FEATURE[str(axes[0]["name"])])) & 1
        return axes[0]["possible_meanings"][axes[0]["forms"].index(form) ^ bit]
    raise ValueError(f"unsupported task {public['task']!r}")


def exact_program_metadata(declaration: Mapping[str, Any]) -> tuple[list[int], str, str]:
    """Enumerate a published program and recompute its program and class hashes."""
    program = declaration["boolean_program"]
    features = [str(item["name"]) for item in declaration["feature_variables_lsb_to_msb"]]
    accepted = [
        index for index in range(256) if program_accepts(program, assignment(index, features))
    ]
    program_hash = object_hash(program)
    class_hash = object_hash(
        {
            "dsl_version": program["dsl_version"],
            "program_hash": program_hash,
            "hypotheses": [hypothesis_dict(index, declaration) for index in accepted],
        }
    )
    return accepted, program_hash, class_hash


def exact_certificate(
    public: Mapping[str, Any], accepted: Sequence[int], declaration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute a row's certificate and gold decision from public data only."""
    observed = observed_bits(public["evidence"], declaration)
    names = [str(item["name"]) for item in declaration["feature_variables_lsb_to_msb"]]
    offsets = {feature: names.index(feature) for feature in observed}
    survivors = [
        index
        for index in accepted
        if all(((index >> offsets[feature]) & 1) == bit for feature, bit in observed.items())
    ]
    groups: dict[str, list[int]] = defaultdict(list)
    answers = {}
    for index in survivors:
        answer = query_answer(index, public, declaration)
        groups[canonical_json(answer)].append(index)
        answers[canonical_json(answer)] = answer
    support = [
        {
            "answer": answers[key],
            "count": len(groups[key]),
            "hypothesis_ids_hash": object_hash([f"h{index:03d}" for index in groups[key]]),
        }
        for key in sorted(groups)
    ]
    consensus = len(groups) == 1
    answer = support[0]["answer"] if consensus else None
    witness = None
    if not consensus:
        first, second = sorted(groups)[:2]
        witness = [
            {"hypothesis_id": f"h{groups[first][0]:03d}", "answer": answers[first]},
            {"hypothesis_id": f"h{groups[second][0]:03d}", "answer": answers[second]},
        ]
    experiment_keys = [
        object_hash(
            {"channel": item["channel"], "category": item["category"], "semantic": item["semantic"]}
        )
        for item in public["evidence"]
    ]
    certificate = {
        "hypothesis_class_size": len(accepted),
        "hypothesis_class_hash": declaration["class_hash"],
        "class_semantic_hash": object_hash(list(accepted)),
        "boolean_program_hash": object_hash(declaration["boolean_program"]),
        "evidence_hash": object_hash(sorted(public["evidence"], key=lambda item: item["id"])),
        "ordered_experiment_hash": object_hash(experiment_keys),
        "query_hash": object_hash({"task": public["task"], "query": public["query"]}),
        "hypothesis_count": len(survivors),
        "version_space_hash": object_hash([f"h{index:03d}" for index in survivors]),
        "support": support,
        "support_hash": object_hash(support),
        "consensus": consensus,
        "decision": "ANSWER" if consensus else "ABSTAIN",
        "answer_hash": object_hash(answer),
        "disagreement_witness": witness,
    }
    return certificate, {"decision": certificate["decision"], "answer": answer}


def paired_semantics(declaration: Mapping[str, Any], route: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a route's template and paired-behaviour hashes from its published program."""
    features = [str(item["name"]) for item in declaration["feature_variables_lsb_to_msb"]]
    placeholder = inverse_placeholder_program(declaration["boolean_program"], route)
    accepted_placeholder = [
        index
        for index in range(32)
        if program_accepts(placeholder, assignment(index, PLACEHOLDER_VARIABLES))
    ]
    abstract = abstract_roles(accepted_placeholder)
    offsets = [features.index(feature) for feature in route["ordered_evidence_features"]]
    signatures = {}
    for role, feature in ((0, route["role_zero_feature"]), (1, route["role_one_feature"])):
        signatures[str(feature)] = [
            abstract[role][
                sum(((index >> offset) & 1) << position for position, offset in enumerate(offsets))
            ]
            for index in range(256)
        ]
    queried = sorted(signatures)
    return {
        "template_hash": object_hash(placeholder),
        "placeholder_full_semantic_hash": object_hash(accepted_placeholder),
        "paired_template_query_behavior_semantic_hash": object_hash(
            {"abstract_role_0": abstract[0], "abstract_role_1": abstract[1]}
        ),
        "paired_query_behavior_semantic_hash": object_hash(
            {feature: signatures[feature] for feature in queried}
        ),
        "logical_unit_semantic_hash": object_hash(
            {
                "class_semantic_hash": declaration["class_semantic_hash"],
                "queried_features_unordered": queried,
            }
        ),
    }


def audit_route_coverage(
    public_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    route_by_class: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Rebuild the balanced route-state schedule and compare it with the manifest."""
    registry = {
        str(item["template_id"]): item
        for item in manifest["private_template_registry"]
        if item["partition"] == "evaluation"
    }
    vectors = {
        template_id: placeholder_semantics(item["placeholder_program"])["paired_vector"]
        for template_id, item in registry.items()
    }
    paired: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in public_rows:
        paired[str(row["paired_query_unit_id"])].append(row)
    template_routes: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    route_details = []
    for paired_id, rows in sorted(paired.items()):
        template_ids = {str(row["mechanism"]["template_id"]) for row in rows}
        class_hashes = {str(row["mechanism"]["class_semantic_hash"]) for row in rows}
        if len(template_ids) != 1 or len(class_hashes) != 1:
            raise ValueError(f"{paired_id}: route spans multiple templates or classes")
        template_id = next(iter(template_ids))
        route = route_by_class[next(iter(class_hashes))]
        axis_pair_id = str(route["axis_pair_id"])
        if not axis_pair_id.startswith("pair-") or int(axis_pair_id[5:]) not in range(8):
            raise ValueError(f"{paired_id}: malformed axis-pair route")
        pair_index = int(axis_pair_id[5:])
        states = {}
        for row in rows:
            by_feature = observed_bits(row["evidence"], row["hypothesis_declaration"])
            bits = tuple(
                int(by_feature[str(value)]) for value in route["ordered_evidence_features"]
            )
            x_index = bits[0] + 2 * bits[1] + 4 * bits[2]
            role_zero = vectors[template_id][x_index]
            role_one = vectors[template_id][8 + x_index]
            if role_zero in (0, 1) and role_one == -1:
                stratum = f"active_{role_zero}"
            elif role_zero == -1 and role_one in (0, 1):
                stratum = f"inactive_{role_one}"
            else:
                raise ValueError(f"{template_id}: invalid guarded paired vector")
            if states.setdefault(str(row["paired_state_id"]), (stratum, x_index)) != (
                stratum,
                x_index,
            ):
                raise ValueError(f"{paired_id}: a public state disagrees")
        selected = dict(states.values())
        if (
            len(states) != 4
            or set(selected) != {"active_0", "inactive_0", "active_1", "inactive_1"}
            or len(set(selected.values())) != 4
            or pair_index in template_routes[template_id]
        ):
            raise ValueError(f"{paired_id}: invalid route-state selection")
        template_routes[template_id][pair_index] = selected
        route_details.append((template_id, pair_index, tuple(sorted(selected.items()))))

    diagnostics = []
    stratum_sizes: list[int] = []
    spreads: list[int] = []
    union_sizes: list[int] = []
    for template_id, routes in sorted(template_routes.items()):
        if set(routes) != set(range(8)):
            raise ValueError(f"{template_id}: incomplete 8-route coverage")
        vector = vectors[template_id]
        strata = {
            "active_0": tuple(index for index in range(8) if vector[index] == 0),
            "inactive_0": tuple(index for index in range(8) if vector[8 + index] == 0),
            "active_1": tuple(index for index in range(8) if vector[index] == 1),
            "inactive_1": tuple(index for index in range(8) if vector[8 + index] == 1),
        }
        if any(not members or len(members) > 5 for members in strata.values()):
            raise ValueError(f"{template_id}: invalid route stratum sizes")
        counts: dict[str, Counter[int]] = {stratum: Counter() for stratum in strata}
        observed_union = set()
        for pair_index in range(8):
            for stratum, members in strata.items():
                expected = members[pair_index % len(members)]
                if routes[pair_index][stratum] != expected:
                    raise ValueError(f"{template_id}: route-cycling selection drift")
                counts[stratum][expected] += 1
                observed_union.add(expected)
        if observed_union != set(range(8)):
            raise ValueError(f"{template_id}: route union omits an X state")
        union_sizes.append(len(observed_union))
        stratum_diagnostics = {}
        for stratum, members in strata.items():
            multiplicities = [counts[stratum][member] for member in members]
            if sum(multiplicities) != 8 or max(multiplicities) - min(multiplicities) > 1:
                raise ValueError(f"{template_id}/{stratum}: selection imbalance")
            stratum_sizes.append(len(members))
            spreads.append(max(multiplicities) - min(multiplicities))
            stratum_diagnostics[stratum] = {
                "member_count": len(members),
                "minimum_selection_multiplicity": min(multiplicities),
                "maximum_selection_multiplicity": max(multiplicities),
                "selection_count": sum(multiplicities),
            }
        diagnostics.append(
            {
                "template_id": template_id,
                "family": registry[template_id]["mechanism_family"],
                "route_count": len(routes),
                "observed_state_union": sorted(observed_union),
                "strata": stratum_diagnostics,
            }
        )
    if set(template_routes) != set(registry):
        raise ValueError("route coverage omits an evaluation template")
    reconstructed = {
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
        "observed_state_union_size_histogram": histogram(union_sizes),
        "stratum_member_count_histogram": histogram(stratum_sizes),
        "per_member_selection_spread_histogram": histogram(spreads),
        "route_selection_hash": object_hash(route_details),
        "template_diagnostics_hash": object_hash(diagnostics),
    }
    if reconstructed != manifest["acceptance_audit"]["route_state_coverage"]:
        raise ValueError("the declared route-state coverage audit does not reconstruct")
    return reconstructed


def audit(dataset: Path) -> dict[str, Any]:
    """Run every clean-room check and return the summary."""
    manifest = load_json(dataset / "manifest.json")
    packing = audit_template_packing(manifest)
    if forbidden_paths(load_json(dataset / "public" / "manifest.json")):
        raise ValueError("the public manifest leaks evaluator-only fields")
    for entry in [*manifest["public_files"], *manifest["gold_files"]]:
        if file_sha256(dataset / entry["path"]) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {entry['path']}")
    public_rows = [
        row for entry in manifest["public_files"] for row in load_jsonl(dataset / entry["path"])
    ]
    gold_rows = [
        row for entry in manifest["gold_files"] for row in load_jsonl(dataset / entry["path"])
    ]
    gold_by_id = {str(item["id"]): item for item in gold_rows}
    if (
        len(public_rows) != RELEASE_ROWS
        or len(gold_rows) != RELEASE_ROWS
        or len(gold_by_id) != RELEASE_ROWS
        or {str(item["id"]) for item in public_rows} != set(gold_by_id)
    ):
        raise ValueError("public/gold scale or ID join mismatch")
    route_by_class = {
        str(item["class_semantic_hash"]): item
        for item in manifest["private_instantiation_registry"]
    }
    if len(route_by_class) != 1632:
        raise ValueError("private registry class hashes are not unique")

    program_cache: dict[str, list[int]] = {}
    semantic_cache: dict[str, dict[str, Any]] = {}
    decision_counts: Counter[str] = Counter()
    posterior_counts: list[int] = []
    class_sizes: list[int] = []
    exact_rows = 0
    for public in public_rows:
        identifier = str(public["id"])
        private = gold_by_id[identifier]
        if forbidden_paths(public):
            raise ValueError(f"{identifier}: public leakage")
        if (
            identifier
            != "silica-" + object_hash({k: v for k, v in public.items() if k != "id"})[:24]
        ):
            raise ValueError(f"{identifier}: public ID mismatch")
        declaration = public["hypothesis_declaration"]
        program_hash = str(declaration["boolean_program_hash"])
        if program_hash not in program_cache:
            accepted, recomputed_hash, class_hash = exact_program_metadata(declaration)
            if (
                recomputed_hash != program_hash
                or class_hash != declaration["class_hash"]
                or len(accepted) != 128
                or object_hash(accepted) != declaration["class_semantic_hash"]
            ):
                raise ValueError(f"{identifier}: class integrity mismatch")
            program_cache[program_hash] = accepted
        accepted = program_cache[program_hash]
        certificate, gold = exact_certificate(public, accepted, declaration)
        if certificate != private["certificate"]:
            raise ValueError(f"{identifier}: certificate mismatch")
        if {"decision": private["gold"]["decision"], "answer": private["gold"]["answer"]} != gold:
            raise ValueError(f"{identifier}: gold mismatch")
        if private["oracle"]["public_record_hash"] != object_hash(public):
            raise ValueError(f"{identifier}: public/gold binding mismatch")
        mechanism = public["mechanism"]
        class_hash = str(mechanism["class_semantic_hash"])
        if class_hash not in route_by_class:
            raise ValueError(f"{identifier}: missing private semantic route")
        if class_hash not in semantic_cache:
            semantic_cache[class_hash] = paired_semantics(declaration, route_by_class[class_hash])
        for key, value in semantic_cache[class_hash].items():
            if mechanism.get(key) != value:
                raise ValueError(f"{identifier}: semantic hash mismatch for {key}")
        decision_counts[gold["decision"]] += 1
        posterior_counts.append(certificate["hypothesis_count"])
        class_sizes.append(len(accepted))
        exact_rows += 1

    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    complete: dict[str, list[dict[str, Any]]] = defaultdict(list)
    states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in public_rows:
        paired[str(row["paired_query_unit_id"])].append(row)
        complete[str(row["complete_unit_id"])].append(row)
        states[str(row["paired_state_id"])].append(row)
    if (
        Counter(map(len, paired.values())) != Counter({8: 1536})
        or Counter(map(len, complete.values())) != Counter({4: 3072})
        or Counter(map(len, states.values())) != Counter({2: 6144})
    ):
        raise ValueError("paired/evaluator/state routing sizes mismatch")
    query_mask_correct = 0
    for state_rows in states.values():
        decisions = Counter(gold_by_id[str(row["id"])]["gold"]["decision"] for row in state_rows)
        shared = {
            canonical_json({key: value for key, value in row.items() if key not in QUERY_FIELDS})
            for row in state_rows
        }
        if decisions != Counter({"ANSWER": 1, "ABSTAIN": 1}) or len(shared) != 1:
            raise ValueError("query-masked raw-state pairing mismatch")
        query_mask_correct += 1

    def truth_digest(rows: Sequence[Mapping[str, Any]]) -> str:
        """Hash the ID-joined truth of a row order."""
        return object_hash(
            sorted(
                (str(row["id"]), str(gold_by_id[str(row["id"])]["gold"]["decision"]))
                for row in rows
            )
        )

    reverse_agreement = truth_digest(public_rows) == truth_digest(list(reversed(public_rows)))
    if not reverse_agreement:
        raise ValueError("row permutation changes ID-joined truth")
    return {
        "schema_version": "silica-cleanroom-oracle-v1",
        "implementation": "python_standard_library_only_no_silica_import",
        "rows_checked": exact_rows,
        "programs_checked": len(program_cache),
        "paired_semantics_checked": len(semantic_cache),
        "template_semantic_packing": packing,
        "route_state_coverage": audit_route_coverage(public_rows, manifest, route_by_class),
        "decision_counts": dict(sorted(decision_counts.items())),
        "declared_class_size_histogram": histogram(class_sizes),
        "posterior_size_histogram": histogram(posterior_counts),
        "paired_query_groups": len(paired),
        "four_cell_evaluator_units": len(complete),
        "same_state_query_pairs": len(states),
        "query_masked_raw_state_probe": {
            "correct": query_mask_correct,
            "total": len(public_rows),
            "balanced_decision_accuracy": query_mask_correct / len(public_rows),
            "exactly_one_half": 2 * query_mask_correct == len(public_rows),
        },
        "oracle_joint_accuracy": {
            "correct": exact_rows,
            "total": len(public_rows),
            "accuracy": exact_rows / len(public_rows),
        },
        "reverse_order_agreement": reverse_agreement,
        "dataset_sha256": manifest["dataset_sha256"],
        "public_file_sha256": {item["path"]: item["sha256"] for item in manifest["public_files"]},
        "gold_file_sha256": {item["path"]: item["sha256"] for item in manifest["gold_files"]},
    }


def main(argv: list[str] | None = None) -> int:
    """Run the clean-room oracle and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    serialized = canonical_json(audit(args.dataset)) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
