"""Instantiation routes: binding template placeholders to SILICA features.

Each evaluation template is instantiated on eight query-axis pairs, one
lexical and one grammar feature.  For every pair, the two query-facing
placeholders take the two queried features in one of two orientations and the
three observed placeholders take three of the remaining lexical features in
one of six orders.  A deterministic augmenting-path matching picks, per axis
pair, a mapping for every template so that no two instantiated programs share
the same paired query behaviour over all 256 languages.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any

from ._frozen import (
    INSTANTIATION_PROTOCOL_SCHEMA,
    LIFECYCLE,
    ROUTE_CYCLING_ALGORITHM,
    ROUTE_PROTOCOL_SCHEMA,
)
from .declaration import CLASS_SEMANTIC_HASH_DEFINITION
from .dsl import (
    DSL_VERSION,
    accepted_indices,
    canonicalize_expression,
    canonicalize_program,
)
from .hypotheses import FEATURE_NAMES, GRAMMAR_FEATURES, LEXICAL_FEATURES
from .templates import (
    OBSERVED_PLACEHOLDERS,
    PLACEHOLDER_FULL_SEMANTIC_HASH_DEFINITION,
    QUERY_BEHAVIOR_HASH_DEFINITION,
    STRATA,
    MechanismTemplate,
    apply,
    const,
    expression_truth,
    negate,
    template_registry,
    var,
)
from .util import object_hash

STATES_PER_UNIT = 4
QUERY_AXIS_PAIRS = tuple(
    (
        f"pair-{index:02d}",
        LEXICAL_FEATURES[index % 4],
        GRAMMAR_FEATURES[(index % 4 + index // 4) % 4],
    )
    for index in range(8)
)
MAPPED_QUERY_BEHAVIOR_HASH_DEFINITION = {
    "hash": "sha256_of_canonical_json",
    "payload": (
        "query behavior at all 256 SILICA assignments in hypothesis-ID order; "
        "each value is -1 (ABSTAIN), 0 (ANSWER:0), or 1 (ANSWER:1)"
    ),
}

RouteMapping = tuple[str, str, tuple[str, str, str]]
Signature = tuple[tuple[str, tuple[int, ...]], ...]


def substitute(expression: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    """Rename the variables of an expression."""
    operation = str(expression["op"])
    if operation == "var":
        return var(mapping[str(expression["name"])])
    if operation == "const":
        return const(bool(expression["value"]))
    if operation == "not":
        return negate(substitute(expression["arg"], mapping))
    return apply(operation, *(substitute(item, mapping) for item in expression["args"]))


def _mapping_candidates(lexical_feature: str, grammar_feature: str) -> tuple[RouteMapping, ...]:
    """Return the twelve role orientations and evidence orders of one axis pair."""
    evidence_pool = tuple(feature for feature in LEXICAL_FEATURES if feature != lexical_feature)
    candidates = tuple(
        (role_zero, role_one, tuple(evidence))
        for role_zero, role_one in (
            (lexical_feature, grammar_feature),
            (grammar_feature, lexical_feature),
        )
        for evidence in permutations(evidence_pool, 3)
    )
    if len(candidates) != 12:
        raise AssertionError("each paired route must have 12 mappings")
    return candidates


def placeholder_mapping(
    role_zero_feature: str, role_one_feature: str, evidence_features: Sequence[str]
) -> dict[str, str]:
    """Return the placeholder-to-feature mapping of one route."""
    if (
        len(evidence_features) != 3
        or len(set(evidence_features)) != 3
        or any(feature not in LEXICAL_FEATURES for feature in evidence_features)
    ):
        raise ValueError("evidence features must be three distinct lexical axes")
    if (
        role_zero_feature == role_one_feature
        or role_zero_feature in evidence_features
        or role_one_feature in evidence_features
    ):
        raise ValueError("queried roles and evidence axes must be disjoint")
    return {
        "Q": role_zero_feature,
        "N": role_one_feature,
        **dict(zip(OBSERVED_PLACEHOLDERS, evidence_features, strict=True)),
    }


def mapped_program(
    template: MechanismTemplate,
    role_zero_feature: str,
    role_one_feature: str,
    evidence_features: Sequence[str],
) -> dict[str, Any]:
    """Return the canonical feature-level program of one mapped template."""
    mapping = placeholder_mapping(role_zero_feature, role_one_feature, evidence_features)
    return canonicalize_program(
        {
            "dsl_version": DSL_VERSION,
            "variables": list(FEATURE_NAMES),
            "constraints": [
                substitute(constraint, mapping)
                for constraint in template.placeholder_program["constraints"]
            ],
        }
    )


def _role_behavior(
    template: MechanismTemplate, evidence_features: Sequence[str], *, role: int
) -> tuple[int, ...]:
    """Return one role's query behaviour at all 256 languages."""
    offsets = {feature: FEATURE_NAMES.index(feature) for feature in evidence_features}
    nuisance_truth = expression_truth(template.nuisance_payload)
    values = []
    for assignment_index in range(256):
        x_index = sum(
            ((assignment_index >> offsets[feature]) & 1) << position
            for position, feature in enumerate(evidence_features)
        )
        guard_active = template.query_behavior_signature[x_index] >= 0
        if role == 0:
            values.append(template.query_behavior_signature[x_index] if guard_active else -1)
        elif role == 1:
            values.append(-1 if guard_active else nuisance_truth[x_index])
        else:
            raise ValueError("paired query role must be zero or one")
    return tuple(values)


def _paired_signature(
    template: MechanismTemplate,
    role_zero_feature: str,
    role_one_feature: str,
    evidence_features: Sequence[str],
) -> Signature:
    """Return both roles' 256-language behaviours keyed by queried feature."""
    return tuple(
        sorted(
            (
                (role_zero_feature, _role_behavior(template, evidence_features, role=0)),
                (role_one_feature, _role_behavior(template, evidence_features, role=1)),
            )
        )
    )


def _signature_hash(signature: Signature) -> str:
    """Hash a paired signature as a feature-keyed object."""
    return object_hash({feature: list(values) for feature, values in signature})


@dataclass(frozen=True)
class InstantiationRoute:
    """One template bound to one query-axis pair."""

    template_id: str
    partition: str
    axis_pair_id: str
    queried_features: tuple[str, str]
    role_zero_feature: str
    role_one_feature: str
    ordered_evidence_features: tuple[str, str, str]
    class_semantic_hash: str
    logical_unit_semantic_hash: str
    paired_query_behavior_semantic_hash: str
    query_task_semantic_hashes: tuple[tuple[str, str], ...]
    query_behavior_task_semantic_hashes: tuple[tuple[str, str], ...]

    def public_metadata(self) -> dict[str, Any]:
        """Return the role-free route metadata published in the manifests."""
        return {
            "template_id": self.template_id,
            "partition": self.partition,
            "axis_pair_id": self.axis_pair_id,
            "queried_features_unordered": list(self.queried_features),
            "class_semantic_hash": self.class_semantic_hash,
            "logical_unit_semantic_hash": self.logical_unit_semantic_hash,
            "paired_query_behavior_semantic_hash": self.paired_query_behavior_semantic_hash,
            "query_task_semantic_hashes": dict(self.query_task_semantic_hashes),
            "query_behavior_task_semantic_hashes": dict(self.query_behavior_task_semantic_hashes),
        }

    def private_mapping(self) -> dict[str, Any]:
        """Return the evaluator-only route metadata, including the hidden roles."""
        return {
            **self.public_metadata(),
            "role_zero_feature": self.role_zero_feature,
            "role_one_feature": self.role_one_feature,
            "ordered_evidence_features": list(self.ordered_evidence_features),
        }


def _match_mappings(
    templates: Sequence[MechanismTemplate], lexical_feature: str, grammar_feature: str
) -> dict[str, RouteMapping]:
    """Assign every template a mapping with a globally unique paired signature."""
    candidates_by_template: dict[str, dict[Signature, RouteMapping]] = {}
    for template in templates:
        by_signature: dict[Signature, RouteMapping] = {}
        for role_zero, role_one, evidence in _mapping_candidates(lexical_feature, grammar_feature):
            signature = _paired_signature(template, role_zero, role_one, evidence)
            mapping = (role_zero, role_one, evidence)
            previous = by_signature.get(signature)
            if previous is None or mapping < previous:
                by_signature[signature] = mapping
        candidates_by_template[template.template_id] = by_signature

    owner_by_signature: dict[Signature, str] = {}
    selected_by_template: dict[str, Signature] = {}

    def augment(template_id: str, visited: set[Signature]) -> bool:
        """Try to give a template a signature, displacing earlier owners if needed."""
        candidates = candidates_by_template[template_id]
        for signature in sorted(
            candidates, key=lambda value: (_signature_hash(value), candidates[value])
        ):
            if signature in visited:
                continue
            visited.add(signature)
            owner = owner_by_signature.get(signature)
            if owner is None or augment(owner, visited):
                owner_by_signature[signature] = template_id
                selected_by_template[template_id] = signature
                return True
        return False

    ordered = sorted(
        templates,
        key=lambda item: (
            len(candidates_by_template[item.template_id]),
            item.partition != "support",
            item.template_id,
        ),
    )
    for template in ordered:
        if not augment(template.template_id, set()):
            raise AssertionError(
                "no collision-free paired-query mapping for "
                f"{template.template_id}/{lexical_feature}+{grammar_feature}"
            )
    return {
        template_id: candidates_by_template[template_id][signature]
        for template_id, signature in selected_by_template.items()
    }


def _task_hashes(
    queried: Sequence[str], payload: Mapping[str, str], key: str
) -> tuple[tuple[str, str], ...]:
    """Hash one semantic identity per queried feature."""
    return tuple(
        sorted(
            (feature, object_hash({key: payload[feature], "query_feature": feature}))
            for feature in queried
        )
    )


def _build_route(
    template: MechanismTemplate, axis_pair_id: str, mapping: RouteMapping
) -> InstantiationRoute:
    """Instantiate one template on one axis pair and hash its semantics."""
    role_zero_feature, role_one_feature, evidence = mapping
    program = mapped_program(template, role_zero_feature, role_one_feature, evidence)
    accepted = accepted_indices(program)
    if len(accepted) != 128:
        raise AssertionError("every declaration must contain exactly 128 hypotheses")
    class_hash = object_hash(list(accepted))
    signature = _paired_signature(template, role_zero_feature, role_one_feature, evidence)
    behaviors = {
        role_zero_feature: _role_behavior(template, evidence, role=0),
        role_one_feature: _role_behavior(template, evidence, role=1),
    }
    queried = tuple(sorted((role_zero_feature, role_one_feature)))
    return InstantiationRoute(
        template_id=template.template_id,
        partition=template.partition,
        axis_pair_id=axis_pair_id,
        queried_features=queried,
        role_zero_feature=role_zero_feature,
        role_one_feature=role_one_feature,
        ordered_evidence_features=evidence,
        class_semantic_hash=class_hash,
        logical_unit_semantic_hash=object_hash(
            {"class_semantic_hash": class_hash, "queried_features_unordered": list(queried)}
        ),
        paired_query_behavior_semantic_hash=_signature_hash(signature),
        query_task_semantic_hashes=_task_hashes(
            queried, {feature: class_hash for feature in queried}, "class_semantic_hash"
        ),
        query_behavior_task_semantic_hashes=_task_hashes(
            queried,
            {feature: object_hash(list(behaviors[feature])) for feature in queried},
            "mapped_query_behavior_semantic_hash",
        ),
    )


@dataclass(frozen=True)
class RouteRegistry:
    """Every instantiation route, indexed by template and axis pair."""

    routes: tuple[InstantiationRoute, ...]
    by_key: dict[tuple[str, str], InstantiationRoute]
    registry_hash: str

    def partition(self, name: str) -> list[InstantiationRoute]:
        """Return the routes of the support or evaluation partition."""
        return [item for item in self.routes if item.partition == name]


@lru_cache(maxsize=1)
def route_registry() -> RouteRegistry:
    """Build the deterministic instantiation registry once per process."""
    templates = tuple(
        sorted(
            template_registry().all,
            key=lambda item: (item.partition != "support", item.template_id),
        )
    )
    routes: list[InstantiationRoute] = []
    for axis_pair_id, lexical_feature, grammar_feature in QUERY_AXIS_PAIRS:
        selected = _match_mappings(templates, lexical_feature, grammar_feature)
        routes.extend(
            _build_route(template, axis_pair_id, selected[template.template_id])
            for template in templates
        )
    routes.sort(key=lambda item: (item.partition != "support", item.template_id, item.axis_pair_id))
    for field in (
        "class_semantic_hash",
        "logical_unit_semantic_hash",
        "paired_query_behavior_semantic_hash",
    ):
        values = [getattr(item, field) for item in routes]
        if len(set(values)) != len(values):
            raise AssertionError(f"instantiated {field} values are not unique")
        support = {getattr(item, field) for item in routes if item.partition == "support"}
        evaluation = {getattr(item, field) for item in routes if item.partition == "evaluation"}
        if support & evaluation:
            raise AssertionError(f"support/evaluation overlap under {field}")
    return RouteRegistry(
        routes=tuple(routes),
        by_key={(item.template_id, item.axis_pair_id): item for item in routes},
        registry_hash=object_hash([item.private_mapping() for item in routes]),
    )


@dataclass(frozen=True)
class Instantiation:
    """The feature-level program and expressions of one route."""

    program: dict[str, Any]
    guard: dict[str, Any]
    role_zero_payload: dict[str, Any]
    role_one_payload: dict[str, Any]
    route: InstantiationRoute


def instantiate(template: MechanismTemplate, axis_pair_id: str) -> Instantiation:
    """Return the instantiated program, guard and payloads of one route."""
    try:
        route = route_registry().by_key[(template.template_id, axis_pair_id)]
    except KeyError as exc:
        raise ValueError("unknown template/axis-pair route") from exc
    mapping = placeholder_mapping(
        route.role_zero_feature, route.role_one_feature, route.ordered_evidence_features
    )
    program = mapped_program(
        template, route.role_zero_feature, route.role_one_feature, route.ordered_evidence_features
    )
    if object_hash(list(accepted_indices(program))) != route.class_semantic_hash:
        raise AssertionError("instantiated program differs from its registry")
    return Instantiation(
        program=program,
        guard=canonicalize_expression(substitute(template.guard, mapping)),
        role_zero_payload=canonicalize_expression(substitute(template.query_payload, mapping)),
        role_one_payload=canonicalize_expression(substitute(template.nuisance_payload, mapping)),
        route=route,
    )


def route_protocol() -> dict[str, Any]:
    """Return the template partition and route-sampling protocol of the manifests."""
    templates = template_registry()
    support_behavior = {item.paired_behavior_hash for item in templates.support}
    evaluation_behavior = {item.paired_behavior_hash for item in templates.evaluation}
    support_full = {item.full_semantic_hash for item in templates.support}
    evaluation_full = {item.full_semantic_hash for item in templates.evaluation}
    return {
        "schema_version": ROUTE_PROTOCOL_SCHEMA,
        "development_only": LIFECYCLE["development_only"],
        "confirmatory_status": LIFECYCLE["confirmatory_status"],
        "partition_units": [
            "full five-placeholder accepted-assignment truth table",
            "paired query-behavior truth tables over all eight X triples",
        ],
        "placeholder_full_semantic_hash_definition": PLACEHOLDER_FULL_SEMANTIC_HASH_DEFINITION,
        "query_behavior_hash_definition": {
            **QUERY_BEHAVIOR_HASH_DEFINITION,
            "paired_roles": (
                "both conditionally constrained coordinates, with no privileged public role"
            ),
        },
        "template_semantic_packing": templates.packing_audit,
        "route_state_sampling": {
            "algorithm_version": ROUTE_CYCLING_ALGORITHM,
            "strata": list(STRATA),
            "route_rule": "route j selects sorted stratum member j modulo stratum size",
            "routes_per_template": len(QUERY_AXIS_PAIRS),
            "states_per_route": STATES_PER_UNIT,
            "required_observed_state_union": list(range(8)),
            "required_per_member_multiplicity": "floor_or_ceiling",
        },
        "support_templates": [item.public_metadata() for item in templates.support],
        "evaluation_templates": [item.public_metadata() for item in templates.evaluation],
        "support_template_count": len(templates.support),
        "evaluation_template_count": len(templates.evaluation),
        "evaluation_templates_per_family": dict(
            sorted(Counter(item.family for item in templates.evaluation).items())
        ),
        "heldout_template_families": sorted({item.family for item in templates.support}),
        "heldout_mechanism_families": ["selector_multiplexer"],
        "support_evaluation_full_semantic_intersection": sorted(support_full & evaluation_full),
        "support_evaluation_paired_query_behavior_intersection": sorted(
            support_behavior & evaluation_behavior
        ),
        "all_template_full_semantics_unique": (
            len(support_full | evaluation_full) == len(templates.all)
        ),
        "all_template_paired_query_behaviors_unique": (
            len(support_behavior | evaluation_behavior) == len(templates.all)
        ),
    }


def query_axis_pairs() -> list[dict[str, Any]]:
    """Return the eight query-axis pairs with their unordered features."""
    return [
        {"axis_pair_id": pair_id, "queried_features_unordered": sorted((lexical, grammar))}
        for pair_id, lexical, grammar in QUERY_AXIS_PAIRS
    ]


def evaluation_semantic_counts(routes: Sequence[InstantiationRoute]) -> dict[str, int]:
    """Count the distinct semantic identities of the evaluation routes."""
    return {
        "full_class_semantics": len({item.class_semantic_hash for item in routes}),
        "logical_unit_semantics": len({item.logical_unit_semantic_hash for item in routes}),
        "paired_query_behaviors": len(
            {item.paired_query_behavior_semantic_hash for item in routes}
        ),
        "query_tasks": len(
            {task_hash for item in routes for _, task_hash in item.query_task_semantic_hashes}
        ),
    }


def instantiation_protocol() -> dict[str, Any]:
    """Return the instantiation protocol published in the manifests."""
    registry = route_registry()
    support = registry.partition("support")
    evaluation = registry.partition("evaluation")
    counts = evaluation_semantic_counts(evaluation)
    return {
        "schema_version": INSTANTIATION_PROTOCOL_SCHEMA,
        "selection_unit": "template_id_by_lexical_grammar_query_pair",
        "candidate_mapping_rule": (
            "both hidden role orientations and all six permutations of the "
            "three remaining lexical evidence features"
        ),
        "selection_algorithm": (
            "deterministic augmenting-path matching over exact paired query-"
            "behavior truth tables independently per axis pair"
        ),
        "class_semantic_hash_definition": CLASS_SEMANTIC_HASH_DEFINITION,
        "mapped_query_behavior_hash_definition": MAPPED_QUERY_BEHAVIOR_HASH_DEFINITION,
        "registry_hash": registry.registry_hash,
        "public_role_metadata_released": False,
        "query_axis_pairs": query_axis_pairs(),
        "registry_entries": [item.public_metadata() for item in registry.routes],
        "support_paired_units": len(support),
        "evaluation_paired_units": len(evaluation),
        "evaluation_query_tasks": 2 * len(evaluation),
        "evaluation_unique_full_class_semantics": counts["full_class_semantics"],
        "evaluation_unique_logical_unit_semantics": counts["logical_unit_semantics"],
        "evaluation_unique_paired_query_behaviors": counts["paired_query_behaviors"],
        "evaluation_unique_query_tasks": counts["query_tasks"],
        "support_evaluation_full_class_intersection": sorted(
            {item.class_semantic_hash for item in support}
            & {item.class_semantic_hash for item in evaluation}
        ),
        "support_evaluation_paired_query_behavior_intersection": sorted(
            {item.paired_query_behavior_semantic_hash for item in support}
            & {item.paired_query_behavior_semantic_hash for item in evaluation}
        ),
    }
