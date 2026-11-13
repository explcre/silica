"""Deterministic generation of the released SILICA dataset.

The release contains 12,288 rows: 192 evaluation templates (48 in each of four
mechanism families), each instantiated on eight query-axis pairs.  One
paired-query unit is one instantiated program with four observation states,
and each state is asked under both query-facing features, so every unit has
eight rows: four ``ANSWER`` and four ``ABSTAIN``.  The two rows that ask about
the same feature across the four states form a four-cell evaluator unit.

Public rows carry only what a model may see.  Gold rows carry the decision,
the certificate, and evaluator-only routing metadata.  Every identifier is a
content hash, so regenerating with the release seed reproduces the released
files byte for byte.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._frozen import (
    ACRONYM,
    ADAPTER_STATUS,
    CELL_DOMAIN,
    CELL_PREFIX,
    COHORT_VALIDATOR,
    COMPLETE_UNIT_DOMAIN,
    COMPLETE_UNIT_PREFIX,
    CONSTRUCTION_FREEZE_SCHEMA,
    DATASET_FORMAT,
    DATASET_NAME,
    DATASET_VERSION,
    GENERATOR_VERSION,
    INVENTORY_DOMAIN,
    JOINED_SEMANTIC_CONTENT_SHA256,
    LANGUAGE_DOMAIN,
    LIFECYCLE,
    PAIRED_STATE_DOMAIN,
    PAIRED_STATE_PREFIX,
    PAIRED_UNIT_PREFIX,
    PREDECESSOR,
    PUBLIC_FORMAT,
    PUBLIC_SEMANTIC_CONTENT_SHA256,
    RELEASE_EQUIVALENCE_SCHEMA,
    ROUTE_CYCLING_ALGORITHM,
    SCHEMA_FAMILY,
    SEALED_AT_UTC,
    SEED,
    SEMANTIC_CONTENT_SCHEMA,
    TEMPLATE_PACKING_ALGORITHM,
    TRACK,
)
from .certificates import TASKS, answer_query, make_certificate, minimum_sufficient_evidence
from .declaration import declaration_hash, program_declaration
from .dsl import (
    DSL_SEMANTICS,
    DSL_VERSION,
    evaluate_expression,
    hypothesis_assignment,
    program_accepts,
    program_metrics,
)
from .evidence import DictionaryEvidence
from .hypotheses import FEATURE_NAMES, HYPOTHESIS_CLASS_HASH, Hypothesis
from .routes import (
    QUERY_AXIS_PAIRS,
    Instantiation,
    instantiate,
    instantiation_protocol,
    query_axis_pairs,
    route_protocol,
    route_registry,
)
from .surface import SurfaceInventory, root_for
from .templates import MECHANISM_FAMILIES, MechanismTemplate, balanced_quartet, template_registry
from .util import file_sha256, object_hash, write_json, write_jsonl

SPLIT = "test"
POLICY = "fixed_counterfactual_three_dictionary"
BUDGET = {"id": "dictionary_exact_3", "dictionary": 3, "parallel": 0}
RELEASE_ROWS = 12_288
PUBLIC_FORBIDDEN_KEYS = frozenset(
    (
        "seed",
        "target_hypothesis_id",
        "surface_inventory",
        "gold",
        "certificate",
        "support",
        "oracle",
    )
)
# Query-derived public fields; everything else is shared by the two queries of a state.
QUERY_FIELDS = (
    "id",
    "complete_unit_id",
    "task",
    "item_index",
    "query_id",
    "counterfactual_cell_id",
    "query",
)
# The observation each evidence feature reveals: the form of one fixed meaning.
_EVIDENCE_PROMPTS: dict[str, tuple[str, str]] = {
    "noun_swap_01": ("noun", "fox"),
    "noun_swap_23": ("noun", "dog"),
    "verb_swap_01": ("verb", "see"),
    "modifier_swap_01": ("modifier", "red"),
}
# The form slot a lexical query asks about.
_LEXICAL_QUERY_SLOTS: dict[str, tuple[str, int]] = {
    "noun_swap_01": ("noun", 0),
    "noun_swap_23": ("noun", 2),
    "verb_swap_01": ("verb", 0),
    "modifier_swap_01": ("modifier", 0),
}


def forbidden_paths(value: Any, path: str = "$") -> list[str]:
    """Return the JSON paths of evaluator-only keys inside a public value."""
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if str(key) in PUBLIC_FORBIDDEN_KEYS:
                found.append(child_path)
            found.extend(forbidden_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{path}[{index}]"))
    return found


def _require_public(value: Mapping[str, Any]) -> None:
    """Fail if a public value contains an evaluator-only key."""
    leaked = forbidden_paths(value)
    if leaked:
        raise ValueError(f"forbidden public key at {leaked[0]}")


def public_instance_id(public_without_id: Mapping[str, Any]) -> str:
    """Return the content-derived identifier of a public row."""
    return "silica-" + object_hash(dict(public_without_id))[:24]


def _short_id(prefix: str, payload: Mapping[str, Any]) -> str:
    """Return a prefixed 20-hex-digit content identifier."""
    return prefix + object_hash(payload)[:20]


def _query_for_feature(
    feature: str, inventory: SurfaceInventory
) -> tuple[str, dict[str, Any], str]:
    """Return the task, public query and query variant that ask about one feature."""
    if feature in _LEXICAL_QUERY_SLOTS:
        category, slot = _LEXICAL_QUERY_SLOTS[feature]
        query = {"category": category, "form": inventory.roots(category)[slot]}
        return "lexicon", query, f"lexical_axis_{feature}"
    return "grammar_feature", {"feature": feature}, f"grammar_axis_{feature}"


def _bit_from_hash(*parts: Any) -> int:
    """Return one deterministic pseudo-random bit."""
    return int(object_hash(list(parts))[-1], 16) & 1


def _target(
    seed: str,
    template: MechanismTemplate,
    instance: Instantiation,
    blueprint: Mapping[str, Any],
    cell_nonce: int,
) -> Hypothesis:
    """Return the hidden target language of one observation state."""
    route = instance.route
    assigned = {
        route.role_zero_feature: int(blueprint["role_zero_bit"]),
        route.role_one_feature: int(blueprint["role_one_bit"]),
        **{
            feature: int(bit)
            for feature, bit in zip(
                route.ordered_evidence_features, blueprint["observed_bits"], strict=True
            )
        },
    }
    return Hypothesis(
        tuple(
            assigned.get(
                feature,
                _bit_from_hash(
                    SCHEMA_FAMILY,
                    seed,
                    template.template_id,
                    route.role_zero_feature,
                    route.role_one_feature,
                    cell_nonce,
                    feature,
                ),
            )
            for feature in FEATURE_NAMES
        )
    )


def _check_target(
    target: Hypothesis, instance: Instantiation, blueprint: Mapping[str, Any]
) -> bool:
    """Verify a target satisfies its program, guard state and pinned payload; return the guard."""
    assignment = hypothesis_assignment(target)
    route = instance.route
    if not program_accepts(instance.program, assignment):
        raise AssertionError("constructed paired target violates its program")
    guard_active = evaluate_expression(instance.guard, assignment)
    if guard_active != bool(blueprint["guard_active"]):
        raise AssertionError("constructed paired target has the wrong guard state")
    if guard_active:
        if target.bit(route.role_zero_feature) != int(
            evaluate_expression(instance.role_zero_payload, assignment)
        ):
            raise AssertionError("active state violates the fixed role-zero payload")
    elif target.bit(route.role_one_feature) != int(
        evaluate_expression(instance.role_one_payload, assignment)
    ):
        raise AssertionError("inactive state violates the fixed role-one payload")
    return guard_active


def _paired_unit_rows(
    seed: str, template: MechanismTemplate, family_index: int, pair_index: int, axis_pair_id: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build the eight public/gold row pairs of one template on one axis pair."""
    quartet = balanced_quartet(
        template.guard, template.query_payload, template.nuisance_payload, route_index=pair_index
    )
    if quartet is None:
        raise AssertionError(f"template lacks a matched quartet: {template.template_id}")
    instance = instantiate(template, axis_pair_id)
    route = instance.route
    paired_query_unit_id = _short_id(
        PAIRED_UNIT_PREFIX,
        {
            "schema_family": SCHEMA_FAMILY,
            "seed": seed,
            "template_id": template.template_id,
            "axis_pair_id": axis_pair_id,
            "class_semantic_hash": route.class_semantic_hash,
            "logical_unit_semantic_hash": route.logical_unit_semantic_hash,
            "paired_query_behavior_semantic_hash": route.paired_query_behavior_semantic_hash,
            "instantiation_registry_hash": route_registry().registry_hash,
        },
    )
    language_id = _short_id(
        "lang-", {"domain": LANGUAGE_DOMAIN, "paired_query_unit_id": paired_query_unit_id}
    )
    inventory = SurfaceInventory.from_seed(
        object_hash(
            {"domain": INVENTORY_DOMAIN, "seed": seed, "paired_query_unit_id": paired_query_unit_id}
        )
    )
    declaration = program_declaration(inventory, instance.program)
    mechanism = {
        "family": template.family,
        "template_id": template.template_id,
        "route": template.route,
        "template_hash": template.template_hash,
        "placeholder_full_semantic_hash": template.full_semantic_hash,
        "paired_template_query_behavior_semantic_hash": template.paired_behavior_hash,
        "class_semantic_hash": route.class_semantic_hash,
        "logical_unit_semantic_hash": route.logical_unit_semantic_hash,
        "paired_query_behavior_semantic_hash": route.paired_query_behavior_semantic_hash,
        "instantiation_registry_hash": route_registry().registry_hash,
        "public_role_metadata_present": False,
    }
    complete_unit_ids = {
        feature: _short_id(
            COMPLETE_UNIT_PREFIX,
            {
                "domain": COMPLETE_UNIT_DOMAIN,
                "paired_query_unit_id": paired_query_unit_id,
                "query_feature": feature,
            },
        )
        for feature in route.queried_features
    }
    structure = program_metrics(instance.program)
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for state_index, blueprint in enumerate(quartet):
        target = _target(seed, template, instance, blueprint, state_index)
        guard_active = _check_target(target, instance, blueprint)
        assignment = hypothesis_assignment(target)
        evidence = tuple(
            DictionaryEvidence(
                *_EVIDENCE_PROMPTS[feature],
                root_for(*_EVIDENCE_PROMPTS[feature], target, inventory),
            )
            for feature in route.ordered_evidence_features
        )
        paired_state_id = _short_id(
            PAIRED_STATE_PREFIX,
            {
                "domain": PAIRED_STATE_DOMAIN,
                "paired_query_unit_id": paired_query_unit_id,
                "state_index": state_index,
                "observations": [item.to_dict() for item in evidence],
            },
        )
        shared = {
            "dataset_version": DATASET_VERSION,
            "track": TRACK,
            "split": SPLIT,
            "language_id": language_id,
            "policy": POLICY,
            "budget": dict(BUDGET),
            "paired_query_unit_id": paired_query_unit_id,
            "paired_state_id": paired_state_id,
            "mechanism": mechanism,
            "structural_complexity": structure,
            "hypothesis_declaration": declaration,
            "evidence": [item.to_dict() for item in evidence],
        }
        orientations = [(0, route.role_zero_feature), (1, route.role_one_feature)]
        if (family_index + pair_index) % 2:
            orientations.reverse()
        state_rows = []
        for orientation_position, (hidden_role, query_feature) in enumerate(orientations):
            task, query, query_variant = _query_for_feature(query_feature, inventory)
            complete_unit_id = complete_unit_ids[query_feature]
            certificate, consensus_gold = make_certificate(
                task, query, evidence, inventory, instance.program
            )
            expected = "ANSWER" if guard_active == (hidden_role == 0) else "ABSTAIN"
            if consensus_gold["decision"] != expected:
                raise AssertionError("paired query complementarity failed")
            base = {
                **shared,
                "complete_unit_id": complete_unit_id,
                "task": task,
                "item_index": FEATURE_NAMES.index(query_feature),
                "query_id": _short_id(
                    "query-", {"complete_unit_id": complete_unit_id, "task": task, "query": query}
                ),
                "counterfactual_cell_id": _short_id(
                    CELL_PREFIX,
                    {"domain": CELL_DOMAIN, "paired_state_id": paired_state_id, "query": query},
                ),
                "query": query,
            }
            identifier = public_instance_id(base)
            public = {"id": identifier, **base}
            _require_public(public)
            target_answer = answer_query(task, query, target, inventory)
            oracle = {
                "surface_inventory": inventory.to_dict(),
                "target_hypothesis_id": target.id,
                "target_answer": target_answer,
                "query_variant": query_variant,
                "query_feature": query_feature,
                "hidden_role_index": hidden_role,
                "role_zero_feature": route.role_zero_feature,
                "role_one_feature": route.role_one_feature,
                "ordered_evidence_features": list(route.ordered_evidence_features),
                "complete_unit_id": complete_unit_id,
                "paired_query_unit_id": paired_query_unit_id,
                "paired_state_id": paired_state_id,
                "public_state_index": state_index,
                "orientation_position": orientation_position,
                "blueprint_index": state_index,
                "guard_active": guard_active,
                "observed_feature_bits": dict(
                    zip(route.ordered_evidence_features, blueprint["observed_bits"], strict=True)
                ),
                "role_zero_feature_bit": target.bit(route.role_zero_feature),
                "role_one_feature_bit": target.bit(route.role_one_feature),
                "role_zero_payload_value": evaluate_expression(
                    instance.role_zero_payload, assignment
                ),
                "role_one_payload_value": evaluate_expression(
                    instance.role_one_payload, assignment
                ),
                "minimum_sufficient_proof": minimum_sufficient_evidence(
                    task, query, evidence, inventory, instance.program
                ),
                "public_record_hash": object_hash(public),
                "declaration_hash": declaration_hash(declaration),
            }
            gold = {
                "id": identifier,
                "gold": {**consensus_gold, "target_answer": target_answer},
                "certificate": certificate,
                "oracle": oracle,
            }
            state_rows.append((public, gold))
        decisions = Counter(gold["gold"]["decision"] for _, gold in state_rows)
        if decisions != Counter({"ANSWER": 1, "ABSTAIN": 1}):
            raise AssertionError("each state must have complementary decisions")
        if len({object_hash(query_masked_state(public)) for public, _ in state_rows}) != 1:
            raise AssertionError("paired rows differ outside query-derived fields")
        rows.extend(state_rows)
    return rows


def query_masked_state(public: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the query and the query-derived fields of a public row."""
    return {key: value for key, value in public.items() if key not in QUERY_FIELDS}


def build_records(
    *, seed: str = SEED, template_ids: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the public and gold rows, optionally for a subset of evaluation templates."""
    public_records: list[dict[str, Any]] = []
    gold_records: list[dict[str, Any]] = []
    family_ordinals: Counter[str] = Counter()
    for template in template_registry().evaluation:
        family_index = family_ordinals[template.family]
        family_ordinals[template.family] += 1
        if template_ids is not None and template.template_id not in template_ids:
            continue
        for pair_index, (axis_pair_id, _, _) in enumerate(QUERY_AXIS_PAIRS):
            for public, gold in _paired_unit_rows(
                seed, template, family_index, pair_index, axis_pair_id
            ):
                public_records.append(public)
                gold_records.append(gold)
    if template_ids is None and len(public_records) != RELEASE_ROWS:
        raise AssertionError(f"built {len(public_records)} rows, expected {RELEASE_ROWS}")
    return public_records, gold_records


def semantic_content_commitments(
    public_records: Sequence[Mapping[str, Any]], gold_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Hash the release-invariant meaning of every row, excluding derived identities."""
    gold_by_id = {str(item["id"]): item for item in gold_records}
    if len(gold_by_id) != len(gold_records) or set(gold_by_id) != {
        str(item["id"]) for item in public_records
    }:
        raise ValueError("cannot commit semantics without an exact public/gold join")
    normalized_public = []
    normalized_joined = []
    for public in public_records:
        public_semantics = copy.deepcopy(dict(public))
        for release_field in ("id", "dataset_version", "split"):
            public_semantics.pop(release_field)
        gold_semantics = copy.deepcopy(dict(gold_by_id[str(public["id"])]))
        gold_semantics.pop("id")
        oracle = gold_semantics.get("oracle")
        if not isinstance(oracle, dict):
            raise ValueError("gold row lacks an oracle object")
        oracle.pop("public_record_hash")
        key = (
            str(public["paired_query_unit_id"]),
            str(public["paired_state_id"]),
            str(public["query_id"]),
        )
        normalized_public.append((key, public_semantics))
        normalized_joined.append((key, {"public": public_semantics, "gold": gold_semantics}))
    normalized_public.sort(key=lambda item: item[0])
    normalized_joined.sort(key=lambda item: item[0])
    if len({item[0] for item in normalized_joined}) != len(normalized_joined):
        raise ValueError("release-invariant semantic row keys are not unique")
    return {
        "schema_version": SEMANTIC_CONTENT_SCHEMA,
        "normalization": {
            "public_fields_removed": ["dataset_version", "id", "split"],
            "gold_fields_removed": ["id", "oracle.public_record_hash"],
            "reason": "release metadata and values deterministically derived from it",
            "ordering_key": ["paired_query_unit_id", "paired_state_id", "query_id"],
        },
        "rows": len(normalized_joined),
        "public_semantic_content_sha256": object_hash([item[1] for item in normalized_public]),
        "joined_semantic_content_sha256": object_hash([item[1] for item in normalized_joined]),
    }


def require_frozen_semantics(seed: str, commitments: Mapping[str, Any]) -> None:
    """Fail unless the rows carry exactly the frozen semantic content of the release."""
    if seed != SEED:
        raise ValueError("the sealed release requires the frozen construction seed")
    expected = {
        "public_semantic_content_sha256": PUBLIC_SEMANTIC_CONTENT_SHA256,
        "joined_semantic_content_sha256": JOINED_SEMANTIC_CONTENT_SHA256,
    }
    if {key: commitments.get(key) for key in expected} != expected:
        raise ValueError("semantic content differs from the frozen release")


def lifecycle_metadata(seed: str, commitments: Mapping[str, Any]) -> dict[str, Any]:
    """Return the lifecycle, freeze and equivalence fields shared by both manifests."""
    packing_selection = template_registry().packing_audit["selection"]["evaluation_candidate_hash"]
    freeze = {
        "schema_version": CONSTRUCTION_FREEZE_SCHEMA,
        "status": "frozen",
        "sealed_at_utc": SEALED_AT_UTC,
        "sealed_before_direct_model_outputs": True,
        "direct_model_output_status_at_seal": "not_generated",
        "seed_sha256": object_hash({"seed": seed}),
        "semantic_content_schema_version": commitments["schema_version"],
        "public_semantic_content_sha256": commitments["public_semantic_content_sha256"],
        "joined_semantic_content_sha256": commitments["joined_semantic_content_sha256"],
        "template_packing_algorithm_version": TEMPLATE_PACKING_ALGORITHM,
        "template_packing_selection_hash": packing_selection,
        "instantiation_registry_hash": route_registry().registry_hash,
        "route_cycling_algorithm_version": ROUTE_CYCLING_ALGORITHM,
    }
    return {
        **copy.deepcopy(LIFECYCLE),
        "construction_freeze": {**freeze, "freeze_hash": object_hash(freeze)},
        "semantic_content": dict(commitments),
        "semantic_equivalence": {
            "schema_version": RELEASE_EQUIVALENCE_SCHEMA,
            "status": "verified_equal",
            "predecessor": dict(PREDECESSOR),
            "allowed_release_transformations": [
                "dataset_version",
                "split",
                "derived_public_id",
                "derived_gold_id",
                "oracle.public_record_hash",
                "shard_path",
                "manifest_lifecycle_metadata",
            ],
            "joined_semantic_content_sha256": JOINED_SEMANTIC_CONTENT_SHA256,
        },
    }


def shard_name(family: str) -> str:
    """Return the file name of one mechanism family's shard."""
    return f"{TRACK}__{SPLIT}__{family}.jsonl"


def _sha256sums(root: Path, paths: Sequence[Path]) -> str:
    """Return the SHA256SUMS text of the given files, sorted by relative path."""
    lines = [
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())
    ]
    return "\n".join(lines) + "\n"


def sha256sums_paths(root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    """Return every file listed in SHA256SUMS: shards and both manifests."""
    entries = [*manifest.get("public_files", []), *manifest.get("gold_files", [])]
    return [
        *(root / str(item["path"]) for item in entries),
        root / "public" / "manifest.json",
        root / "manifest.json",
    ]


def _write_dataset(root: Path, seed: str) -> dict[str, Any]:
    """Write shards, manifests and SHA256SUMS into an empty directory."""
    from .acceptance import acceptance_audit

    public_dir = root / "public"
    gold_dir = root / "gold"
    public_dir.mkdir(parents=True)
    gold_dir.mkdir(parents=True)
    public_records, gold_records = build_records(seed=seed)
    commitments = semantic_content_commitments(public_records, gold_records)
    require_frozen_semantics(seed, commitments)
    lifecycle = lifecycle_metadata(seed, commitments)
    gold_by_id = {str(item["id"]): item for item in gold_records}
    acceptance = acceptance_audit(public_records, gold_records)
    public_entries = []
    gold_entries = []
    for family in MECHANISM_FAMILIES:
        family_public = [item for item in public_records if item["mechanism"]["family"] == family]
        family_gold = [gold_by_id[str(item["id"])] for item in family_public]
        stem = shard_name(family)
        public_count = write_jsonl(public_dir / stem, family_public)
        gold_count = write_jsonl(gold_dir / stem, family_gold)
        if public_count != RELEASE_ROWS // len(MECHANISM_FAMILIES) or gold_count != public_count:
            raise AssertionError(f"{family}: unexpected shard size {public_count}")
        common = {
            "records": public_count,
            "track": TRACK,
            "split": SPLIT,
            "mechanism_family": family,
        }
        public_entries.append(
            {"path": f"public/{stem}", "sha256": file_sha256(public_dir / stem), **common}
        )
        gold_entries.append(
            {"path": f"gold/{stem}", "sha256": file_sha256(gold_dir / stem), **common}
        )

    routes = route_protocol()
    instantiations = instantiation_protocol()
    scale = acceptance["scale_counts"]
    shared = {
        "schema_family": SCHEMA_FAMILY,
        **lifecycle,
        "generic_llm_pipeline_compatible": False,
        "evaluation_adapter_status": ADAPTER_STATUS,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "acronym": ACRONYM,
        "generator_version": GENERATOR_VERSION,
        "serialization": "canonical-json-v1",
        "base_hypothesis_class_size": 256,
        "base_hypothesis_class_hash": HYPOTHESIS_CLASS_HASH,
        "boolean_dsl_version": DSL_VERSION,
        "route_protocol": routes,
        "instantiation_protocol": instantiations,
        "semantic_scale": acceptance["semantic_scale"],
    }
    public_manifest = {
        **shared,
        "format": PUBLIC_FORMAT,
        "view": "public",
        "boolean_dsl_semantics": DSL_SEMANTICS,
        "tracks": [TRACK],
        "splits": [SPLIT],
        "tasks": list(TASKS),
        "evidence_policies": [POLICY],
        "evidence_budgets": [dict(BUDGET)],
        "scale_counts": scale,
        "matched_design": {
            "paired_query_unit_definition": (
                "one exact program and four evidence states, each evaluated "
                "under both conditionally constrained query coordinates"
            ),
            "paired_query_groups": scale["paired_counterfactual_groups"],
            "four_cell_evaluator_units": scale["four_cell_evaluator_units"],
            "rows": scale["rows"],
            "primary_generalization_cluster": "semantic template",
            "primary_generalization_clusters": len(template_registry().evaluation),
            "counterfactual_states_per_paired_group": 4,
            "query_orientations_per_state": 2,
            "decision_balance_per_paired_group": {"ANSWER": 4, "ABSTAIN": 4},
            "decision_balance_per_complete_unit": {"ANSWER": 2, "ABSTAIN": 2},
            "paired_query_unit_id_is_public": True,
            "complete_unit_id_is_public": True,
            "hidden_role_assignments_are_evaluator_only": True,
        },
        "cohort_sampling_contract": {
            "required_sampling_unit": "paired_query_unit_id",
            "complete_unit_id_only_subsampling_is_valid": False,
            "full_dataset_evaluation_is_valid": True,
            "validator": COHORT_VALIDATOR,
        },
        "files": public_entries,
    }
    _require_public(public_manifest)
    write_json(public_dir / "manifest.json", public_manifest)

    config = {
        "artifact_profile": LIFECYCLE["artifact_profile"],
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "seed": seed,
        "split": SPLIT,
        "development_only": LIFECYCLE["development_only"],
        "construction_freeze_hash": lifecycle["construction_freeze"]["freeze_hash"],
        "evaluation_template_ids": [item.template_id for item in template_registry().evaluation],
        "query_axis_pairs": query_axis_pairs(),
        "instantiation_registry_hash": route_registry().registry_hash,
        "template_packing_algorithm_version": TEMPLATE_PACKING_ALGORITHM,
        "template_packing_selection_hash": lifecycle["construction_freeze"][
            "template_packing_selection_hash"
        ],
        "route_cycling_algorithm_version": ROUTE_CYCLING_ALGORITHM,
    }
    entries = sorted([*public_entries, *gold_entries], key=lambda item: item["path"])
    manifest = {
        **shared,
        "format": DATASET_FORMAT,
        "config": config,
        "config_hash": object_hash(config),
        "private_instantiation_registry": [
            item.private_mapping() for item in route_registry().routes
        ],
        "private_template_registry": [item.private_metadata() for item in template_registry().all],
        "public_manifest": {
            "path": "public/manifest.json",
            "sha256": file_sha256(public_dir / "manifest.json"),
        },
        "public_files": public_entries,
        "gold_files": gold_entries,
        "dataset_sha256": dataset_sha256(entries),
        "statistics": {
            **scale,
            "decision_counts": acceptance["decision_counts"],
            "declared_class_size_histogram": acceptance["declared_class_size_histogram"],
            "posterior_size_histogram": acceptance["posterior_size_histogram"],
        },
        "acceptance_audit": acceptance,
    }
    write_json(root / "manifest.json", manifest)
    (root / "SHA256SUMS").write_text(
        _sha256sums(root, sha256sums_paths(root, manifest)), encoding="utf-8", newline="\n"
    )
    return manifest


def dataset_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    """Return the aggregate checksum over path, digest and record count of every shard."""
    ordered = sorted(entries, key=lambda item: str(item["path"]))
    return object_hash([(item["path"], item["sha256"], item["records"]) for item in ordered])


def sha256sums_text(root: Path, manifest: Mapping[str, Any]) -> str:
    """Return the expected SHA256SUMS content of a dataset directory."""
    return _sha256sums(root, sha256sums_paths(root, manifest))


def build_dataset(output: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Generate, validate, and atomically publish the release into a directory."""
    from .validate import validate_dataset

    root = Path(output)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is nonempty: {root}; pass overwrite=True")
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-staging-", dir=root.parent) as temporary:
        staging = Path(temporary) / "artifact"
        manifest = _write_dataset(staging, SEED)
        validate_dataset(staging).require_ok()
        if root.exists():
            if any(root.iterdir()):
                shutil.rmtree(root)
            else:
                root.rmdir()
        staging.replace(root)
    return manifest
