"""Frozen identifiers of the evaluated SILICA release.

Every value in this module is copied byte for byte from the released dataset,
its manifests, or the model-facing prompt.  Several predate the final name of
the benchmark and are kept verbatim because changing a single byte would
change the regenerated artifact, every derived identifier, or the prompts the
evaluated models received.  Each constant is defined here once and nowhere
else in the package.
"""

from __future__ import annotations

from typing import Any

# Frozen identifier of the evaluated release: dataset name.
DATASET_NAME = "silica-hard-v2-rc1"
# Frozen identifier of the evaluated release: dataset version.
DATASET_VERSION = "0.2.0-rc1"
# Frozen identifier of the evaluated release: lifecycle profile name.
ARTIFACT_PROFILE = "rc1"
# Frozen identifier of the evaluated release: construction seed.
SEED = "silica-hard-v2-development-20260803"
# Frozen identifier of the evaluated release: generator version in the manifests.
GENERATOR_VERSION = "0.1.1"
# Frozen identifier of the evaluated release: track name of every row.
TRACK = "hard_posterior_matched"
# Frozen identifier of the evaluated release: schema family of rows and manifests.
SCHEMA_FAMILY = "silica-hard-posterior-matched-v2"
# Frozen identifier of the evaluated release: root manifest format.
DATASET_FORMAT = "silica-hard-v2-dataset-v1"
# Frozen identifier of the evaluated release: public manifest format.
PUBLIC_FORMAT = "silica-hard-v2-public-v1"
# Frozen identifier of the evaluated release: evaluation adapter status.
ADAPTER_STATUS = "pair_aware_sampling_adapter_v2_implemented"
# Frozen identifier of the evaluated release: cohort validator named in the manifest.
COHORT_VALIDATOR = "silica.hard_v2.validate_paired_cohort"
# Frozen identifier of the evaluated release: seed of the opaque surface forms.
SURFACE_FORM_DOMAIN = "silica-surface-v2"

# Frozen identifier of the evaluated release: template-packing algorithm.
TEMPLATE_PACKING_ALGORITHM = "silica-hard-v2-static-degree-balanced-round-robin-v1"
# Frozen identifier of the evaluated release: route-cycling algorithm.
ROUTE_CYCLING_ALGORITHM = "silica-hard-v2-balanced-strata-route-cycling-v1"
# Frozen identifier of the evaluated release: template-packing audit schema.
TEMPLATE_PACKING_SCHEMA = "silica-hard-v2-template-packing-audit-v1"
# Frozen identifier of the evaluated release: route protocol schema.
ROUTE_PROTOCOL_SCHEMA = "silica-hard-v2-route-protocol-v2"
# Frozen identifier of the evaluated release: instantiation protocol schema.
INSTANTIATION_PROTOCOL_SCHEMA = "silica-hard-v2-instantiation-protocol-v1"
# Frozen identifier of the evaluated release: route-state coverage audit schema.
ROUTE_COVERAGE_SCHEMA = "silica-hard-v2-route-state-coverage-audit-v1"
# Frozen identifier of the evaluated release: acceptance audit schema.
ACCEPTANCE_AUDIT_SCHEMA = "silica-hard-v2-acceptance-audit-v2"
# Frozen identifier of the evaluated release: semantic content commitment schema.
SEMANTIC_CONTENT_SCHEMA = "silica-hard-v2-semantic-content-v1"
# Frozen identifier of the evaluated release: construction freeze schema.
CONSTRUCTION_FREEZE_SCHEMA = "silica-hard-v2-construction-freeze-v1"
# Frozen identifier of the evaluated release: release equivalence schema.
RELEASE_EQUIVALENCE_SCHEMA = "silica-hard-v2-release-equivalence-v1"

# Frozen identifier of the evaluated release: prefix of paired-query unit IDs.
PAIRED_UNIT_PREFIX = "hard-v2-paired-"
# Frozen identifier of the evaluated release: prefix of four-cell unit IDs.
COMPLETE_UNIT_PREFIX = "hard-v2-unit-"
# Frozen identifier of the evaluated release: prefix of paired-state IDs.
PAIRED_STATE_PREFIX = "hard-v2-state-"
# Frozen identifier of the evaluated release: prefix of counterfactual-cell IDs.
CELL_PREFIX = "hard-v2-cell-"
# Frozen identifier of the evaluated release: hash domain of language IDs.
LANGUAGE_DOMAIN = "hard-v2-paired-language"
# Frozen identifier of the evaluated release: hash domain of surface inventories.
INVENTORY_DOMAIN = "hard-v2-paired-inventory"
# Frozen identifier of the evaluated release: hash domain of four-cell unit IDs.
COMPLETE_UNIT_DOMAIN = "hard-v2-four-cell-query-unit"
# Frozen identifier of the evaluated release: hash domain of paired-state IDs.
PAIRED_STATE_DOMAIN = "opaque-hard-v2-paired-state"
# Frozen identifier of the evaluated release: hash domain of counterfactual-cell IDs.
CELL_DOMAIN = "opaque-hard-v2-query-cell"

# Frozen identifier of the evaluated release: public semantic content commitment.
PUBLIC_SEMANTIC_CONTENT_SHA256 = "b53b704157f1c92e006e71c4cc4ea7e1f41598845960e6ec62f590a0ccfc82e3"
# Frozen identifier of the evaluated release: joined semantic content commitment.
JOINED_SEMANTIC_CONTENT_SHA256 = "f4f9e625dded0c0ee712012f14cd7ea3ad4f896ec991c2389139412dc2d584dd"
# Frozen identifier of the evaluated release: semantically equal predecessor.
PREDECESSOR = {
    "dataset_name": "silica-hard-v2-dev",
    "dataset_version": "0.2.0-dev",
    "dataset_sha256": ("816ae66196ff47fe043614e1bc833a9bbc9fcb74a7ab970328521080b7b853eb"),
    "strict_release_identity": ("311c4c14858cc72f1377b5e450a803dd0b452eac99c8aeebcafc2601e85c810f"),
}
# Frozen identifier of the evaluated release: lifecycle fields of both manifests.
LIFECYCLE = {
    "artifact_profile": ARTIFACT_PROFILE,
    "release_stage": "release_candidate",
    "artifact_sealed": True,
    "development_only": False,
    "confirmatory_status": "construction_frozen_pre_model_evaluation",
    "model_output_status": "not_generated",
    "hard_v2_rc1_schema_conformance": True,
    "rc1_schema_conformance": False,
    "rc1_schema_conformance_scope": "legacy_silica_dataset_v1",
    "schema_conformance": {
        "hard_v2_release_schema": DATASET_FORMAT,
        "hard_v2_release_schema_conformance": True,
        "validator": "silica.hard_v2.validate_hard_v2_dataset",
        "legacy_silica_rc1_schema": "silica-dataset-v1",
        "legacy_silica_rc1_schema_conformance": False,
        "legacy_nonconformance_reason": (
            "Hard-v2 uses a 128-member conditional class and paired eight-row "
            "sampling contract rather than the legacy 256-member language grid"
        ),
    },
}
# Frozen identifier of the evaluated release: UTC time the construction was sealed.
SEALED_AT_UTC = "2026-08-04T03:43:35Z"

# Frozen identifier of the evaluated release: prompt protocol version.
PROMPT_VERSION = "silica-hard-posterior-matched-json-v2"
# Frozen identifier of the evaluated release: evaluation adapter version.
ADAPTER_VERSION = "silica-hard-pair-aware-adapter-v2"
# Frozen identifier of the evaluated release: response JSON-schema version.
RESPONSE_SCHEMA_VERSION = "silica-answer-json-schema-v2"
# Frozen identifier of the evaluated release: public cohort selection method.
COHORT_METHOD = "public_hard_pair_aware_units_hash_v2"
# Frozen identifier of the evaluated release: public cohort selection seed.
COHORT_SEED = "silica-hard-public-units-v2"
# Frozen identifier of the evaluated release: adapter protocol bound into every prompt.
ADAPTER_PROTOCOL: dict[str, Any] = {
    "adapter_version": ADAPTER_VERSION,
    "cohort_method": COHORT_METHOD,
    "gold_audit_format": "silica-hard-post-selection-gold-audit-v3",
    "prompt_versions": {
        "complete_unit": "silica-hard-counterfactual-json-v1",
        "paired_query_unit": PROMPT_VERSION,
    },
    "response_schema_version": RESPONSE_SCHEMA_VERSION,
    "selection_order": "public_selection_then_privileged_gold_audit",
    "selection_modes": {
        "complete_unit": {
            "condition": "paired_query_unit_id absent from every public row",
            "sampling_unit_key": "complete_unit_id",
            "rows_per_sampling_unit": 4,
        },
        "paired_query_unit": {
            "condition": "paired_query_unit_id present on every public row",
            "sampling_unit_key": "paired_query_unit_id",
            "rows_per_sampling_unit": 8,
            "complete_unit_key": "complete_unit_id",
            "rows_per_complete_unit": 4,
            "complete_units_per_sampling_unit": 2,
            "paired_state_key": "paired_state_id",
            "paired_states_per_sampling_unit": 4,
            "query_orientations_per_paired_state": 2,
            "capped_selection": {
                "template_count": 192,
                "templates_per_family": 48,
                "query_axis_pair_routes_per_template": 8,
                "required_cap_multiple": 1536,
                "groups_per_template": "equal",
                "query_axis_pair_route_counts": "equal_globally_and_within_family",
            },
        },
    },
    "mixed_mode": "reject",
    "cap_policy": "zero_for_all_otherwise_exact_multiple_of_sampling_unit_size",
    "duplicate_public_item_ids": "reject",
}
# Frozen identifier of the evaluated release: system message of every prompt.
# Frozen identifier: the acronym expansion recorded in the released manifest.
ACRONYM = "Sparse Induction of Language with Identifiability Certificates and Abstention"

SYSTEM_PROMPT = (
    "You are solving the SILICA-Hard posterior-matched paired-query "
    "identifiability reasoning benchmark. The supplied declaration defines an "
    "exact finite hypothesis class: enumerate all Boolean feature assignments, "
    "retain only assignments accepted by every declared Boolean-program "
    "constraint, and then retain only assignments consistent with every observed "
    "form/meaning pair. Evaluate the query under every surviving assignment. "
    "Return ANSWER only if all survivors produce one identical canonical answer; "
    "otherwise return ABSTAIN. Reason silently. Emit exactly one JSON object "
    "satisfying canonical_response_schema, with no markdown, rationale, "
    "confidence, or additional keys."
)
