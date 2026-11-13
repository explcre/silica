"""Closed-world hypothesis declarations shown to benchmark participants.

A declaration names the opaque forms of every lexical axis, the grammar axes
and their semantics, the constant markers, and the Boolean program that
restricts the 256-language base class to the 128 languages of one row.  It is
everything needed to enumerate the admitted languages exactly, and nothing
that reveals the hidden target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._frozen import SCHEMA_FAMILY
from .dsl import DSL_SEMANTICS, class_semantic_hash, enumerate_program
from .hypotheses import FEATURE_DECLARATION, FEATURE_NAMES, HYPOTHESIS_CLASS_HASH
from .surface import SurfaceInventory
from .util import object_hash

DECLARED_CLASS_SIZE = 128
FULL_EVIDENCE_POSTERIOR_SIZE = 16
CLASS_SEMANTIC_HASH_DEFINITION = {
    "hash": "sha256_of_canonical_json",
    "payload": "accepted_assignment_indices",
    "variables_lsb_to_msb": list(FEATURE_NAMES),
    "assignment_indices": "0_through_255_inclusive",
}


def base_declaration(inventory: SurfaceInventory) -> dict[str, Any]:
    """Return the declaration of the unrestricted 256-language base class."""
    forms = inventory.forms
    return {
        "type": "closed_world_binary_product",
        "class_size": 256,
        "class_hash": HYPOTHESIS_CLASS_HASH,
        "lexical_axes": [
            {
                "name": "noun_pair_01",
                "category": "noun",
                "forms": [forms["noun_0"], forms["noun_1"]],
                "possible_meanings": ["fox", "owl"],
                "constraint": "either_bijection",
            },
            {
                "name": "noun_pair_23",
                "category": "noun",
                "forms": [forms["noun_2"], forms["noun_3"]],
                "possible_meanings": ["dog", "bear"],
                "constraint": "either_bijection",
            },
            {
                "name": "verb_pair_01",
                "category": "verb",
                "forms": [forms["verb_0"], forms["verb_1"]],
                "possible_meanings": ["see", "help"],
                "constraint": "either_bijection",
            },
            {
                "name": "modifier_pair_01",
                "category": "modifier",
                "forms": [forms["modifier_0"], forms["modifier_1"]],
                "possible_meanings": ["red", "blue"],
                "constraint": "either_bijection",
            },
        ],
        "fixed_lexicon": [
            {"category": "verb", "form": forms["verb_2"], "meaning": "chase"},
            {
                "category": "verb",
                "form": forms["verb_3"],
                "meaning": "sleep",
                "valency": "intransitive",
            },
            {"category": "modifier", "form": forms["modifier_2"], "meaning": "green"},
            {"category": "modifier", "form": forms["modifier_3"], "meaning": "small"},
        ],
        "grammar_axes": [
            {"name": "order_family", "values": ["subject_initial", "verb_initial"]},
            {"name": "order_variant", "values": ["variant_a", "variant_b"]},
            {"name": "alignment", "values": ["nominative_accusative", "ergative_absolutive"]},
            {"name": "agreement", "values": ["zero", "agent_number"]},
        ],
        "grammar_semantics": {
            "order_mapping": [
                {
                    "order_family": "subject_initial",
                    "order_variant": "variant_a",
                    "word_order": "SVO",
                },
                {
                    "order_family": "subject_initial",
                    "order_variant": "variant_b",
                    "word_order": "SOV",
                },
                {
                    "order_family": "verb_initial",
                    "order_variant": "variant_a",
                    "word_order": "VSO",
                },
                {
                    "order_family": "verb_initial",
                    "order_variant": "variant_b",
                    "word_order": "VOS",
                },
            ],
            "alignment_mapping": {
                "nominative_accusative": {
                    "transitive_agent_A": "case_slot_a",
                    "intransitive_subject_S": "case_slot_a",
                    "transitive_patient_O": "case_slot_b",
                },
                "ergative_absolutive": {
                    "transitive_agent_A": "case_slot_a",
                    "intransitive_subject_S": "case_slot_b",
                    "transitive_patient_O": "case_slot_b",
                },
            },
            "agreement_mapping": {
                "zero": "no_agreement_suffix",
                "agent_number": {
                    "singular": "agreement_singular",
                    "plural": "agreement_plural",
                },
            },
            "intransitive_order": {"subject_initial": "SV", "verb_initial": "VS"},
        },
        "constant_markers": {
            "plural_suffix": forms["plural"],
            "case_slot_a": forms["case_a"],
            "case_slot_b": forms["case_b"],
            "past_suffix": forms["past"],
            "preverbal_negation": forms["negation"],
            "agreement_singular": forms["agree_singular"],
            "agreement_plural": forms["agree_plural"],
        },
        "constant_rules": [
            "modifiers precede nouns",
            "plural and case are suffixes",
            "past precedes agreement on the verb",
        ],
    }


def program_declaration(inventory: SurfaceInventory, program: Mapping[str, Any]) -> dict[str, Any]:
    """Return the declaration of the 128 languages one Boolean program admits."""
    enumerated = enumerate_program(program)
    if len(enumerated.hypotheses) != DECLARED_CLASS_SIZE:
        raise AssertionError("a declared class must contain exactly 128 languages")
    declaration = base_declaration(inventory)
    declaration.update(
        {
            "type": "closed_world_posterior_matched_boolean_program_subset",
            "schema_family": SCHEMA_FAMILY,
            "base_class_size": 256,
            "base_class_hash": HYPOTHESIS_CLASS_HASH,
            "class_size": len(enumerated.hypotheses),
            "class_hash": enumerated.class_hash,
            "class_semantic_hash": class_semantic_hash(program),
            "class_semantic_hash_definition": CLASS_SEMANTIC_HASH_DEFINITION,
            "feature_variables_lsb_to_msb": [dict(item) for item in FEATURE_DECLARATION],
            "boolean_dsl_semantics": DSL_SEMANTICS,
            "boolean_program": enumerated.program,
            "boolean_program_hash": enumerated.program_hash,
            "exact_enumeration": {
                "universe": "all 2^8 assignments in feature_variables_lsb_to_msb order",
                "filter": "retain assignments satisfying every Boolean-program constraint",
                "allowlist_required": False,
            },
            "posterior_matching_contract": {
                "observed_feature_count": 3,
                "declared_class_size": DECLARED_CLASS_SIZE,
                "full_evidence_posterior_size": FULL_EVIDENCE_POSTERIOR_SIZE,
                "paired_query_requirement": (
                    "each evidence state is evaluated under both conditionally "
                    "constrained coordinates"
                ),
                "public_decision_field_present": False,
                "public_role_metadata_present": False,
            },
        }
    )
    return declaration


def declaration_hash(declaration: Mapping[str, Any]) -> str:
    """Hash one declaration."""
    return object_hash(dict(declaration))


def inventory_from_declaration(declaration: Mapping[str, Any]) -> SurfaceInventory:
    """Reconstruct every executable opaque form from public information only."""
    axes = {str(axis["name"]): axis for axis in declaration["lexical_axes"]}
    fixed = {
        (str(item["category"]), str(item["meaning"])): str(item["form"])
        for item in declaration["fixed_lexicon"]
    }
    markers = declaration["constant_markers"]
    forms = {
        "noun_0": axes["noun_pair_01"]["forms"][0],
        "noun_1": axes["noun_pair_01"]["forms"][1],
        "noun_2": axes["noun_pair_23"]["forms"][0],
        "noun_3": axes["noun_pair_23"]["forms"][1],
        "verb_0": axes["verb_pair_01"]["forms"][0],
        "verb_1": axes["verb_pair_01"]["forms"][1],
        "verb_2": fixed[("verb", "chase")],
        "verb_3": fixed[("verb", "sleep")],
        "modifier_0": axes["modifier_pair_01"]["forms"][0],
        "modifier_1": axes["modifier_pair_01"]["forms"][1],
        "modifier_2": fixed[("modifier", "green")],
        "modifier_3": fixed[("modifier", "small")],
        "plural": markers["plural_suffix"],
        "case_a": markers["case_slot_a"],
        "case_b": markers["case_slot_b"],
        "past": markers["past_suffix"],
        "negation": markers["preverbal_negation"],
        "agree_singular": markers["agreement_singular"],
        "agree_plural": markers["agreement_plural"],
    }
    return SurfaceInventory(seed="public-declaration", forms=forms)
