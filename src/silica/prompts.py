"""Rendering of the exact chat messages the evaluated models received.

Every row becomes one system message and one user message.  The user message
is the canonical JSON of the public declaration fields, the observations, the
query, the decision rule, and the row's response JSON schema.  Nothing
evaluator-only is rendered: rows carrying a gold, certificate or oracle field
are rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._frozen import ADAPTER_VERSION, PROMPT_VERSION, RESPONSE_SCHEMA_VERSION, SYSTEM_PROMPT, TRACK
from .cohort import ADAPTER_PROTOCOL_HASH
from .generator import forbidden_paths
from .hypotheses import FEATURE_DECLARATION, GRAMMAR_FEATURES, LEXICON
from .util import canonical_json, object_hash

# Declaration fields a model sees; hashes and evaluator bookkeeping are omitted.
DECLARATION_FIELDS = (
    "type",
    "schema_family",
    "base_class_size",
    "class_size",
    "boolean_dsl_semantics",
    "boolean_program",
    "feature_variables_lsb_to_msb",
    "exact_enumeration",
    "lexical_axes",
    "fixed_lexicon",
    "grammar_axes",
    "grammar_semantics",
    "constant_markers",
    "constant_rules",
)
ROUTING_FIELDS = (
    "complete_unit_id",
    "counterfactual_cell_id",
    "paired_query_unit_id",
    "paired_state_id",
)
DECISION_RULE = {
    "ANSWER": "every observation-compatible hypothesis has the same canonical query answer",
    "ABSTAIN": (
        "at least two observation-compatible hypotheses have different canonical query answers"
    ),
}
_GRAMMAR_VALUES = {
    item["name"]: sorted(item["values"])
    for item in FEATURE_DECLARATION
    if item["name"] in GRAMMAR_FEATURES
}


def answer_domain(task: str, query: Mapping[str, Any]) -> list[str]:
    """Return the canonical answers a row's query admits."""
    values = list(query.values()) if isinstance(query, Mapping) else []
    if values and all(isinstance(value, str) and value for value in values):
        if (
            task == "lexicon"
            and set(query) == {"category", "form"}
            and query["category"] in LEXICON
        ):
            return list(LEXICON[query["category"]])
        if (
            task == "grammar_feature"
            and set(query) == {"feature"}
            and query["feature"] in _GRAMMAR_VALUES
        ):
            return list(_GRAMMAR_VALUES[query["feature"]])
    raise ValueError(f"unsupported {task} query: {query!r}")


def response_schema(task: str, query: Mapping[str, Any]) -> dict[str, Any]:
    """Return the JSON schema of a valid response to one row."""
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["ANSWER"]},
                    "answer": {"type": "string", "enum": answer_domain(task, query)},
                },
                "required": ["decision", "answer"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["ABSTAIN"]},
                    "answer": {"type": "null"},
                },
                "required": ["decision", "answer"],
                "additionalProperties": False,
            },
        ],
    }


@dataclass(frozen=True)
class Prompt:
    """The chat messages of one row and their hash."""

    messages: tuple[dict[str, str], ...]
    prompt_hash: str

    def as_messages(self) -> list[dict[str, str]]:
        """Return the messages as fresh dictionaries."""
        return [dict(message) for message in self.messages]


def render_prompt(item: Mapping[str, Any]) -> Prompt:
    """Render the system and user messages of one public row."""
    leaked = forbidden_paths(item)
    if leaked:
        raise ValueError(f"model-facing row contains evaluator-only field {leaked[0]}")
    if item.get("track") != TRACK:
        raise ValueError("the prompt requires a row of the release track")
    for name in ROUTING_FIELDS:
        if not isinstance(item.get(name), str) or not item[name]:
            raise ValueError(f"public row lacks {name}")
    declaration = item["hypothesis_declaration"]
    missing = [name for name in DECLARATION_FIELDS if name not in declaration]
    if missing:
        raise ValueError("declaration lacks semantic fields: " + ", ".join(missing))
    evidence = item["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("observations must be a non-empty array")
    observations = []
    for index, observation in enumerate(evidence):
        fields = ("channel", "category", "semantic", "form")
        if not isinstance(observation, Mapping) or any(
            not isinstance(observation.get(name), str) or not observation[name] for name in fields
        ):
            raise ValueError(f"observation {index} is incomplete")
        observations.append({name: observation[name] for name in fields})
    task = str(item["task"])
    payload = {
        "prompt_version": PROMPT_VERSION,
        "adapter_task": TRACK,
        "response_task": task,
        "adapter_version": ADAPTER_VERSION,
        "adapter_protocol_hash": ADAPTER_PROTOCOL_HASH,
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "hypothesis_declaration": {name: declaration[name] for name in DECLARATION_FIELDS},
        "observations": observations,
        "query": item["query"],
        "decision_rule": DECISION_RULE,
        "canonical_response_schema": response_schema(task, item["query"]),
    }
    messages = (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    )
    return Prompt(messages, object_hash(list(messages)))


def prompt_ledger_identity(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash every prompt keyed by its stable semantic row key, in key order."""
    keyed = sorted(
        (
            [
                str(item["paired_query_unit_id"]),
                str(item["paired_state_id"]),
                str(item["query_id"]),
            ],
            render_prompt(item).prompt_hash,
        )
        for item in items
    )
    return object_hash([[key, prompt_hash] for key, prompt_hash in keyed])


def request_prompt_manifest(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash the row IDs and prompt hashes of a request list, in request order."""
    return object_hash([[str(item["id"]), render_prompt(item).prompt_hash] for item in items])
