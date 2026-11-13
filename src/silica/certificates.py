"""Exact answers, identifiability certificates and minimal proofs.

The gold label of a row is computed, not annotated: enumerate the languages a
row's program admits, keep those consistent with its observations, and answer
the query under each survivor.  ``ANSWER`` iff every survivor gives the same
canonical answer; otherwise ``ABSTAIN`` with a two-language disagreement
witness.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

from .dsl import class_semantic_hash, enumerate_program
from .evidence import DictionaryEvidence, evidence_hash, experiment_key, filter_hypotheses
from .hypotheses import Hypothesis
from .surface import SurfaceInventory, semantic_for_root
from .util import canonical_json, object_hash

TASKS = ("lexicon", "grammar_feature")


def answer_query(
    task: str, query: Mapping[str, Any], hypothesis: Hypothesis, inventory: SurfaceInventory
) -> Any:
    """Return the canonical answer of a query in one language."""
    if task == "lexicon":
        return semantic_for_root(str(query["category"]), str(query["form"]), hypothesis, inventory)
    if task == "grammar_feature":
        return hypothesis.feature_value(str(query["feature"]))
    raise ValueError(f"unknown task: {task!r}")


def make_certificate(
    task: str,
    query: Mapping[str, Any],
    evidence: Sequence[DictionaryEvidence],
    inventory: SurfaceInventory,
    program: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the identifiability certificate and the gold decision of one row."""
    enumerated = enumerate_program(program)
    survivors = filter_hypotheses(enumerated.hypotheses, evidence, inventory)
    if len(evidence) == 3 and len(survivors) != 16:
        raise AssertionError(f"a full-evidence cell has {len(survivors)} survivors, not 16")
    if not survivors:
        raise ValueError("evidence eliminates the declared class")
    groups: dict[str, list[Hypothesis]] = defaultdict(list)
    answers: dict[str, Any] = {}
    for hypothesis in survivors:
        answer = answer_query(task, query, hypothesis, inventory)
        key = canonical_json(answer)
        groups[key].append(hypothesis)
        answers[key] = answer
    support = [
        {
            "answer": answers[key],
            "count": len(groups[key]),
            "hypothesis_ids_hash": object_hash(sorted(item.id for item in groups[key])),
        }
        for key in sorted(groups)
    ]
    consensus = len(groups) == 1
    answer = support[0]["answer"] if consensus else None
    witness = None
    if not consensus:
        first, second = sorted(groups)[:2]
        witness = [
            {"hypothesis_id": groups[first][0].id, "answer": answers[first]},
            {"hypothesis_id": groups[second][0].id, "answer": answers[second]},
        ]
    certificate = {
        "hypothesis_class_size": len(enumerated.hypotheses),
        "hypothesis_class_hash": enumerated.class_hash,
        "class_semantic_hash": class_semantic_hash(program),
        "boolean_program_hash": enumerated.program_hash,
        "evidence_hash": evidence_hash(evidence),
        "ordered_experiment_hash": object_hash([experiment_key(item) for item in evidence]),
        "query_hash": object_hash({"task": task, "query": dict(query)}),
        "hypothesis_count": len(survivors),
        "version_space_hash": object_hash(sorted(item.id for item in survivors)),
        "support": support,
        "support_hash": object_hash(support),
        "consensus": consensus,
        "decision": "ANSWER" if consensus else "ABSTAIN",
        "answer_hash": object_hash(answer),
        "disagreement_witness": witness,
    }
    return certificate, {"decision": certificate["decision"], "answer": answer}


def certificate_answer(certificate: Mapping[str, Any]) -> Any:
    """Return the answer a well-formed certificate licenses, or None for ABSTAIN."""
    decision = certificate.get("decision")
    support = certificate.get("support")
    if decision == "ANSWER":
        if not isinstance(support, list) or len(support) != 1 or "answer" not in support[0]:
            raise ValueError("malformed ANSWER certificate")
        return support[0]["answer"]
    if decision == "ABSTAIN":
        if not isinstance(support, list) or len(support) < 2:
            raise ValueError("malformed ABSTAIN certificate")
        return None
    raise ValueError("malformed certificate decision")


def minimum_sufficient_evidence(
    task: str,
    query: Mapping[str, Any],
    evidence: Sequence[DictionaryEvidence],
    inventory: SurfaceInventory,
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the smallest observation subset that already fixes the gold result."""
    full_certificate, full_gold = make_certificate(task, query, evidence, inventory, program)
    candidates: list[tuple[int, ...]] = []
    subsets_checked = 0
    for size in range(len(evidence) + 1):
        for indices in combinations(range(len(evidence)), size):
            subsets_checked += 1
            subset = tuple(evidence[index] for index in indices)
            _, subset_gold = make_certificate(task, query, subset, inventory, program)
            if subset_gold == full_gold:
                candidates.append(indices)
        if candidates:
            break
    chosen = min(candidates)
    witness_atoms = 1 if full_certificate["consensus"] else 2
    return {
        "definition": (
            "smallest ordered-evidence subset whose exact decision and answer "
            "equal the full-evidence result"
        ),
        "minimum_sufficient_evidence_count": len(chosen),
        "minimum_sufficient_evidence_indices": list(chosen),
        "minimum_sufficient_experiment_keys": [experiment_key(evidence[index]) for index in chosen],
        "subsets_exhaustively_checked_until_minimum": subsets_checked,
        "full_evidence_count": len(evidence),
        "consensus_or_witness_atoms": witness_atoms,
        "minimum_proof_atom_count": len(chosen) + witness_atoms,
    }
