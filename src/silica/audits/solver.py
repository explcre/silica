"""A solver that reads only the rendered prompt.

The oracles read dataset records.  This solver answers a different question:
does the text a model receives suffice?  It renders each row, discards the
row, and works only from the JSON user message: it re-derives the feature
inventory and the lexical-axis bindings, enumerates the declared universe,
applies the published program, filters by the observations, and evaluates
the query under every survivor.  Gold labels are read only to score it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..dsl import index_assignment, program_accepts
from ..prompts import render_prompt


class PromptOnlySolverError(RuntimeError):
    """Raised when a rendered prompt does not license a unique derivation."""


def _lexical_meaning(axis: Mapping[str, Any], form: str, swapped: int) -> str:
    """Return the meaning a lexical axis assigns to a form under one swap bit."""
    forms, meanings = axis["forms"], axis["possible_meanings"]
    if len(forms) != 2 or len(meanings) != 2:
        raise PromptOnlySolverError("lexical axis is not a declared pair")
    index = forms.index(form)
    return meanings[index] if not swapped else meanings[1 - index]


def _axis_bindings(declaration: Mapping[str, Any]) -> dict[str, str]:
    """Bind each lexical axis to its swap bit by matching category and index names."""
    features = declaration["feature_variables_lsb_to_msb"]
    lexical_bits = [item for item in features if item.get("kind") == "lexical"]
    axes = declaration["lexical_axes"]
    if len(lexical_bits) != len(axes):
        raise PromptOnlySolverError("lexical axes and lexical bits do not correspond")
    bindings = {}
    for axis, feature in zip(axes, lexical_bits, strict=True):
        axis_parts, bit_parts = axis["name"].split("_"), feature["name"].split("_")
        if (
            len(axis_parts) != 3
            or len(bit_parts) != 3
            or (axis_parts[0], axis_parts[2])
            != (
                bit_parts[0],
                bit_parts[2],
            )
        ):
            raise PromptOnlySolverError(
                f"binding {axis['name']} -> {feature['name']} is not name-consistent"
            )
        bindings[axis["name"]] = feature["name"]
    return bindings


def solve_from_prompt(user_message: str) -> dict[str, Any]:
    """Derive the decision and answer from one rendered user message."""
    payload = json.loads(user_message)
    declaration = payload["hypothesis_declaration"]
    names = [item["name"] for item in declaration["feature_variables_lsb_to_msb"]]
    if len(names) != len(set(names)) or declaration["base_class_size"] != 1 << len(names):
        raise PromptOnlySolverError("the feature inventory disagrees with the declared base class")
    bindings = _axis_bindings(declaration)
    grammar_axes = {axis["name"]: axis for axis in declaration["grammar_axes"]}
    grammar_bits = {
        item["name"]
        for item in declaration["feature_variables_lsb_to_msb"]
        if item.get("kind") == "grammar"
    }
    if grammar_bits != set(grammar_axes):
        raise PromptOnlySolverError("grammar axes and grammar bits do not correspond")
    form_to_axis = {form: axis for axis in declaration["lexical_axes"] for form in axis["forms"]}
    program = declaration["boolean_program"]
    admitted = [
        assignment
        for assignment in (index_assignment(index, names) for index in range(1 << len(names)))
        if program_accepts(program, assignment)
    ]
    if len(admitted) != declaration["class_size"]:
        raise PromptOnlySolverError(
            f"program admits {len(admitted)} hypotheses, declared {declaration['class_size']}"
        )

    def meaning(form: str, bits: Mapping[str, bool]) -> str:
        """Return the meaning of a declared form in one admitted language."""
        axis = form_to_axis.get(form)
        if axis is None:
            raise PromptOnlySolverError(f"form {form} is not on any declared axis")
        return _lexical_meaning(axis, form, int(bits[bindings[axis["name"]]]))

    survivors = [
        bits
        for bits in admitted
        if all(meaning(item["form"], bits) == item["semantic"] for item in payload["observations"])
    ]
    if not survivors:
        raise PromptOnlySolverError("observations admit no hypothesis")
    query, task = payload["query"], payload["response_task"]
    if task == "lexicon":
        answers = {meaning(query["form"], bits) for bits in survivors}
    elif task == "grammar_feature":
        axis = grammar_axes.get(query["feature"])
        if axis is None:
            raise PromptOnlySolverError(f"queried grammar feature {query['feature']} is undeclared")
        answers = {axis["values"][int(bits[query["feature"]])] for bits in survivors}
    else:
        raise PromptOnlySolverError(f"unknown response task {task}")
    unique = len(answers) == 1
    return {
        "decision": "ANSWER" if unique else "ABSTAIN",
        "answer": answers.pop() if unique else None,
        "survivors": len(survivors),
        "admitted": len(admitted),
    }


def prompt_sufficiency(
    items: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Solve every row from its rendered user message and score against gold."""
    decision_correct = joint_correct = 0
    survivors: Counter[int] = Counter()
    admitted: Counter[int] = Counter()
    failures: list[dict[str, Any]] = []
    for item in items:
        user = [
            message["content"]
            for message in render_prompt(item).messages
            if message["role"] == "user"
        ]
        if len(user) != 1:
            raise PromptOnlySolverError("expected exactly one user message")
        derived = solve_from_prompt(user[0])
        truth = gold_by_id[str(item["id"])]["gold"]
        survivors[derived["survivors"]] += 1
        admitted[derived["admitted"]] += 1
        decision_correct += derived["decision"] == truth["decision"]
        exact = derived["decision"] == truth["decision"] and derived["answer"] == truth.get(
            "answer"
        )
        joint_correct += exact
        if not exact and len(failures) < 5:
            failures.append({"id": item["id"], "derived": derived, "gold": truth})
    return {
        "scored_rows": len(items),
        "decision_correct": decision_correct,
        "strict_joint_correct": joint_correct,
        "decision_accuracy": decision_correct / len(items),
        "strict_joint_accuracy": joint_correct / len(items),
        "admitted_class_size_histogram": dict(sorted(admitted.items())),
        "posterior_size_histogram": dict(sorted(survivors.items())),
        "first_failures": failures,
        "inputs_used": "rendered user message only; no record, certificate, or oracle",
    }
