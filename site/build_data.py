#!/usr/bin/env python3
"""Build the data files of the SILICA project page from the released dataset.

Every number the page shows is computed here, with the ``silica`` package,
from the released public rows, gold rows and model responses, and formatted
the way the paper prints it.  Each value is then compared with
``scripts/paper_values.json`` (the values printed in the paper); on any
difference the script lists it, writes nothing, and exits with status 1.

Usage, from a clone of the repository with the dataset downloaded::

    python site/build_data.py --hf silica-data

It writes ``meta.json``, ``example.json``, ``construction.json`` and
``results.json`` to ``site/static/data``.  The full run recomputes every
certificate, both oracles, the prompt-only solver, all shortcut audits and
the paper's tables with 10,000-replicate intervals; it needs the ``audit``
extra and takes about twenty-five minutes on eight cores.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import silica
from silica.acceptance import acceptance_audit
from silica.audits.probes import learned_probes
from silica.audits.solver import prompt_sufficiency
from silica.audits.structure import census, guard_probes
from silica.certificates import answer_query, make_certificate
from silica.dataset import load_gold_by_id, load_public_rows
from silica.declaration import (
    DECLARED_CLASS_SIZE,
    FULL_EVIDENCE_POSTERIOR_SIZE,
    inventory_from_declaration,
)
from silica.dsl import accepted_indices
from silica.evaluate import BOOTSTRAP_REPLICATES, score_rows
from silica.evidence import evidence_from_dict, is_consistent
from silica.hypotheses import FEATURE_NAMES, Hypothesis
from silica.predictions import load_predictions
from silica.prompts import render_prompt
from silica.util import object_hash, read_json
from silica.validate import validate_dataset

REPO = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "static" / "data"
VALUES = REPO / "scripts" / "paper_values.json"
OPERATORS = {"and": " ∧ ", "or": " ∨ ", "xor": " ⊕ ", "eq": " = ", "implies": " ⇒ "}
# Matched controls (Table 3): one checkpoint, one factor switched off and on.
SWITCHED = (
    ("qwen3-235b-trace-off", "qwen3-235b-trace-on"),
    ("qwen3-17b-trace-off", "qwen3-17b-trace-on"),
    ("qwen25-14b-ablation-verbatim", "qwen25-14b-ablation-scratchpad"),
    ("qwen25-32b-ablation-verbatim", "qwen25-32b-ablation-scratchpad"),
)
DECODING = ("qwen35-2b-greedy", "qwen35-2b-stochastic")
# Gridlines of the page's unit-scale axes (rates and accuracies).
UNIT_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_script(directory: Path, name: str) -> ModuleType:
    """Import a script of this repository that is not part of the package."""
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module(name)


def count(value: int) -> str:
    """Format an exact count with thousands separators, e.g. '12,288'."""
    return f"{value:,}"


def number(text: str | None) -> float | None:
    """Parse a value printed by the paper ('.423', '+.008', '1.000', '---') into a float."""
    if text is None or text == "---":
        return None
    return float(text)


def ratio(text: str) -> float:
    """Parse a printed exact count such as '11/192' into a proportion."""
    numerator, denominator = text.replace(",", "").split("/")
    return int(numerator) / int(denominator)


def tick(value: float) -> str:
    """Label an axis tick of a unit scale the way the paper prints rates: '0', '.25', '1'."""
    return f"{value:g}".removeprefix("0") or "0"


def fraction_text(value: Fraction) -> str:
    """Write a simple fraction as 'n/d', or as an integer when it is one."""
    return str(value.numerator) if value.denominator == 1 else f"{value}"


def infix(expression: Mapping[str, Any], names: Sequence[str], top: bool = True) -> str:
    """Render a Boolean DSL expression in infix form with bit names b0..b7."""
    operation = str(expression["op"])
    if operation == "var":
        return f"b{names.index(str(expression['name']))}"
    if operation == "const":
        return "1" if expression["value"] else "0"
    if operation == "not":
        return "¬" + infix(expression["arg"], names, top=False)
    arguments = [infix(item, names, top=operation == "ite") for item in expression["args"]]
    if operation == "ite":
        return "ITE(" + ", ".join(arguments) + ")"
    body = OPERATORS[operation].join(arguments)
    return body if top else f"({body})"


# --- the worked example (Figure 1) -------------------------------------------------


def partial(expression: Mapping[str, Any], known: Mapping[str, bool]) -> bool | None:
    """Evaluate an expression when only some bits are known; None while its value is open."""
    operation = str(expression["op"])
    if operation == "var":
        return known.get(str(expression["name"]))
    if operation == "const":
        return bool(expression["value"])
    if operation == "not":
        value = partial(expression["arg"], known)
        return None if value is None else not value
    values = [partial(item, known) for item in expression["args"]]
    if operation == "and":
        return False if False in values else None if None in values else True
    if operation == "or":
        return True if True in values else None if None in values else False
    if operation == "implies":
        if values[0] is False or values[1] is True:
            return True
        return None if None in values else False
    if operation == "ite":
        if values[0] is None:
            return values[1] if values[1] == values[2] else None
        return values[1] if values[0] else values[2]
    if None in values:
        return None
    if operation == "xor":
        return sum(values) % 2 == 1
    return values[0] == values[1]


def substitute(expression: Mapping[str, Any], known: Mapping[str, bool]) -> dict[str, Any]:
    """Replace every known bit of an expression by its value."""
    operation = str(expression["op"])
    if operation == "var":
        name = str(expression["name"])
        return {"op": "const", "value": known[name]} if name in known else dict(expression)
    if operation == "const":
        return dict(expression)
    if operation == "not":
        return {"op": "not", "arg": substitute(expression["arg"], known)}
    return {"op": operation, "args": [substitute(item, known) for item in expression["args"]]}


def gloss(expression: Mapping[str, Any], names: Sequence[str], top: bool = True) -> str:
    """Say an expression in plain English: odd number on, is the same as, not, and, if/then."""
    operation = str(expression["op"])
    if operation == "var":
        return f"b{names.index(str(expression['name']))}"
    if operation == "const":
        return "1" if expression["value"] else "0"
    if operation == "not":
        return "not " + gloss(expression["arg"], names, top=False)
    parts = [gloss(item, names, top=False) for item in expression["args"]]
    if operation == "ite":
        return f"(if {parts[0]} then {parts[1]}, otherwise {parts[2]})"
    if operation == "xor":
        text = f"an odd number of {', '.join(parts)} are on"
    elif operation == "implies":
        text = f"if {parts[0]}, then {parts[1]}"
    else:
        text = {"and": " and ", "or": " or ", "eq": " is the same as "}[operation].join(parts)
    return text if top else f"({text})"


def switches(declaration: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the declared switches: lexical pairs and grammar settings, bit by bit."""
    features = declaration["feature_variables_lsb_to_msb"]
    sizes = {len(item["values"]) for item in features}
    if len(sizes) != 1:
        raise ValueError("the declared features do not all have the same number of values")
    lexical = [item for item in features if item["kind"] == "lexical"]
    for feature, axis in zip(lexical, declaration["lexical_axes"], strict=True):
        if feature["name"].split("_")[::2] != axis["name"].split("_")[::2]:
            raise ValueError(f"lexical axis {axis['name']} does not bind to {feature['name']}")
    grammar = {axis["name"]: axis["values"] for axis in declaration["grammar_axes"]}
    bit = {item["name"]: f"b{index}" for index, item in enumerate(features)}
    return {
        "count": str(len(features)),
        "values_per_switch": str(sizes.pop()),
        "languages": count(declaration["base_class_size"]),
        "lexical": [
            {
                "bit": bit[feature["name"]],
                "feature": feature["name"],
                "category": axis["category"],
                "forms": axis["forms"],
                "meanings": axis["possible_meanings"],
            }
            for feature, axis in zip(lexical, declaration["lexical_axes"], strict=True)
        ],
        "grammar": [
            {"bit": bit[item["name"]], "feature": item["name"], "values": grammar[item["name"]]}
            for item in features
            if item["kind"] == "grammar"
        ],
    }


def deduction(
    declaration: Mapping[str, Any],
    pins: Sequence[tuple[str, int]],
    survivors: Sequence[Sequence[int]],
    queries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Derive the example's decisions from the program and the facts, without enumeration.

    The program is a guarded switch: one rule applies when the guard G is true and
    the other when it is false.  The facts fix G, so exactly one rule fires and forces
    its bit; the other rule says nothing and its bit stays free.  Every step is checked
    against the enumerated survivors.
    """
    names = list(FEATURE_NAMES)
    constraints = [
        part
        for item in declaration["boolean_program"]["constraints"]
        for part in (item["args"] if item["op"] == "and" else [item])
    ]
    if len(constraints) != 2 or any(item["op"] != "implies" for item in constraints):
        raise ValueError("the example program is not two implications")
    first, second = (item["args"][0] for item in constraints)
    if first == {"op": "not", "arg": second}:
        guard = second
    elif second == {"op": "not", "arg": first}:
        guard = first
    else:
        raise ValueError("the example program is not a guarded switch")
    known = {feature: bool(value) for feature, value in pins}
    guard_value = partial(guard, known)
    if guard_value is None:
        raise ValueError("the facts do not decide the guard")
    rules = []
    for constraint in constraints:
        premise, conclusion = constraint["args"]
        free = [
            item
            for item in conclusion.get("args", [])
            if item["op"] == "var" and item["name"] not in known
        ]
        if conclusion["op"] != "eq" or len(free) != 1:
            raise ValueError("a rule's conclusion is not 'bit = expression'")
        target = free[0]
        (expression,) = [item for item in conclusion["args"] if item is not target]
        fires = partial(premise, known)
        step = {
            "when_guard": premise == guard,
            "fires": fires,
            "symbolic": infix(constraint, names),
            "target": gloss(target, names),
            "feature": target["name"],
            "expression": gloss(expression, names),
            "plugged": gloss(substitute(expression, known), names),
            "value": partial(expression, known),
        }
        if expression["op"] == "ite" and fires:
            condition, then, otherwise = expression["args"]
            chosen = then if partial(condition, known) else otherwise
            step["condition"] = {
                "bit": gloss(condition, names),
                "value": str(int(partial(condition, known))),
            }
            step["chosen"] = {
                "bit": gloss(chosen, names),
                "value": str(int(partial(chosen, known))),
            }
        rules.append(step)
    firing = [rule for rule in rules if rule["fires"]]
    silent = [rule for rule in rules if rule["fires"] is False]
    if len(firing) != 1 or len(silent) != 1 or firing[0]["value"] is None:
        raise ValueError("the facts do not make exactly one rule fire and force its bit")
    forced, free = firing[0], silent[0]
    forced_column = [bits[names.index(forced["feature"])] for bits in survivors]
    free_column = Counter(bits[names.index(free["feature"])] for bits in survivors)
    if set(forced_column) != {int(forced["value"])}:
        raise ValueError("the forced bit is not constant over the compatible languages")
    lexical, grammar = queries
    if (lexical["feature"], grammar["feature"]) != (forced["feature"], free["feature"]):
        raise ValueError("the forced and free bits are not the two queried features")
    described = switches(declaration)
    axis = next(item for item in described["lexical"] if item["feature"] == forced["feature"])
    free_values = next(
        item["values"] for item in described["grammar"] if item["feature"] == free["feature"]
    )
    position = axis["forms"].index(lexical["query"]["form"])
    meaning = axis["meanings"][position if not forced_column[0] else 1 - position]
    if meaning != lexical["gold"]["answer"]:
        raise ValueError("the forced bit does not give the gold payload")
    support = sorted(item["count"] for item in grammar["support"])
    if sorted(free_column.values()) != support or len(free_column) != 2:
        raise ValueError("the free bit does not split the languages like the gold support")
    conjuncts = guard["args"] if guard["op"] == "and" else [guard]
    data = {
        "guard": {
            "symbolic": infix(guard, names),
            "plain": gloss(guard, names),
            "value": guard_value,
            "parts": [
                {
                    "symbolic": infix(part, names),
                    "plain": gloss(part, names),
                    "plugged": gloss(substitute(part, known), names),
                    "on": (
                        str(sum(bool(partial(item, known)) for item in part["args"]))
                        if part["op"] == "xor"
                        else None
                    ),
                    "of": str(len(part["args"])) if part["op"] == "xor" else None,
                    "value": partial(part, known),
                }
                for part in conjuncts
            ],
        },
        "rules": rules,
        "facts": [
            {"bit": f"b{names.index(feature)}", "feature": feature, "value": str(value)}
            for feature, value in pins
        ],
        "forced": {
            "bit": forced["target"],
            "value": str(int(forced["value"])),
            "form": lexical["query"]["form"],
            "meaning": meaning,
            "support": lexical["support_text"],
        },
        "free": {
            "bit": free["target"],
            "feature": free["feature"],
            "split": {value: count(free_column[index]) for index, value in enumerate(free_values)},
        },
    }
    printed = {
        "guard_fires": str(guard_value).lower(),
        "constant_feature": forced["feature"],
        "constant_column": forced["target"],
        "constant_value": data["forced"]["value"],
        "free_feature": free["feature"],
        "split_column": free["target"],
    }
    return data, printed


def pinned_bit(fact: Any, universe: Sequence[Hypothesis], inventory: Any) -> tuple[str, int]:
    """Return the one feature bit an observation fixes over the whole base class."""
    consistent = [item for item in universe if is_consistent(item, fact, inventory)]
    fixed = [name for name in FEATURE_NAMES if len({item.bit(name) for item in consistent}) == 1]
    if len(fixed) != 1:
        raise ValueError(f"observation {fact.form} fixes {len(fixed)} feature bits, not one")
    return fixed[0], consistent[0].bit(fixed[0])


def query_view(row: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize one query of the pair: its prompt fields and its evaluator-only results."""
    certificate = gold["certificate"]
    support = [
        {"answer": item["answer"], "count": item["count"]} for item in certificate["support"]
    ]
    witnesses = certificate["disagreement_witness"] or []
    prompt = render_prompt(row)
    feature = gold["oracle"]["query_feature"]
    return {
        "id": row["id"],
        "task": row["task"],
        "query": row["query"],
        "feature": feature,
        "bit": f"b{FEATURE_NAMES.index(feature)}",
        "prompt_hash": prompt.prompt_hash,
        "messages": prompt.as_messages(),
        "gold": {"decision": gold["gold"]["decision"], "answer": gold["gold"]["answer"]},
        "support": support,
        "support_text": ", ".join(
            f"{item['answer']} {item['count']}/{certificate['hypothesis_count']}"
            for item in support
        ),
        "witnesses": [item["hypothesis_id"] for item in witnesses],
    }


def example_responses(
    hf: Path,
    rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return every released model response to the example rows, scored."""
    conditions = read_json(hf / "predictions" / "conditions.json")["conditions"]
    wanted = {str(row["id"]): row for row in rows}
    found = []
    for condition in conditions:
        predictions = [
            item
            for item in load_predictions(hf / condition["predictions_file"])
            if item.id in wanted
        ]
        if not predictions:
            continue
        scored = {
            entry["item_id"]: entry
            for entry in score_rows(
                [wanted[item.id] for item in predictions], gold_by_id, predictions
            )
        }
        found.append(
            {
                "condition_id": condition["condition_id"],
                "model": condition["model"],
                "rows": count(condition["rows"]),
                "responses": [
                    {
                        "id": item.id,
                        "response_text": item.response_text,
                        "finish_reason": item.finish_reason,
                        "completion_tokens": item.completion_tokens,
                        "format_valid": scored[item.id]["format_valid"],
                        "decision": scored[item.id]["pred_decision"],
                        "correct": scored[item.id]["joint_correct"],
                    }
                    for item in predictions
                ],
            }
        )
    return {"conditions_released": count(len(conditions)), "covering": found}


def route_group(
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    group_id: str,
) -> dict[str, Any]:
    """Lay out one eight-row route group: query roles by evidence states."""
    rows = [row for row in public_rows if row["paired_query_unit_id"] == group_id]
    oracle = gold_by_id[str(rows[0]["id"])]["oracle"]
    roles = [oracle["role_zero_feature"], oracle["role_one_feature"]]
    states = sorted(
        {str(row["paired_state_id"]) for row in rows},
        key=lambda state: next(
            gold_by_id[str(row["id"])]["oracle"]["public_state_index"]
            for row in rows
            if row["paired_state_id"] == state
        ),
    )
    cells: list[list[dict[str, Any]]] = []
    for role in roles:
        line = []
        for state in states:
            (row,) = [
                item
                for item in rows
                if item["paired_state_id"] == state
                and gold_by_id[str(item["id"])]["oracle"]["query_feature"] == role
            ]
            gold = gold_by_id[str(row["id"])]
            line.append(
                {
                    "id": row["id"],
                    "unit": row["complete_unit_id"],
                    "bits": gold["oracle"]["observed_feature_bits"],
                    "decision": gold["gold"]["decision"],
                    "answer": gold["gold"]["answer"],
                }
            )
        if len({cell["unit"] for cell in line}) != 1:
            raise ValueError("a query role of the route group spans several four-cell units")
        cells.append(line)
    for column in zip(*cells, strict=True):
        if sorted(cell["decision"] for cell in column) != ["ABSTAIN", "ANSWER"]:
            raise ValueError("a same-state pair does not carry opposite gold actions")
    return {
        "id": group_id,
        "rows": count(len(rows)),
        "roles": roles,
        "observed_features": sorted(oracle["observed_feature_bits"]),
        "cells": cells,
    }


def worked_example(
    hf: Path,
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute Figure 1 cell by cell; return the page data and the printed values."""
    by_id = {str(row["id"]): row for row in public_rows}
    answer, abstain = by_id[spec["answer_item"]["value"]], by_id[spec["abstain_item"]["value"]]
    for field in ("paired_state_id", "hypothesis_declaration", "evidence"):
        if answer[field] != abstain[field]:
            raise ValueError(f"the example pair differs in {field}")
    declaration = answer["hypothesis_declaration"]
    names = [item["name"] for item in declaration["feature_variables_lsb_to_msb"]]
    if names != list(FEATURE_NAMES):
        raise ValueError("the declared feature order differs from the package's")
    inventory = inventory_from_declaration(declaration)
    facts = [evidence_from_dict(item) for item in answer["evidence"]]
    admitted = set(accepted_indices(declaration["boolean_program"]))
    size = 1 << len(names)
    universe = [
        Hypothesis(tuple((index >> bit) & 1 for bit in range(len(names)))) for index in range(size)
    ]

    def last_step(index: int) -> int:
        """Return the last elimination step a language survives (0 = the universe)."""
        if index not in admitted:
            return 0
        held = 0
        while held < len(facts) and is_consistent(universe[index], facts[held], inventory):
            held += 1
        return 1 + held

    steps = [last_step(index) for index in range(size)]
    counts = [sum(step >= stage for step in steps) for stage in range(len(facts) + 2)]
    survivors = [index for index in range(size) if steps[index] == len(facts) + 1]
    rows = (answer, abstain)
    for row in rows:
        certificate, _ = make_certificate(
            str(row["task"]), row["query"], tuple(facts), inventory, declaration["boolean_program"]
        )
        if certificate != gold_by_id[str(row["id"])]["certificate"]:
            raise ValueError(f"{row['id']}: recomputed certificate differs from the release")
        if (
            object_hash(sorted(universe[i].id for i in survivors))
            != certificate["version_space_hash"]
        ):
            raise ValueError(f"{row['id']}: survivors differ from the sealed version space")
    readout = {
        index: [
            answer_query(str(row["task"]), row["query"], universe[index], inventory) for row in rows
        ]
        for index in survivors
    }
    pins = [pinned_bit(fact, universe, inventory) for fact in facts]
    queries = [query_view(row, gold_by_id[str(row["id"])]) for row in rows]
    split = Counter(values[1] for values in readout.values())
    side = int(size**0.5)
    walkthrough, derived = deduction(
        declaration, pins, [universe[index].bits for index in survivors], queries
    )
    described = switches(declaration)
    options = [
        next(
            item["meanings"] for item in described["lexical"] if item["feature"] == query["feature"]
        )
        if query["task"] == "lexicon"
        else next(
            item["values"] for item in described["grammar"] if item["feature"] == query["feature"]
        )
        for query in queries
    ]
    printed = {
        "answer_item": answer["id"],
        "abstain_item": abstain["id"],
        "paired_state": answer["paired_state_id"],
        "answer_prompt_sha256": queries[0]["prompt_hash"],
        "abstain_prompt_sha256": queries[1]["prompt_hash"],
        "user_message_characters": count(len(queries[0]["messages"][1]["content"])),
        "family": answer["mechanism"]["family"],
        "elimination": ", ".join(map(str, counts)),
        "facts": ", ".join(f"{fact.form}->{fact.semantic}" for fact in facts),
        "pinned_bits": ", ".join(str(bit) for _, bit in pins),
        "lexical_query_form": answer["query"]["form"],
        "lexical_support": queries[0]["support_text"],
        "grammar_query_feature": abstain["query"]["feature"],
        "grammar_split": "/".join(str(value) for value in sorted(split.values())),
        "witnesses": ", ".join(queries[1]["witnesses"]),
        "answer_gold": queries[0]["gold"],
        "abstain_gold": queries[1]["gold"],
        **derived,
    }
    data = {
        "grid": {"side": str(side), "size": count(size), "features": str(len(names))},
        "features": [
            {"bit": f"b{index}", "name": item["name"], "kind": item["kind"]}
            for index, item in enumerate(declaration["feature_variables_lsb_to_msb"])
        ],
        "program": {
            "infix": [infix(item, names) for item in declaration["boolean_program"]["constraints"]],
            "json": declaration["boolean_program"],
            "operators": str(answer["structural_complexity"]["operator_count"]),
        },
        "facts": [
            {
                "form": fact.form,
                "category": fact.category,
                "semantic": fact.semantic,
                "feature": feature,
                "bit": f"b{names.index(feature)}",
                "value": str(value),
            }
            for fact, (feature, value) in zip(facts, pins, strict=True)
        ],
        "counts": [count(value) for value in counts],
        "cells": [
            {
                "h": universe[index].id,
                "bits": list(universe[index].bits),
                "last": steps[index],
                "answers": readout.get(index),
            }
            for index in range(size)
        ],
        "survivors": [
            {"h": universe[index].id, "bits": list(universe[index].bits), "answers": readout[index]}
            for index in survivors
        ],
        "split": {key: count(value) for key, value in sorted(split.items())},
        "switches": described,
        "walkthrough": walkthrough,
        "quiz": [{"options": choices} for choices in options],
        "queries": queries,
        "public_rows": [answer, abstain],
        "gold_rows": [gold_by_id[str(row["id"])] for row in rows],
        "route_group": route_group(public_rows, gold_by_id, str(answer["paired_query_unit_id"])),
        "responses": example_responses(hf, rows, gold_by_id),
    }
    return data, printed


# --- construction and audits (Tables 1 and 2, Appendix I) ---------------------------


def each_equal(reports: Mapping[str, Mapping[str, Any]]) -> tuple[int, int]:
    """Return the shared correct/total of several audit reports, or fail if they differ."""
    pairs = {(item["correct"], item["total"]) for item in reports.values()}
    if len(pairs) != 1:
        raise ValueError(f"audit reports disagree: {sorted(pairs)}")
    return pairs.pop()


def construction(
    hf: Path,
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    cleanroom: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the scale hierarchy, the release gates and the decision structure."""
    validation = validate_dataset(hf)
    validation.require_ok()
    gold_rows = [gold_by_id[str(row["id"])] for row in public_rows]
    audit = acceptance_audit(public_rows, gold_rows)
    clean = cleanroom.audit(hf)
    solver = prompt_sufficiency(public_rows, gold_by_id)
    probes = learned_probes(public_rows, gold_by_id)
    records = [(row, gold_by_id[str(row["id"])]) for row in public_rows]
    structure = census(records)
    guards = guard_probes(records)
    total = len(public_rows)
    packing = audit["template_semantic_packing"]
    distances = packing["paired_behavior_distances"]
    clean_packing = clean["template_semantic_packing"]
    if (
        distances["evaluation_evaluation"]["minimum"],
        distances["evaluation_support"]["minimum"],
    ) != (
        clean_packing["minimum_paired_evaluation_distance"],
        clean_packing["minimum_paired_support_distance"],
    ):
        raise ValueError("the clean-room packing distances differ from the package's")
    minimum = min(value for key, value in clean_packing.items() if key.startswith("minimum_"))
    length = packing["algorithm"]["paired_behavior_vector_length"]
    decisions = audit["decision_counts"]
    pairs = audit["matched_unit_balance"]
    families = Counter(str(row["mechanism"]["family"]) for row in public_rows)
    templates = {str(row["mechanism"]["template_id"]) for row in public_rows}
    per_family = Counter(name.rsplit("-eval-", 1)[0] for name in templates)
    minimal = structure["minimal_decision_formula_size"]
    operators = structure["published_program_operator_count"]
    blind = {three(item["balanced_decision_accuracy"]) for item in guards["guard_blind"].values()}
    given = {three(item["balanced_decision_accuracy"]) for item in guards["guard_given"].values()}
    if len(blind) != 1 or len(given) != 1:
        raise ValueError("a guard probe differs between its two splits")
    admitted = solver["admitted_class_size_histogram"]
    posterior = solver["posterior_size_histogram"]
    printed = {
        "records": count(total),
        "same_state_query_pairs": count(len({row["paired_state_id"] for row in public_rows})),
        "four_cell_units": count(len({row["complete_unit_id"] for row in public_rows})),
        "route_groups": count(len({row["paired_query_unit_id"] for row in public_rows})),
        "templates": count(len(templates)),
        "mechanism_families": str(len(families)),
        "templates_per_family": "/".join(sorted({str(value) for value in per_family.values()})),
        "support_templates": str(packing["selection"]["support_count"]),
        "binary_features": str(len(FEATURE_NAMES)),
        "base_class_size": count(1 << len(FEATURE_NAMES)),
        "declared_class_size": "/".join(map(str, admitted)),
        "posterior_size": "/".join(map(str, posterior)),
        "gate.in_generator_oracle": fraction(
            audit["exact_oracle"]["correct"], audit["exact_oracle"]["total"]
        ),
        "gate.cleanroom_oracle": fraction(
            clean["oracle_joint_accuracy"]["correct"], clean["oracle_joint_accuracy"]["total"]
        ),
        "gate.prompt_only_solver": fraction(solver["strict_joint_correct"], solver["scored_rows"]),
        "gate.declared_class_size": fraction(admitted.get(DECLARED_CLASS_SIZE, 0), total),
        "gate.posterior_size": fraction(posterior.get(FULL_EVIDENCE_POSTERIOR_SIZE, 0), total),
        "gate.gold_action_balance": fraction(decisions["ANSWER"], decisions["ABSTAIN"]),
        "gate.same_state_opposite_action_pairs": fraction(
            pairs["complementary_same_state_query_pairs"],
            pairs["expected_complementary_same_state_query_pairs"],
        ),
        "gate.deterministic_shortcuts": fraction(*each_equal(audit["shortcut_probes"])),
        "gate.learned_probes": fraction(*each_equal(probes["probes"])) + " each",
        "gate.unique_prompt_hashes": fraction(
            len({render_prompt(row).prompt_hash for row in public_rows}), total
        ),
        "gate.minimum_signature_distance": f"{minimum} of {length} positions",
        "structure.programs": count(structure["program_count"]),
        "structure.decision_functions": str(structure["distinct_guard_functions"]),
        "structure.stated_procedure_median": count(
            structure["stated_procedure_operator_evaluations"]["median"]
        ),
        "structure.operator_count_median": str(operators["median"]),
        "structure.operator_count_range": f"{operators['min']}-{operators['max']}",
        "structure.minimal_formula_median": str(minimal["median"]),
        "structure.minimal_formula_max": str(minimal["max"]),
        "structure.programs_needing_no_operation": str(minimal["histogram"].get(0, 0)),
        "structure.guard_blind_probe": blind.pop(),
        "structure.guard_given_probe": given.pop(),
    }
    data = {
        "printed": printed,
        "families": sorted(families),
        "shortcut_controls": sorted(audit["shortcut_probes"]),
        "learned_probes": sorted(probes["probes"]),
        "minimal_formula_histogram": {
            str(key): count(value) for key, value in minimal["histogram"].items()
        },
        "dataset_sha256": read_json(hf / "manifest.json")["dataset_sha256"],
    }
    return data, printed


# --- model results (Tables 3, 8 and 9, Section 6.2, Appendix J) --------------------


def three(value: float | None) -> str:
    """Format a rate as the paper prints it (delegates to the reproduction script)."""
    return REPRODUCE.three(value)


def fraction(numerator: int, denominator: int) -> str:
    """Format an exact count as the paper prints it, e.g. '11/192'."""
    return REPRODUCE.fraction(numerator, denominator)


def table_rows(table: Mapping[str, Any], parse: Callable[[str], Any]) -> dict[str, Any]:
    """Attach parsed values to every printed row of a reproduced table."""
    return {
        key: {
            "label": row["label"],
            "values": row["values"],
            "parsed": [parse(v) for v in row["values"]],
        }
        for key, row in table["rows"].items()
    }


def cell_value(text: str | None) -> Any:
    """Parse a printed table cell: a rate, a count such as '11/192', a mean or an interval."""
    if text is None or text == "---":
        return None
    if text.startswith("["):
        return [float(part) for part in text.strip("[]").split(", ")]
    if "/" in text:
        return ratio(text)
    return float(text.replace(",", ""))


def constant_policy(panel: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    """Tabulate a query-blind policy that answers at rate alpha, from the two constant rows.

    Every row metric is linear in the per-row outcomes, so the expected value of
    a policy answering each row with probability alpha is the alpha-mixture of
    the always-ANSWER and always-ABSTAIN rows of the reproduced Table 9.
    """
    answer, abstain = panel["always-answer"]["values"], panel["always-abstain"]["values"]
    shown = ("AR", "FCR", "BDA", "BJA")
    index = {name: columns.index(name) for name in shown}

    def mixed(name: str, alpha: Fraction) -> str:
        """Return one metric of the alpha-mixture, rounded half up as printed."""
        value = alpha * Fraction(answer[index[name]]) + (1 - alpha) * Fraction(abstain[index[name]])
        exact = Decimal(value.numerator) / Decimal(value.denominator)
        return three(float(exact.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)))

    grid = [
        {
            "alpha": f"{step / 100:.2f}",
            **{name: mixed(name, Fraction(step, 100)) for name in shown},
        }
        for step in range(101)
    ]
    intercept = Fraction(abstain[index["BJA"]])
    slope = Fraction(answer[index["BJA"]]) - intercept
    if slope.numerator != -1:
        raise ValueError("the BJA of a constant policy is not intercept - alpha/n")
    return {
        "grid": grid,
        "bja_formula": f"BJA = {fraction_text(intercept)} − α/{slope.denominator}",
        "bda_formula": f"BDA = {fraction_text(Fraction(abstain[index['BDA']]))}",
    }


def results(hf: Path, paper: Mapping[str, Any], jobs: int) -> tuple[dict[str, Any], list[str]]:
    """Reproduce the paper's tables and key numbers and shape them for the page."""
    reproduced, mismatches = REPRODUCE.reproduce(hf, paper, jobs)
    expected = {
        "tables": paper["tables"],
        "numbers": {key: item["value"] for key, item in paper["numbers"].items()},
    }
    REPRODUCE.compare(expected, reproduced, "paper", mismatches)
    tables = reproduced["tables"]
    numbers = reproduced["numbers"]
    panel = tables["model_panel"]
    columns = panel["columns"]
    valid = columns.index("Valid")
    gate = REPRODUCE.FORMAT_GATE
    arms = list(REPRODUCE.ARMS)
    ablations = {}
    for model in REPRODUCE.ABLATED:
        if model == "qwen25-14b":
            rows = tables["prompt_ablations"]["rows"]
            recall = [rows[f"{model}-ablation-{arm}"]["values"][1] for arm in arms]
            signal = [rows[f"{model}-ablation-{arm}"]["values"][2] for arm in arms]
        else:
            recall = numbers[f"ablation.{model}.answerable_decision_recall"].split(", ")
            signal = numbers[f"ablation.{model}.decision_signal"].split(", ")
        ablations[model] = {
            "label": panel["rows"][model]["label"],
            "arms": [
                {
                    "arm": arm,
                    "AR": ar,
                    "signal": sig,
                    "x": round(float(ar) - float(sig), 3),
                    "y": float(ar),
                }
                for arm, ar, sig in zip(arms, recall, signal, strict=True)
            ],
            "recall_range": numbers[f"ablation.{model}.answerable_decision_recall_range"],
        }
    data = {
        "panel": {
            "title": panel["title"],
            "columns": columns,
            "conditions": list(REPRODUCE.PANEL),
            "references": ["always-answer", "always-abstain", "exact-oracle"],
            "rows": table_rows(panel, cell_value),
            "sub_gate": [
                key for key in REPRODUCE.PANEL if float(panel["rows"][key]["values"][valid]) < gate
            ],
            "gate_family": os.path.commonprefix(
                [
                    panel["rows"][key]["label"]
                    for key in REPRODUCE.PANEL
                    if float(panel["rows"][key]["values"][valid]) >= gate
                ]
            ).rstrip("-"),
        },
        "matched": {
            "title": tables["matched_controls"]["title"],
            "columns": tables["matched_controls"]["columns"],
            "rows": table_rows(tables["matched_controls"], cell_value),
            "models": {
                key: row["label"].split(" ")[0]
                for key, row in tables["matched_controls"]["rows"].items()
            },
            "switched": [
                {
                    "model": tables["matched_controls"]["rows"][off]["label"].split(" ")[0],
                    "off": off,
                    "on": on,
                }
                for off, on in SWITCHED
            ],
            "decoding": list(DECODING),
        },
        "ablations": {
            "title": tables["prompt_ablations"]["title"],
            "columns": tables["prompt_ablations"]["columns"],
            "rows": table_rows(tables["prompt_ablations"], cell_value),
            "models": ablations,
        },
        "constant_policy": constant_policy(panel["rows"], columns),
        "axes": {"unit": [{"value": value, "text": tick(value)} for value in UNIT_TICKS]},
        "numbers": numbers,
    }
    return data, mismatches


# --- page metadata -------------------------------------------------------------------


def readme_blocks(readme: str) -> dict[str, list[str]]:
    """Return the fenced code blocks of the README, grouped by their section heading."""
    blocks: dict[str, list[str]] = {}
    heading = ""
    for part in re.split(r"(^## [^\n]*$|^```[a-z]*\n.*?^```$)", readme, flags=re.M | re.S):
        if part.startswith("## "):
            heading = part[3:].strip().lower()
        elif part.startswith("```"):
            body = part.split("\n", 1)[1].rsplit("```", 1)[0].rstrip("\n")
            blocks.setdefault(heading, []).append(body)
    return blocks


def meta(paper: Mapping[str, Any], protocol: Mapping[str, str], hf: Path) -> dict[str, Any]:
    """Return the title, venue, citation, commands and protocol constants of the page."""
    blocks = readme_blocks((REPO / "README.md").read_text(encoding="utf-8"))
    title, venue = re.fullmatch(r"(.*) \((.*)\), camera-ready text\..*", paper["source"]).groups()
    code_license = re.search(
        r'^license = "([^"]+)"$', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M
    ).group(1)
    card = (hf / "README.md").read_text(encoding="utf-8")
    family, holder, version = re.search(r"^license: (\w+)-(\w+)-([\d.]+)$", card, re.M).groups()
    return {
        "title": title,
        "venue": venue,
        "version": silica.__version__,
        "licenses": {"code": code_license, "data": f"{family.upper()} {holder.upper()} {version}"},
        "bibtex": blocks["citation"][0],
        "install": blocks["install"][0],
        "quickstart": blocks["quickstart"][0],
        "reproduce": blocks["reproduce the paper"],
        "protocol": protocol,
        "conditions": count(len(read_json(hf / "predictions" / "conditions.json")["conditions"])),
    }


def protocol_values(hf: Path) -> dict[str, str]:
    """Read the evaluation protocol constants from the released condition settings."""
    conditions = {
        item["condition_id"]: item
        for item in read_json(hf / "predictions" / "conditions.json")["conditions"]
    }
    budgets = {
        key: {conditions[name]["decoding"]["max_new_tokens"] for name in names}
        for key, names in {
            "panel_max_new_tokens": REPRODUCE.PANEL,
            "trace_token_budget": [name for pair in SWITCHED[:2] for name in pair],
            "scratchpad_token_budget": [pair[1] for pair in SWITCHED[2:]],
        }.items()
    }
    if any(len(values) != 1 for values in budgets.values()):
        raise ValueError(f"token budgets differ within a condition group: {budgets}")
    common = {conditions[name]["rows"] for name in REPRODUCE.PANEL}
    return {
        "panel_conditions": str(len(REPRODUCE.PANEL)),
        "common_cohort_rows": count(common.pop()),
        **{key: count(values.pop()) for key, values in budgets.items()},
        "format_gate": f"{REPRODUCE.FORMAT_GATE:.2f}".removeprefix("0"),
        "bootstrap_replicates": count(BOOTSTRAP_REPLICATES),
    }


def check(section: str, expected: Mapping[str, Any], printed: Mapping[str, Any]) -> list[str]:
    """Compare one section of computed values with the values printed in the paper."""
    if set(expected) != set(printed):
        missing, extra = sorted(set(expected) - set(printed)), sorted(set(printed) - set(expected))
        return [f"{section}: keys differ (missing {missing}, extra {extra})"]
    return [
        f"{section}.{key}: paper {expected[key]['value']!r}, computed {printed[key]!r}"
        for key in expected
        if expected[key]["value"] != printed[key]
    ]


def write(directory: Path, name: str, value: Any) -> None:
    """Write one data file as UTF-8 JSON."""
    path = directory / f"{name}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


REPRODUCE = load_script(REPO / "scripts", "reproduce_paper")


def main() -> int:
    """Compute every page value, check it against the paper, and write the data files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf", type=Path, required=True, help="downloaded dataset directory")
    parser.add_argument("--values", type=Path, default=VALUES, help="paper values to compare with")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="directory of the data files")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    paper = json.loads(args.values.read_text(encoding="utf-8"))
    cleanroom = load_script(REPO / "audits", "cleanroom_oracle")
    public_rows = load_public_rows(args.hf)
    gold_by_id = load_gold_by_id(args.hf)
    example, example_printed = worked_example(
        args.hf, public_rows, gold_by_id, paper["worked_example"]
    )
    built, built_printed = construction(args.hf, public_rows, gold_by_id, cleanroom)
    protocol = protocol_values(args.hf)
    page_results, mismatches = results(args.hf, paper, args.jobs)
    mismatches += check("worked_example", paper["worked_example"], example_printed)
    mismatches += check("construction", paper["construction"], built_printed)
    mismatches += check("protocol", paper["protocol"], protocol)
    if mismatches:
        for line in mismatches:
            print(f"MISMATCH {line}")
        print(f"{len(mismatches)} values differ from the paper; nothing written")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output, "meta", meta(paper, protocol, args.hf))
    write(args.output, "example", example)
    write(args.output, "construction", built)
    write(args.output, "results", page_results)
    print("every page value matches the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
