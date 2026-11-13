"""Canonical, executable Boolean constraint programs over SILICA features.

The language is deliberately small and finite.  A public declaration carries
the operator semantics, the ordered binary variables, and a canonical program,
so the admitted hypotheses can be enumerated exactly with no opaque allowlist.

An expression is a JSON object: ``{"op": "var", "name": ...}``,
``{"op": "const", "value": ...}``, ``{"op": "not", "arg": ...}``, or
``{"op": <and|or|xor|eq|implies|ite>, "args": [...]}``.  A program is
``{"dsl_version", "variables", "constraints"}`` and admits an assignment iff
every constraint is true.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from .hypotheses import FEATURE_NAMES, HYPOTHESIS_CLASS, Hypothesis
from .util import canonical_json, object_hash

DSL_VERSION = "silica-boolean-constraint-program-v1"
DSL_SEMANTICS: dict[str, Any] = {
    "domain": "Boolean values false/true, encoded by SILICA feature bits 0/1",
    "program_acceptance": "a hypothesis is admitted iff every constraint evaluates to true",
    "operators": {
        "var": "the Boolean bit of the named feature",
        "const": "the literal Boolean value",
        "not": "Boolean negation of arg",
        "and": "true iff every arg is true",
        "or": "true iff at least one arg is true",
        "xor": "true iff an odd number of args are true",
        "eq": "true iff its two args have the same Boolean value",
        "implies": "equivalent to (not first) or second",
        "ite": "second arg if first arg is true, otherwise third arg",
    },
    "canonicalization": {
        "object_keys": "lexicographic canonical JSON ordering",
        "constraint_order": "lexicographic canonical JSON ordering",
        "commutative_argument_order": "lexicographic canonical JSON ordering for and/or/xor/eq",
        "arities": {
            "not": 1,
            "and": "at least 2",
            "or": "at least 2",
            "xor": "at least 2",
            "eq": 2,
            "implies": 2,
            "ite": 3,
        },
    },
}
_COMMUTATIVE = frozenset({"and", "or", "xor", "eq"})
_FIXED_ARITY = {"eq": 2, "implies": 2, "ite": 3}


class DSLValidationError(ValueError):
    """Raised when a Boolean expression or program is not well formed."""


def _require_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    """Require an expression node to carry exactly the expected keys."""
    if set(value) != expected:
        raise DSLValidationError(
            f"{context} keys must be {sorted(expected)}, found {sorted(value)}"
        )


def canonicalize_expression(
    expression: Mapping[str, Any], *, allowed_variables: Sequence[str] = FEATURE_NAMES
) -> dict[str, Any]:
    """Validate one expression and return its canonical AST."""
    if not isinstance(expression, Mapping):
        raise DSLValidationError("an expression must be an object")
    operation = expression.get("op")
    if operation == "var":
        _require_keys(expression, {"op", "name"}, "var")
        name = expression["name"]
        if not isinstance(name, str) or name not in set(allowed_variables):
            raise DSLValidationError(f"unknown Boolean variable {name!r}")
        return {"op": "var", "name": name}
    if operation == "const":
        _require_keys(expression, {"op", "value"}, "const")
        if not isinstance(expression["value"], bool):
            raise DSLValidationError("const.value must be Boolean")
        return {"op": "const", "value": expression["value"]}
    if operation == "not":
        _require_keys(expression, {"op", "arg"}, "not")
        argument = canonicalize_expression(expression["arg"], allowed_variables=allowed_variables)
        return {"op": "not", "arg": argument}
    if operation not in {"and", "or", "xor", "eq", "implies", "ite"}:
        raise DSLValidationError(f"unknown Boolean operator {operation!r}")
    _require_keys(expression, {"op", "args"}, str(operation))
    raw_arguments = expression["args"]
    if not isinstance(raw_arguments, list):
        raise DSLValidationError(f"{operation}.args must be a list")
    arity = _FIXED_ARITY.get(str(operation))
    if arity is not None and len(raw_arguments) != arity:
        raise DSLValidationError(f"{operation} requires {arity} args")
    if arity is None and len(raw_arguments) < 2:
        raise DSLValidationError(f"{operation} requires at least two args")
    arguments = [
        canonicalize_expression(item, allowed_variables=allowed_variables) for item in raw_arguments
    ]
    if operation in _COMMUTATIVE:
        arguments.sort(key=canonical_json)
    return {"op": str(operation), "args": arguments}


def canonicalize_program(
    program: Mapping[str, Any], *, variables: Sequence[str] = FEATURE_NAMES
) -> dict[str, Any]:
    """Validate a complete program and return its canonical form."""
    if not isinstance(program, Mapping):
        raise DSLValidationError("a Boolean program must be an object")
    _require_keys(program, {"dsl_version", "variables", "constraints"}, "program")
    if program.get("dsl_version") != DSL_VERSION:
        raise DSLValidationError("Boolean DSL version mismatch")
    if program.get("variables") != list(variables):
        raise DSLValidationError("program variables must equal the declared ordered feature list")
    constraints = program.get("constraints")
    if not isinstance(constraints, list) or not constraints:
        raise DSLValidationError("program constraints must be a nonempty list")
    canonical = [canonicalize_expression(item, allowed_variables=variables) for item in constraints]
    canonical.sort(key=canonical_json)
    return {"dsl_version": DSL_VERSION, "variables": list(variables), "constraints": canonical}


def evaluate_expression(
    expression: Mapping[str, Any], assignment: Mapping[str, bool | int]
) -> bool:
    """Evaluate one canonical expression under a total assignment."""
    operation = str(expression["op"])
    if operation == "var":
        name = str(expression["name"])
        if name not in assignment or assignment[name] not in (False, True, 0, 1):
            raise DSLValidationError(f"missing/non-Boolean assignment for {name}")
        return bool(assignment[name])
    if operation == "const":
        return bool(expression["value"])
    if operation == "not":
        return not evaluate_expression(expression["arg"], assignment)
    arguments = [evaluate_expression(item, assignment) for item in expression["args"]]
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
    raise DSLValidationError(f"unknown Boolean operator {operation!r}")


def program_accepts(program: Mapping[str, Any], assignment: Mapping[str, bool | int]) -> bool:
    """Return whether every constraint of a canonical program is true."""
    return all(evaluate_expression(item, assignment) for item in program["constraints"])


def hypothesis_assignment(hypothesis: Hypothesis) -> dict[str, bool]:
    """Map a hypothesis to the DSL's named Boolean assignment."""
    return {feature: bool(bit) for feature, bit in zip(FEATURE_NAMES, hypothesis.bits, strict=True)}


def index_assignment(index: int, variables: Sequence[str]) -> dict[str, bool]:
    """Map an assignment index to named bits, least significant variable first."""
    return {name: bool((index >> offset) & 1) for offset, name in enumerate(variables)}


@cache
def _accepted_indices(program_json: str) -> tuple[int, ...]:
    """Enumerate the accepted base-class indices of one canonical program."""
    program = json.loads(program_json)
    return tuple(
        index
        for index in range(256)
        if program_accepts(program, index_assignment(index, FEATURE_NAMES))
    )


def accepted_indices(program: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the sorted hypothesis indices a program admits."""
    return _accepted_indices(canonical_json(canonicalize_program(program)))


def class_semantic_hash(program: Mapping[str, Any]) -> str:
    """Hash the exact set of admitted hypotheses, independent of program spelling."""
    return object_hash(list(accepted_indices(program)))


def _expression_metrics(expression: Mapping[str, Any]) -> tuple[int, int, set[str]]:
    """Return depth, operator count and variable support of one expression."""
    operation = str(expression["op"])
    if operation == "var":
        return 0, 0, {str(expression["name"])}
    if operation == "const":
        return 0, 0, set()
    children = [expression["arg"]] if operation == "not" else list(expression["args"])
    child_metrics = [_expression_metrics(child) for child in children]
    support: set[str] = set()
    for _, _, child_support in child_metrics:
        support.update(child_support)
    depth = 1 + max(item[0] for item in child_metrics)
    return depth, 1 + sum(item[1] for item in child_metrics), support


def program_metrics(program: Mapping[str, Any]) -> dict[str, Any]:
    """Compute the structural complexity metrics published with every row."""
    canonical = canonicalize_program(program)
    metrics = [_expression_metrics(item) for item in canonical["constraints"]]
    support = sorted({name for _, _, names in metrics for name in names})
    positions = [FEATURE_NAMES.index(name) for name in support]
    return {
        "constraint_count": len(metrics),
        "maximum_dependency_depth": max(item[0] for item in metrics),
        "operator_count": sum(item[1] for item in metrics),
        "variable_support": support,
        "variable_support_size": len(support),
        "dependency_span": max(positions) - min(positions) if positions else 0,
        "proof_length_definition": "operator_count in a full canonical-AST evaluation",
        "full_evaluation_proof_length": sum(item[1] for item in metrics),
    }


@dataclass(frozen=True)
class EnumeratedProgram:
    """A canonical program bound to its exactly enumerated hypotheses."""

    program: dict[str, Any]
    hypotheses: tuple[Hypothesis, ...]
    program_hash: str
    class_hash: str


@cache
def _enumerate(program_json: str) -> EnumeratedProgram:
    """Enumerate and hash one canonical program given as canonical JSON."""
    canonical = json.loads(program_json)
    hypotheses = tuple(
        item for item in HYPOTHESIS_CLASS if program_accepts(canonical, hypothesis_assignment(item))
    )
    if not hypotheses:
        raise DSLValidationError("a declared program admits no hypotheses")
    program_hash = object_hash(canonical)
    class_hash = object_hash(
        {
            "dsl_version": DSL_VERSION,
            "program_hash": program_hash,
            "hypotheses": [item.to_dict() for item in hypotheses],
        }
    )
    return EnumeratedProgram(canonical, hypotheses, program_hash, class_hash)


def enumerate_program(program: Mapping[str, Any]) -> EnumeratedProgram:
    """Canonicalize, enumerate, and bind a program to its exact class hash."""
    return _enumerate(canonical_json(canonicalize_program(program)))
