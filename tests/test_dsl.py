"""Semantics and canonical form of the Boolean DSL."""

from itertools import product

import pytest

from silica.dsl import (
    DSL_VERSION,
    DSLValidationError,
    accepted_indices,
    canonicalize_expression,
    canonicalize_program,
    enumerate_program,
    evaluate_expression,
    program_metrics,
)
from silica.hypotheses import FEATURE_NAMES

A = {"op": "var", "name": "noun_swap_01"}
B = {"op": "var", "name": "order_family"}
C = {"op": "var", "name": "agreement"}
NAMES = ("noun_swap_01", "order_family", "agreement")


def program(*constraints):
    """Wrap constraints in a program over the release features."""
    return {
        "dsl_version": DSL_VERSION,
        "variables": list(FEATURE_NAMES),
        "constraints": list(constraints),
    }


@pytest.mark.parametrize(
    ("operation", "arity", "truth"),
    [
        ("and", 2, lambda a, b: a and b),
        ("or", 2, lambda a, b: a or b),
        ("xor", 2, lambda a, b: a != b),
        ("eq", 2, lambda a, b: a == b),
        ("implies", 2, lambda a, b: (not a) or b),
        ("ite", 3, lambda a, b, c: b if a else c),
    ],
)
def test_operator_semantics(operation, arity, truth):
    """Every operator follows its documented truth table."""
    expression = {"op": operation, "args": [A, B, C][:arity]}
    for bits in product((False, True), repeat=arity):
        assignment = dict(zip(NAMES, (*bits, False, False)[:3], strict=True))
        assert evaluate_expression(expression, assignment) == truth(*bits)


def test_negation_constant_and_odd_parity():
    """Negation, constants and three-way xor behave as declared."""
    assert evaluate_expression({"op": "not", "arg": {"op": "const", "value": False}}, {}) is True
    parity = {"op": "xor", "args": [A, B, C]}
    for bits in product((0, 1), repeat=3):
        assignment = dict(zip(NAMES, bits, strict=True))
        assert evaluate_expression(parity, assignment) == (sum(bits) % 2 == 1)


def test_canonical_form_sorts_commutative_arguments_and_constraints():
    """Commutative arguments and constraints are ordered canonically; implies is not."""
    assert canonicalize_expression({"op": "and", "args": [B, A]}) == canonicalize_expression(
        {"op": "and", "args": [A, B]}
    )
    implication = canonicalize_expression({"op": "implies", "args": [B, A]})
    assert implication["args"] == [B, A]
    assert canonicalize_program(program(B, A)) == canonicalize_program(program(A, B))


@pytest.mark.parametrize(
    "expression",
    [
        {"op": "nand", "args": [A, B]},
        {"op": "eq", "args": [A, B, C]},
        {"op": "and", "args": [A]},
        {"op": "var", "name": "unknown"},
        {"op": "const", "value": 1},
        {"op": "not", "arg": A, "extra": True},
    ],
)
def test_malformed_expressions_are_rejected(expression):
    """Unknown operators, wrong arities, unknown variables and extra keys are errors."""
    with pytest.raises(DSLValidationError):
        canonicalize_expression(expression)


def test_enumeration_and_metrics():
    """A single equality constraint admits exactly half of the 256 languages."""
    equality = program({"op": "eq", "args": [A, B]})
    accepted = accepted_indices(equality)
    assert len(accepted) == 128
    assert all(((index >> 0) & 1) == ((index >> 4) & 1) for index in accepted)
    assert len(enumerate_program(equality).hypotheses) == 128
    metrics = program_metrics(equality)
    assert metrics["operator_count"] == 1
    assert metrics["variable_support"] == ["noun_swap_01", "order_family"]
    assert metrics["dependency_span"] == 4
