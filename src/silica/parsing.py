"""The strict response parser.

A response is scored only if its final content is exactly one JSON object
``{"decision": "ANSWER", "answer": <string>}`` or
``{"decision": "ABSTAIN", "answer": null}``.  Nothing is recovered: prose,
code fences, extra or duplicate keys, non-finite numbers, empty strings and
answers outside the query's declared domain are all invalid, and so is any
generation that did not stop on its own.  Reasoning traces are never parsed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .certificates import TASKS
from .prompts import answer_domain
from .util import DuplicateKeyError, strict_loads

DECISIONS = ("ANSWER", "ABSTAIN")


@dataclass(frozen=True)
class ParsedResponse:
    """The outcome of parsing one response."""

    valid: bool
    decision: str | None = None
    answer: Any = None
    error: str | None = None


def parse_response(text: str | None, task: str) -> ParsedResponse:
    """Parse one final-content string as a strict decision object."""
    if not isinstance(text, str) or not text.strip():
        return ParsedResponse(False, error="empty_response")
    try:
        value = strict_loads(text)
    except DuplicateKeyError:
        return ParsedResponse(False, error="duplicate_key")
    except (ValueError, TypeError, RecursionError):
        return ParsedResponse(False, error="invalid_json")
    if not isinstance(value, dict):
        return ParsedResponse(False, error="not_object")
    if set(value) != {"decision", "answer"}:
        return ParsedResponse(False, error="wrong_keys")
    decision, answer = value["decision"], value["answer"]
    if not isinstance(decision, str) or decision not in DECISIONS:
        return ParsedResponse(False, error="invalid_decision")
    if decision == "ABSTAIN":
        if answer is not None:
            return ParsedResponse(False, error="abstain_answer_must_be_null")
        return ParsedResponse(True, decision="ABSTAIN")
    if answer is None:
        return ParsedResponse(False, error="answer_must_not_be_null")
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}")
    if not isinstance(answer, str) or not answer.strip():
        return ParsedResponse(False, error=f"invalid_answer_type:{task}")
    return ParsedResponse(True, decision="ANSWER", answer=answer)


def score_response(
    text: str | None,
    *,
    finish_reason: str | None,
    request_error: bool,
    task: str,
    query: Mapping[str, Any],
) -> ParsedResponse:
    """Apply the full scoring rule: request errors, stop reason, parse, answer domain."""
    if request_error:
        return ParsedResponse(False, error="request_error")
    if finish_reason != "stop":
        return ParsedResponse(False, error="generation_did_not_stop")
    parsed = parse_response(text, task)
    if (
        parsed.valid
        and parsed.decision == "ANSWER"
        and parsed.answer not in answer_domain(task, query)
    ):
        return ParsedResponse(False, error="answer_outside_query_domain")
    return parsed
