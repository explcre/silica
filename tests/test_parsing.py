"""Accept and reject cases of the strict response parser."""

import pytest

from silica.parsing import parse_response, score_response

QUERY = {"feature": "alignment"}


@pytest.mark.parametrize(
    ("text", "decision", "answer"),
    [
        (
            '{"decision": "ANSWER", "answer": "ergative_absolutive"}',
            "ANSWER",
            "ergative_absolutive",
        ),
        ('\n\n{"answer": null, "decision": "ABSTAIN"}\n', "ABSTAIN", None),
    ],
)
def test_accepts_exact_decision_objects(text, decision, answer):
    """A bare decision object, with surrounding whitespace, is accepted."""
    parsed = score_response(
        text, finish_reason="stop", request_error=False, task="grammar_feature", query=QUERY
    )
    assert (parsed.valid, parsed.decision, parsed.answer) == (True, decision, answer)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("The answer is ergative_absolutive.", "invalid_json"),
        ('```json\n{"decision": "ABSTAIN", "answer": null}\n```', "invalid_json"),
        ('Answer: {"decision": "ABSTAIN", "answer": null}', "invalid_json"),
        ("ABSTAIN", "invalid_json"),
        ("", "empty_response"),
        ('["ABSTAIN"]', "not_object"),
        ('{"decision": "ABSTAIN", "answer": null, "confidence": 0.9}', "wrong_keys"),
        ('{"decision": "ABSTAIN", "decision": "ANSWER", "answer": null}', "duplicate_key"),
        ('{"decision": "ANSWER", "answer": NaN}', "invalid_json"),
        ('{"decision": "abstain", "answer": null}', "invalid_decision"),
        ('{"decision": ["ABSTAIN"], "answer": null}', "invalid_decision"),
        ('{"decision": "ABSTAIN", "answer": "zero"}', "abstain_answer_must_be_null"),
        ('{"decision": "ANSWER", "answer": null}', "answer_must_not_be_null"),
        ('{"decision": "ANSWER", "answer": "  "}', "invalid_answer_type:grammar_feature"),
        (
            '{"decision": "ANSWER", "answer": {"value": "zero"}}',
            "invalid_answer_type:grammar_feature",
        ),
    ],
)
def test_rejects_everything_else(text, error):
    """Prose, fences, extra or duplicate keys, bare non-JSON and wrong types are invalid."""
    parsed = parse_response(text, "grammar_feature")
    assert (parsed.valid, parsed.error) == (False, error)


def test_scoring_rules_beyond_the_parser():
    """Out-of-domain answers, length stops and request errors are invalid."""
    good = '{"decision": "ANSWER", "answer": "ergative_absolutive"}'
    outside = '{"decision": "ANSWER", "answer": "zero"}'
    cases = [
        (outside, "stop", False, "answer_outside_query_domain"),
        (good, "length", False, "generation_did_not_stop"),
        (good, None, False, "generation_did_not_stop"),
        (None, None, True, "request_error"),
    ]
    for text, finish, request_error, error in cases:
        parsed = score_response(
            text,
            finish_reason=finish,
            request_error=request_error,
            task="grammar_feature",
            query=QUERY,
        )
        assert (parsed.valid, parsed.error) == (False, error)
    lexicon = score_response(
        '{"decision": "ANSWER", "answer": "owl"}',
        finish_reason="stop",
        request_error=False,
        task="lexicon",
        query={"category": "noun", "form": "kaso"},
    )
    assert lexicon.valid and lexicon.answer == "owl"
