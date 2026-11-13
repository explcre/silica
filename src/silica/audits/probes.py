"""Learned shortcut probes on observation-masked rows.

Every public row is serialized with identifiers, hashes and routing fields
removed and with every observed form replaced by a mask; the query and the
published program remain.  Word and character TF-IDF logistic regressions are
trained under held-out-template and held-out-family splits.  Because the two
queries of a state see byte-identical masked state, no probe can beat chance:
each must score exactly one half.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..util import canonical_json

LABEL = {"ABSTAIN": 0, "ANSWER": 1}
MASKED_FORM = "__MASKED_OBSERVED_FORM__"


def _is_identifier_or_route(key: str) -> bool:
    """Return whether a key names an identifier, hash, route or semantic hash."""
    lowered = key.lower()
    return (
        lowered in {"id", "item_index", "route", "seed", "language_id"}
        or lowered.endswith("_id")
        or lowered.endswith("_index")
        or "hash" in lowered
        or "semantic" in lowered
    )


def _sanitize(value: Any) -> Any:
    """Drop identifier, hash and routing keys at every depth."""
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(child)
            for key, child in sorted(value.items())
            if not _is_identifier_or_route(str(key))
        }
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    return value


def masked_text(public: Mapping[str, Any]) -> str:
    """Serialize a public row with identifiers removed and observed forms masked."""
    sanitized = _sanitize(public)
    evidence = public.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError("expected exactly three evidence records")
    sanitized["evidence"] = [
        {
            "channel": item["channel"],
            "category": item["category"],
            "semantic": item["semantic"],
            "form": MASKED_FORM,
        }
        for item in evidence
    ]
    return canonical_json(sanitized)


def _fit(
    texts: Sequence[str],
    labels: Sequence[int],
    groups: Sequence[str],
    units: Sequence[str],
    *,
    analyzer: str,
    scheme: str,
) -> dict[str, Any]:
    """Return grouped out-of-fold accuracy of one TF-IDF logistic-regression probe."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline

    splitter = GroupKFold(n_splits=8) if scheme == "heldout_template" else LeaveOneGroupOut()
    folds = list(splitter.split(list(range(len(labels))), labels, groups))
    predictions: dict[int, int] = {}
    for train, test in folds:
        if analyzer == "word":
            vectorizer = TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=50_000,
            )
        model = make_pipeline(
            vectorizer,
            LogisticRegression(C=1.0, max_iter=1000, random_state=20260803, solver="liblinear"),
        )
        model.fit([texts[index] for index in train], [labels[index] for index in train])
        for index, prediction in zip(
            test, model.predict([texts[index] for index in test]), strict=True
        ):
            if index in predictions:
                raise AssertionError("a grouped probe predicted a row twice")
            predictions[int(index)] = int(prediction)
    if len(predictions) != len(labels):
        raise AssertionError("a grouped probe did not cover every row")
    correct = sum(predictions[index] == labels[index] for index in predictions)
    per_unit: dict[str, set[int]] = defaultdict(set)
    for index, prediction in predictions.items():
        per_unit[units[index]].add(prediction)
    if any(len(values) != 1 for values in per_unit.values()):
        raise AssertionError("identical observation-masked rows received different predictions")
    return {
        "analyzer": analyzer,
        "grouping": scheme,
        "folds": len(folds),
        "correct": correct,
        "total": len(labels),
        "balanced_decision_accuracy": correct / len(labels),
        "exactly_one_half": 2 * correct == len(labels),
        "complete_units_with_one_prediction": len(per_unit),
    }


def learned_probes(
    public_rows: Sequence[Mapping[str, Any]], gold_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Fit the four grouped probes and report each one's out-of-fold accuracy."""
    texts = [masked_text(row) for row in public_rows]
    labels = [LABEL[str(gold_by_id[str(row["id"])]["gold"]["decision"])] for row in public_rows]
    templates = [str(row["mechanism"]["template_id"]) for row in public_rows]
    families = [str(row["mechanism"]["family"]) for row in public_rows]
    units = [str(row["complete_unit_id"]) for row in public_rows]
    if Counter(labels)[0] != Counter(labels)[1]:
        raise ValueError("labels are not exactly balanced")
    probes = {
        f"{analyzer}__{scheme}": _fit(
            texts, labels, groups, units, analyzer=analyzer, scheme=scheme
        )
        for analyzer in ("word", "character")
        for scheme, groups in (("heldout_template", templates), ("heldout_family", families))
    }
    return {
        "rows": len(labels),
        "primary_generalization_clusters": len(set(templates)),
        "mechanism_families": len(set(families)),
        "four_cell_evaluator_units": len(set(units)),
        "mask": {
            "observation_forms": MASKED_FORM,
            "query_retained": True,
            "public_program_retained": True,
            "identifiers_hashes_routes_removed": True,
        },
        "probes": probes,
    }
