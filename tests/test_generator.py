"""Determinism and structural guarantees of the generator."""

from collections import Counter, defaultdict

from conftest import TEMPLATES

from silica.generator import PUBLIC_FORBIDDEN_KEYS, build_records, forbidden_paths
from silica.templates import template_registry
from silica.util import canonical_json


def test_registry_scale():
    """The registry fixes 12 support templates and packs 48 evaluation templates per family."""
    registry = template_registry()
    assert len(registry.support) == 12
    assert Counter(item.family for item in registry.evaluation) == {
        "parity_guard": 48,
        "majority_guard": 48,
        "equality_chain": 48,
        "selector_multiplexer": 48,
    }


def test_generation_is_deterministic(small_release):
    """Rebuilding the same templates with the same seed reproduces every byte."""
    public, gold_by_id = small_release
    again_public, again_gold = build_records(template_ids=TEMPLATES)
    assert [canonical_json(row) for row in again_public] == [canonical_json(row) for row in public]
    assert [canonical_json(row) for row in again_gold] == [
        canonical_json(gold_by_id[row["id"]]) for row in public
    ]


def test_seed_changes_every_identifier(small_release):
    """A different seed yields different languages and identifiers."""
    public, _ = small_release
    other, _ = build_records(seed="another-seed", template_ids=TEMPLATES[:1])
    assert not {row["id"] for row in other} & {row["id"] for row in public}


def test_paired_structure(small_release):
    """Each paired unit has eight rows, four states, and balanced decisions."""
    public, gold_by_id = small_release
    assert len(public) == len(TEMPLATES) * 8 * 8
    paired, units = defaultdict(list), defaultdict(list)
    for row in public:
        decision = gold_by_id[row["id"]]["gold"]["decision"]
        paired[row["paired_query_unit_id"]].append((row["paired_state_id"], decision))
        units[row["complete_unit_id"]].append(decision)
        assert row["hypothesis_declaration"]["class_size"] == 128
        assert gold_by_id[row["id"]]["certificate"]["hypothesis_count"] == 16
        assert not forbidden_paths(row) and not PUBLIC_FORBIDDEN_KEYS & set(row)
    for rows in paired.values():
        assert len(rows) == 8 and len({state for state, _ in rows}) == 4
        assert Counter(decision for _, decision in rows) == {"ANSWER": 4, "ABSTAIN": 4}
    assert all(Counter(decisions) == {"ANSWER": 2, "ABSTAIN": 2} for decisions in units.values())
