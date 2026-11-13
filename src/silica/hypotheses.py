"""The explicitly declared base class of 2^8 SILICA languages.

A hypothesis is eight binary involutions: four lexical swaps of form/meaning
pairs and four grammar coordinates.  Bit ``i`` of the integer hypothesis index
is the value of feature ``FEATURE_NAMES[i]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from .util import object_hash

NOUNS = ("fox", "owl", "dog", "bear")
VERBS = ("see", "help", "chase", "sleep")
MODIFIERS = ("red", "blue", "green", "small")
LEXICON = {"noun": NOUNS, "verb": VERBS, "modifier": MODIFIERS}

FEATURE_DECLARATION: tuple[dict[str, Any], ...] = (
    {"name": "noun_swap_01", "kind": "lexical", "values": ["identity", "swapped"]},
    {"name": "noun_swap_23", "kind": "lexical", "values": ["identity", "swapped"]},
    {"name": "verb_swap_01", "kind": "lexical", "values": ["identity", "swapped"]},
    {"name": "modifier_swap_01", "kind": "lexical", "values": ["identity", "swapped"]},
    {
        "name": "order_family",
        "kind": "grammar",
        "values": ["subject_initial", "verb_initial"],
    },
    {"name": "order_variant", "kind": "grammar", "values": ["variant_a", "variant_b"]},
    {
        "name": "alignment",
        "kind": "grammar",
        "values": ["nominative_accusative", "ergative_absolutive"],
    },
    {"name": "agreement", "kind": "grammar", "values": ["zero", "agent_number"]},
)
FEATURE_NAMES = tuple(item["name"] for item in FEATURE_DECLARATION)
LEXICAL_FEATURES = FEATURE_NAMES[:4]
GRAMMAR_FEATURES = FEATURE_NAMES[4:]

# Form slots swapped by each lexical feature, per lexical category.
_SWAPS: dict[str, tuple[tuple[int, int, str], ...]] = {
    "noun": ((0, 1, "noun_swap_01"), (2, 3, "noun_swap_23")),
    "verb": ((0, 1, "verb_swap_01"),),
    "modifier": ((0, 1, "modifier_swap_01"),),
}
_WORD_ORDERS = (("SVO", "SOV"), ("VSO", "VOS"))


@dataclass(frozen=True, order=True)
class Hypothesis:
    """One language of the base class, given by its eight feature bits."""

    bits: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject anything other than exactly eight binary bits."""
        if len(self.bits) != 8 or any(bit not in (0, 1) for bit in self.bits):
            raise ValueError("a hypothesis is exactly eight binary involutions")

    @property
    def integer(self) -> int:
        """Return the hypothesis index with bit ``i`` for ``FEATURE_NAMES[i]``."""
        return sum(bit << index for index, bit in enumerate(self.bits))

    @property
    def id(self) -> str:
        """Return the stable hypothesis identifier, e.g. ``h037``."""
        return f"h{self.integer:03d}"

    def bit(self, feature: str) -> int:
        """Return the bit of one named feature."""
        try:
            return self.bits[FEATURE_NAMES.index(feature)]
        except ValueError as exc:
            raise KeyError(feature) from exc

    @property
    def word_order(self) -> str:
        """Return the word order implied by the two order coordinates."""
        return _WORD_ORDERS[self.bit("order_family")][self.bit("order_variant")]

    def feature_value(self, feature: str) -> str:
        """Return the declared value label of one feature."""
        if feature == "word_order":
            return self.word_order
        try:
            declaration = FEATURE_DECLARATION[FEATURE_NAMES.index(feature)]
        except ValueError as exc:
            raise KeyError(feature) from exc
        return declaration["values"][self.bit(feature)]

    def feature_values(self) -> dict[str, str]:
        """Return every feature value label, including the derived word order."""
        values = {name: self.feature_value(name) for name in FEATURE_NAMES}
        return {**values, "word_order": self.word_order}

    def lexical_form_slot(self, category: str, semantic: str) -> int:
        """Return the form slot that expresses a meaning in this language."""
        if category not in LEXICON or semantic not in LEXICON[category]:
            raise KeyError((category, semantic))
        slot = LEXICON[category].index(semantic)
        for left, right, feature in _SWAPS[category]:
            if self.bit(feature) and slot in (left, right):
                return right if slot == left else left
        return slot

    def semantic_for_slot(self, category: str, slot: int) -> str:
        """Return the meaning expressed by one form slot in this language."""
        if category not in LEXICON or not 0 <= slot < len(LEXICON[category]):
            raise KeyError((category, slot))
        semantic_slot = slot
        for left, right, feature in _SWAPS[category]:
            if self.bit(feature) and slot in (left, right):
                semantic_slot = right if slot == left else left
        return LEXICON[category][semantic_slot]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hypothesis with its bits and value labels."""
        return {"id": self.id, "bits": list(self.bits), "features": self.feature_values()}


HYPOTHESIS_CLASS: tuple[Hypothesis, ...] = tuple(
    sorted(
        (Hypothesis(tuple(bits)) for bits in product((0, 1), repeat=8)),
        key=lambda item: item.id,
    )
)
HYPOTHESIS_CLASS_HASH = object_hash(
    {
        "features": FEATURE_DECLARATION,
        "hypotheses": [item.to_dict() for item in HYPOTHESIS_CLASS],
    }
)
