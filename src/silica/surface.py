"""Deterministic opaque surface inventories and lexical form lookup.

Each synthetic language spells its roots and markers with pseudo-random
four-letter forms derived from a seed, so no form carries meaning by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._frozen import SURFACE_FORM_DOMAIN
from .hypotheses import Hypothesis
from .util import sha256_text

_CONSONANTS = "ptkmnszlrvgdbh"
_VOWELS = "aeiou"
_SLOTS = (
    *(f"{category}_{index}" for category in ("noun", "verb", "modifier") for index in range(4)),
    "plural",
    "case_a",
    "case_b",
    "past",
    "negation",
    "agree_singular",
    "agree_plural",
)


def _opaque_form(seed: str, slot: str, attempt: int) -> str:
    """Return one consonant-vowel-consonant-vowel form for a seeded slot."""
    digest = bytes.fromhex(sha256_text(f"{SURFACE_FORM_DOMAIN}|{seed}|{slot}|{attempt}"))
    return "".join(
        (
            _CONSONANTS[digest[0] % len(_CONSONANTS)],
            _VOWELS[digest[1] % len(_VOWELS)],
            _CONSONANTS[digest[2] % len(_CONSONANTS)],
            _VOWELS[digest[3] % len(_VOWELS)],
        )
    )


@dataclass(frozen=True)
class SurfaceInventory:
    """The opaque forms of one language, keyed by slot name."""

    seed: str
    forms: Mapping[str, str]

    @classmethod
    def from_seed(cls, seed: str) -> SurfaceInventory:
        """Derive a collision-free inventory deterministically from a seed."""
        used: set[str] = set()
        forms: dict[str, str] = {}
        for slot in _SLOTS:
            attempt = 0
            form = _opaque_form(seed, slot, attempt)
            while form in used:
                attempt += 1
                form = _opaque_form(seed, slot, attempt)
            used.add(form)
            forms[slot] = form
        return cls(seed, forms)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the seed and the slot-sorted forms."""
        return {"seed": self.seed, "forms": dict(sorted(self.forms.items()))}

    def roots(self, category: str) -> tuple[str, ...]:
        """Return the four root forms of one lexical category in slot order."""
        if category not in ("noun", "verb", "modifier"):
            raise KeyError(category)
        return tuple(self.forms[f"{category}_{index}"] for index in range(4))


def root_for(
    category: str, semantic: str, hypothesis: Hypothesis, inventory: SurfaceInventory
) -> str:
    """Return the root form that expresses a meaning in one language."""
    return inventory.roots(category)[hypothesis.lexical_form_slot(category, semantic)]


def semantic_for_root(
    category: str, form: str, hypothesis: Hypothesis, inventory: SurfaceInventory
) -> str | None:
    """Return the meaning of a root form in one language, or None if undeclared."""
    try:
        slot = inventory.roots(category).index(form)
    except ValueError:
        return None
    return hypothesis.semantic_for_slot(category, slot)
