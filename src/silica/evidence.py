"""Dictionary observations and exact version-space filtering.

Every released row reveals three dictionary observations: for a lexical
category and meaning, the root form the hidden language uses.  The query asks
about a fourth or fifth coordinate the observations never state directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .hypotheses import Hypothesis
from .surface import SurfaceInventory, root_for
from .util import object_hash


@dataclass(frozen=True)
class DictionaryEvidence:
    """One observed form/meaning pair of a lexical category."""

    category: str
    semantic: str
    form: str

    @property
    def channel(self) -> str:
        """Return the evidence channel name."""
        return "dictionary"

    @property
    def id(self) -> str:
        """Return the content-derived evidence identifier."""
        return "e-d-" + object_hash(self.payload())[:16]

    def payload(self) -> dict[str, Any]:
        """Return the identifying content of the observation."""
        return {
            "channel": self.channel,
            "category": self.category,
            "semantic": self.semantic,
            "form": self.form,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the observation with its identifier."""
        return {"id": self.id, **self.payload()}


def experiment_key(evidence: DictionaryEvidence) -> str:
    """Return the target-blind identity of the question an observation answers."""
    return object_hash(
        {"channel": "dictionary", "category": evidence.category, "semantic": evidence.semantic}
    )


def evidence_from_dict(data: Mapping[str, Any]) -> DictionaryEvidence:
    """Parse one serialized observation, rejecting noncanonical payloads."""
    if data.get("channel") != "dictionary":
        raise ValueError(f"unknown evidence channel: {data.get('channel')!r}")
    evidence = DictionaryEvidence(
        category=str(data["category"]), semantic=str(data["semantic"]), form=str(data["form"])
    )
    if dict(data) != evidence.to_dict():
        raise ValueError("evidence payload is non-canonical or has an invalid id")
    return evidence


def is_consistent(
    hypothesis: Hypothesis, evidence: DictionaryEvidence, inventory: SurfaceInventory
) -> bool:
    """Return whether a language would produce an observation."""
    return root_for(evidence.category, evidence.semantic, hypothesis, inventory) == evidence.form


def filter_hypotheses(
    hypotheses: Iterable[Hypothesis],
    evidence: Iterable[DictionaryEvidence],
    inventory: SurfaceInventory,
) -> tuple[Hypothesis, ...]:
    """Return the hypotheses consistent with every observation."""
    observations = tuple(evidence)
    return tuple(
        hypothesis
        for hypothesis in hypotheses
        if all(is_consistent(hypothesis, item, inventory) for item in observations)
    )


def evidence_hash(evidence: Sequence[DictionaryEvidence]) -> str:
    """Hash an observation set independently of its order."""
    return object_hash([item.to_dict() for item in sorted(evidence, key=lambda item: item.id)])
