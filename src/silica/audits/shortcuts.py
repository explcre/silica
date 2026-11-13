"""Query-masked shortcut controls recomputed from the released files.

Each of the eight controls maps a row to a query-blind signature (for
example, the row with its query and query-derived fields removed) and scores
the best predictor that assigns one decision per signature.  Because the two
queries of every state share their raw state, every control must score
exactly one half.  The recomputed controls are compared with the values the
release manifest records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..acceptance import shortcut_probes


def shortcut_controls(
    public_rows: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the controls and compare them with the manifest's record."""
    probes = shortcut_probes(public_rows, gold_by_id)
    return {
        "probes": probes,
        "all_exactly_one_half": all(
            item["exactly_one_half"] and item["all_signatures_exactly_balanced"]
            for item in probes.values()
        ),
        "matches_manifest": probes == manifest.get("acceptance_audit", {}).get("shortcut_probes"),
    }
