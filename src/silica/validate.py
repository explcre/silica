"""Fail-closed validation of a released dataset directory.

Validation checks every checksum and manifest field, recomputes every row's
certificate and gold decision from public information alone, recomputes the
acceptance audit and the semantic-content commitments, and compares each
result to what the files declare.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._frozen import (
    DATASET_FORMAT,
    DATASET_NAME,
    DATASET_VERSION,
    LIFECYCLE,
    PUBLIC_FORMAT,
    SCHEMA_FAMILY,
)
from .acceptance import acceptance_audit
from .certificates import make_certificate
from .declaration import inventory_from_declaration
from .evidence import evidence_from_dict
from .generator import (
    RELEASE_ROWS,
    SPLIT,
    dataset_sha256,
    forbidden_paths,
    lifecycle_metadata,
    public_instance_id,
    require_frozen_semantics,
    semantic_content_commitments,
    sha256sums_text,
)
from .routes import instantiation_protocol, route_protocol, route_registry
from .templates import template_registry
from .util import file_sha256, object_hash, read_json, read_jsonl

_HIDDEN_ROLE_FIELDS = (
    "query_feature",
    "nuisance_feature",
    "role_zero_feature",
    "role_one_feature",
    "ordered_evidence_features",
)


@dataclass
class ValidationReport:
    """The errors found in a dataset directory and its basic statistics."""

    errors: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return whether no error was found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {"ok": self.ok, "errors": list(self.errors), "statistics": dict(self.statistics)}

    def require_ok(self) -> None:
        """Raise if any error was found."""
        if self.errors:
            raise ValueError("invalid SILICA dataset:\n- " + "\n- ".join(self.errors))


# Release identity fields every manifest must carry.
_IDENTITY = {
    "schema_family": SCHEMA_FAMILY,
    "artifact_profile": LIFECYCLE["artifact_profile"],
    "dataset_name": DATASET_NAME,
    "dataset_version": DATASET_VERSION,
    "development_only": LIFECYCLE["development_only"],
    "confirmatory_status": LIFECYCLE["confirmatory_status"],
}


def _check_manifest(manifest: Mapping[str, Any], report: ValidationReport) -> None:
    """Check the manifest identity, protocols and configuration."""
    for name, value in {"format": DATASET_FORMAT, **_IDENTITY}.items():
        if manifest.get(name) != value:
            report.errors.append(f"manifest field mismatch: {name}")
    if manifest.get("route_protocol") != route_protocol():
        report.errors.append("route protocol mismatch")
    if manifest.get("instantiation_protocol") != instantiation_protocol():
        report.errors.append("instantiation protocol mismatch")
    if manifest.get("private_instantiation_registry") != [
        item.private_mapping() for item in route_registry().routes
    ]:
        report.errors.append("private route registry mismatch")
    if manifest.get("private_template_registry") != [
        item.private_metadata() for item in template_registry().all
    ]:
        report.errors.append("private template registry mismatch")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or object_hash(config) != manifest.get("config_hash"):
        report.errors.append("configuration hash mismatch")
    elif any(
        config.get(name) != value
        for name, value in {
            "artifact_profile": LIFECYCLE["artifact_profile"],
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "split": SPLIT,
            "development_only": LIFECYCLE["development_only"],
        }.items()
    ):
        report.errors.append("configuration lifecycle identity mismatch")


def _check_public_manifest(
    root: Path, manifest: Mapping[str, Any], report: ValidationReport
) -> dict[str, Any] | None:
    """Check the public manifest's checksum, leakage and lifecycle identity."""
    path = root / "public" / "manifest.json"
    try:
        public_manifest = read_json(path)
    except (OSError, ValueError) as exc:
        report.errors.append(f"cannot read public manifest: {exc}")
        return None
    if file_sha256(path) != manifest.get("public_manifest", {}).get("sha256"):
        report.errors.append("public manifest checksum mismatch")
    if public_manifest.get("format") != PUBLIC_FORMAT:
        report.errors.append("unexpected public manifest format")
    if forbidden_paths(public_manifest):
        report.errors.append("public manifest contains evaluator-only fields")
    for name, value in _IDENTITY.items():
        if public_manifest.get(name) != value:
            report.errors.append(f"public manifest lifecycle mismatch: {name}")
    if public_manifest.get("splits") != [SPLIT]:
        report.errors.append("public manifest split mismatch")
    if public_manifest.get("files") != manifest.get("public_files"):
        report.errors.append("public/root file inventory mismatch")
    return public_manifest


def _check_public_rows(public_rows: list[dict[str, Any]], report: ValidationReport) -> None:
    """Check leakage, release identity and content-derived IDs of every public row."""
    for row in public_rows:
        problem = None
        mechanism = row.get("mechanism")
        if forbidden_paths(row):
            problem = "public row contains forbidden fields"
        elif row.get("dataset_version") != DATASET_VERSION:
            problem = "row dataset version mismatch"
        elif row.get("split") != SPLIT:
            problem = "row split mismatch"
        elif not isinstance(mechanism, Mapping) or any(
            key in mechanism for key in _HIDDEN_ROLE_FIELDS
        ):
            problem = "public mechanism exposes hidden roles"
        elif public_instance_id(
            {key: value for key, value in row.items() if key != "id"}
        ) != row.get("id"):
            problem = "public instance ID mismatch"
        if problem:
            report.errors.append(f"{row.get('id')}: {problem}")
            return


def _check_certificates(
    public_by_id: Mapping[str, Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
    report: ValidationReport,
) -> None:
    """Recompute every certificate and gold decision from the public row alone."""
    for identifier, public in public_by_id.items():
        private = gold_by_id[identifier]
        try:
            declaration = public["hypothesis_declaration"]
            certificate, gold = make_certificate(
                str(public["task"]),
                public["query"],
                tuple(evidence_from_dict(item) for item in public["evidence"]),
                inventory_from_declaration(declaration),
                declaration["boolean_program"],
            )
            if certificate != private.get("certificate"):
                raise ValueError("certificate differs from public recomputation")
            serialized = private.get("gold")
            if (
                not isinstance(serialized, Mapping)
                or {
                    "decision": serialized.get("decision"),
                    "answer": serialized.get("answer"),
                }
                != gold
            ):
                raise ValueError("gold differs from public recomputation")
            oracle = private.get("oracle")
            if not isinstance(oracle, Mapping) or oracle.get("public_record_hash") != object_hash(
                public
            ):
                raise ValueError("oracle public-record binding mismatch")
        except (KeyError, TypeError, ValueError, AssertionError) as exc:
            report.errors.append(f"{identifier}: exact public recomputation failed: {exc}")
            return


def validate_dataset(dataset: str | Path) -> ValidationReport:
    """Validate a dataset directory completely and return every error found."""
    root = Path(dataset)
    report = ValidationReport()
    try:
        manifest = read_json(root / "manifest.json")
    except (OSError, ValueError) as exc:
        report.errors.append(f"cannot read root manifest: {exc}")
        return report
    _check_manifest(manifest, report)
    entries = [*manifest.get("public_files", []), *manifest.get("gold_files", [])]
    loaded: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        path = root / str(entry.get("path"))
        if not path.is_file():
            report.errors.append(f"missing data file {entry.get('path')}")
            continue
        if file_sha256(path) != entry.get("sha256"):
            report.errors.append(f"checksum mismatch for {entry.get('path')}")
        try:
            rows = read_jsonl(path, canonical=True)
        except ValueError as exc:
            report.errors.append(str(exc))
            continue
        loaded[str(entry["path"])] = rows
        if len(rows) != entry.get("records"):
            report.errors.append(f"record count mismatch for {entry.get('path')}")
    if manifest.get("dataset_sha256") != dataset_sha256(entries):
        report.errors.append("aggregate dataset checksum mismatch")
    public_manifest = _check_public_manifest(root, manifest, report)

    public_rows = [
        row
        for entry in manifest.get("public_files", [])
        for row in loaded.get(str(entry["path"]), [])
    ]
    gold_rows = [
        row
        for entry in manifest.get("gold_files", [])
        for row in loaded.get(str(entry["path"]), [])
    ]
    if len(public_rows) != RELEASE_ROWS:
        report.errors.append(f"found {len(public_rows)} public rows, expected {RELEASE_ROWS}")
    _check_public_rows(public_rows, report)
    public_by_id = {str(row.get("id")): row for row in public_rows}
    gold_by_id = {str(row.get("id")): row for row in gold_rows}
    if (
        len(public_by_id) != len(public_rows)
        or len(gold_by_id) != len(gold_rows)
        or set(public_by_id) != set(gold_by_id)
    ):
        report.errors.append("serialized public/gold ID join is invalid")
    elif not report.errors:
        _check_certificates(public_by_id, gold_by_id, report)
    try:
        if acceptance_audit(public_rows, gold_rows) != manifest.get("acceptance_audit"):
            report.errors.append("serialized acceptance audit mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        report.errors.append(f"acceptance audit failed: {exc}")
    if len(public_rows) == len(gold_rows) == RELEASE_ROWS:
        try:
            seed = str(manifest.get("config", {}).get("seed"))
            commitments = semantic_content_commitments(public_rows, gold_rows)
            require_frozen_semantics(seed, commitments)
            for name, expected in lifecycle_metadata(seed, commitments).items():
                if manifest.get(name) != expected:
                    report.errors.append(f"lifecycle mismatch: {name}")
                if public_manifest is not None and public_manifest.get(name) != expected:
                    report.errors.append(f"public lifecycle mismatch: {name}")
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"semantic commitment failed: {exc}")
    sums = root / "SHA256SUMS"
    if not sums.is_file():
        report.errors.append("missing SHA256SUMS")
    else:
        try:
            if sums.read_text(encoding="utf-8") != sha256sums_text(root, manifest):
                report.errors.append("SHA256SUMS content mismatch")
        except OSError as exc:
            report.errors.append(f"cannot check SHA256SUMS: {exc}")
    report.statistics = {
        "rows": len(public_rows),
        "paired_query_groups": len({str(row.get("paired_query_unit_id")) for row in public_rows}),
        "four_cell_evaluator_units": len({str(row.get("complete_unit_id")) for row in public_rows}),
        "primary_semantic_template_clusters": len(
            {str(row.get("mechanism", {}).get("template_id")) for row in public_rows}
        ),
    }
    return report
