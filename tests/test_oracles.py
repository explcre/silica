"""Agreement of the generator oracle, the clean-room oracle and the prompt-only solver."""

import importlib.util
from pathlib import Path

from silica.audits.solver import prompt_sufficiency
from silica.certificates import make_certificate
from silica.declaration import inventory_from_declaration
from silica.evidence import evidence_from_dict

CLEANROOM = Path(__file__).resolve().parents[1] / "audits" / "cleanroom_oracle.py"


def _cleanroom():
    """Load the clean-room oracle module from its file."""
    spec = importlib.util.spec_from_file_location("cleanroom_oracle", CLEANROOM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_recomputation_matches_gold(small_release):
    """Certificates recomputed from public data equal the stored gold certificates."""
    public, gold_by_id = small_release
    for row in public:
        declaration = row["hypothesis_declaration"]
        certificate, gold = make_certificate(
            row["task"],
            row["query"],
            tuple(evidence_from_dict(item) for item in row["evidence"]),
            inventory_from_declaration(declaration),
            declaration["boolean_program"],
        )
        assert certificate == gold_by_id[row["id"]]["certificate"]
        assert gold == {key: gold_by_id[row["id"]]["gold"][key] for key in ("decision", "answer")}


def test_cleanroom_oracle_agrees(small_release):
    """The standard-library oracle reproduces every certificate of the sample."""
    cleanroom = _cleanroom()
    public, gold_by_id = small_release
    for row in public:
        accepted, _, class_hash = cleanroom.exact_program_metadata(row["hypothesis_declaration"])
        assert class_hash == row["hypothesis_declaration"]["class_hash"]
        certificate, _ = cleanroom.exact_certificate(row, accepted, row["hypothesis_declaration"])
        assert certificate == gold_by_id[row["id"]]["certificate"]


def test_cleanroom_oracle_imports_no_silica():
    """The clean-room oracle stays independent of the package it audits."""
    imports = [
        line.strip()
        for line in CLEANROOM.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]
    assert imports and not any("silica" in line for line in imports)


def test_prompt_only_solver_agrees(small_release):
    """The solver that reads only the rendered prompt reproduces every gold label."""
    public, gold_by_id = small_release
    report = prompt_sufficiency(public, gold_by_id)
    assert report["strict_joint_correct"] == report["scored_rows"] == len(public)
    assert report["posterior_size_histogram"] == {16: len(public)}
