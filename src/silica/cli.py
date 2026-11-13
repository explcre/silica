"""The ``silica`` command line.

``silica generate``  regenerate the released dataset byte for byte
``silica verify``    check checksums, recompute every certificate, run the prompt-only solver
``silica prompts``   write the rendered chat messages of the full set or the common cohort
``silica evaluate``  score a predictions file
``silica audit``     run the shortcut controls, learned probes, or guard-structure census
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _emit(report: Any, output: Path | None) -> None:
    """Print a JSON report and optionally write it to a file."""
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _generate(args: argparse.Namespace) -> int:
    """Regenerate the release into an output directory."""
    from .generator import build_dataset

    manifest = build_dataset(args.output, overwrite=args.force)
    _emit(
        {
            "dataset": str(args.output),
            "dataset_sha256": manifest["dataset_sha256"],
            "rows": manifest["statistics"]["rows"],
            "decision_counts": manifest["statistics"]["decision_counts"],
        },
        None,
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Validate a dataset and solve every row from its prompt."""
    from .audits.solver import prompt_sufficiency
    from .dataset import load_gold_by_id, load_public_rows
    from .validate import validate_dataset

    validation = validate_dataset(args.dataset)
    solver = prompt_sufficiency(load_public_rows(args.dataset), load_gold_by_id(args.dataset))
    passed = validation.ok and solver["strict_joint_correct"] == solver["scored_rows"]
    _emit(
        {"passed": passed, "validation": validation.to_dict(), "prompt_only_solver": solver},
        args.output,
    )
    return 0 if passed else 1


def _prompts(args: argparse.Namespace) -> int:
    """Write rendered prompts for the chosen cohort as JSON lines."""
    from .cohort import COMMON_COHORT_ROWS, select_cohort
    from .dataset import load_public_rows
    from .prompts import prompt_ledger_identity, render_prompt, request_prompt_manifest
    from .util import write_jsonl

    rows = load_public_rows(args.dataset)
    selection = select_cohort(
        rows, rows=COMMON_COHORT_ROWS if args.cohort == "common" else len(rows)
    )
    items = selection.items
    records = []
    for item in items:
        prompt = render_prompt(item)
        records.append(
            {"id": item["id"], "prompt_hash": prompt.prompt_hash, "messages": prompt.as_messages()}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    _emit(
        {
            "cohort": args.cohort,
            "rows": len(items),
            "cohort_hash": selection.cohort_hash,
            "request_prompt_manifest_sha256": request_prompt_manifest(items),
            "prompt_ledger_sha256": prompt_ledger_identity(items),
            "output": str(args.output),
        },
        None,
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Score a predictions file."""
    from .evaluate import evaluate_predictions

    report = evaluate_predictions(
        args.dataset,
        args.predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    _emit(report, args.output)
    return 0


def _audit(args: argparse.Namespace) -> int:
    """Run one dataset audit."""
    from .dataset import load_gold_by_id, load_public_rows
    from .util import read_json

    rows = load_public_rows(args.dataset)
    gold_by_id = load_gold_by_id(args.dataset)
    if args.audit == "shortcuts":
        from .audits.shortcuts import shortcut_controls

        report = shortcut_controls(rows, gold_by_id, read_json(args.dataset / "manifest.json"))
        passed = report["all_exactly_one_half"] and report["matches_manifest"]
    elif args.audit == "probes":
        from .audits.probes import learned_probes

        report = learned_probes(rows, gold_by_id)
        passed = all(item["exactly_one_half"] for item in report["probes"].values())
    else:
        from .audits.structure import census, guard_probes

        records = [(row, gold_by_id[str(row["id"])]) for row in rows]
        report = {"census": census(records)}
        if not args.skip_probes:
            report["probes"] = guard_probes(records)
        passed = True
    _emit(report, args.output)
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of every subcommand."""
    from .evaluate import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED

    parser = argparse.ArgumentParser(
        prog="silica", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(
        name: str, handler: Callable[[argparse.Namespace], int], text: str
    ) -> argparse.ArgumentParser:
        """Register one subcommand."""
        sub = commands.add_parser(name, help=text, description=text)
        sub.set_defaults(handler=handler)
        return sub

    generate = command("generate", _generate, "Regenerate the released dataset.")
    generate.add_argument(
        "--output", type=Path, required=True, help="empty or new output directory"
    )
    generate.add_argument(
        "--force", action="store_true", help="replace a nonempty output directory"
    )

    verify = command(
        "verify", _verify, "Verify checksums, certificates and prompt-only decidability."
    )
    verify.add_argument("--dataset", type=Path, required=True)
    verify.add_argument("--output", type=Path, help="also write the JSON report here")

    prompts = command("prompts", _prompts, "Write rendered chat messages as JSON lines.")
    prompts.add_argument("--dataset", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True, help="JSONL file to write")
    prompts.add_argument("--cohort", choices=("common", "full"), default="common")

    evaluate = command(
        "evaluate", _evaluate, "Score a predictions file (release prediction format)."
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, help="also write the JSON report here")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    evaluate.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)

    audit = command("audit", _audit, "Run a dataset audit.")
    audit.add_argument("audit", choices=("shortcuts", "probes", "structure"))
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path, help="also write the JSON report here")
    audit.add_argument(
        "--skip-probes", action="store_true", help="structure: census only, no learned probes"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line."""
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
