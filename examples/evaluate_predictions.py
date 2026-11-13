"""Evaluate a predictions file and print the headline metrics."""

import sys
from pathlib import Path

from silica.evaluate import evaluate_predictions

report = evaluate_predictions(Path(sys.argv[1]), Path(sys.argv[2]))
rows, metrics = report["rows"], report["template_cluster_bootstrap"]["metrics"]
low, high = metrics["balanced_strict_joint_accuracy"]["ci95"]
print(f"{report['cohort']['name']} cohort: {rows['n']} rows, valid {rows['format_valid_rate']:.3f}")
print(f"AR {rows['answerable_decision_recall']:.3f}  FCR {rows['false_certainty_rate']:.3f}")
print(f"BJA {rows['balanced_strict_joint_accuracy']:.3f} [{low:.3f}, {high:.3f}]")
