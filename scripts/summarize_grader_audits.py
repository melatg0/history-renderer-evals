#!/usr/bin/env python
"""Produce anonymous condition-level grader-audit tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

HUMAN_COLLISION_EXCLUSIONS = {"ADJ-3691ff4e"}
HUMAN_POSITIVE_DISAGREEMENTS = {
    "ADJ-09e7212c",
    "ADJ-0d58b379",
    "ADJ-1ba7520a",
    "ADJ-2bd82c2b",
    "ADJ-26f93cbf",
    "ADJ-2872f720",
}


def confusion_rows(
    independent_rows: list[dict[str, str]],
    adjudication_key: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    for condition in ("A_raw", "A3"):
        selected = [
            row for row in independent_rows if row["cond"] == condition
        ]
        counts = Counter(
            (int(row["primary_final"]), int(row["alt_final"]))
            for row in selected
        )
        for reference, audit in ((0, 0), (0, 1), (1, 0), (1, 1)):
            output.append(
                {
                    "audit": "independent_gpt4o",
                    "condition": condition,
                    "reference_label": reference,
                    "audit_label": audit,
                    "count": counts[(reference, audit)],
                }
            )

    human_labels: list[dict[str, Any]] = []
    for blinded_id, entry in sorted(adjudication_key.items()):
        if blinded_id in HUMAN_COLLISION_EXCLUSIONS:
            continue
        automated = int(entry["auto_final"])
        human = (
            1
            if blinded_id in HUMAN_POSITIVE_DISAGREEMENTS
            else automated
        )
        human_labels.append(
            {
                "blinded_id": blinded_id,
                "condition": entry["condition"],
                "stratum": entry["stratum"],
                "automated_label": automated,
                "human_label": human,
            }
        )

    for condition in ("A_raw", "A3"):
        selected = [
            row for row in human_labels if row["condition"] == condition
        ]
        counts = Counter(
            (int(row["automated_label"]), int(row["human_label"]))
            for row in selected
        )
        for reference, audit in ((0, 0), (0, 1), (1, 0), (1, 1)):
            output.append(
                {
                    "audit": "single_author_human",
                    "condition": condition,
                    "reference_label": reference,
                    "audit_label": audit,
                    "count": counts[(reference, audit)],
                }
            )
    return output, human_labels


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--independent",
        type=Path,
        default=Path("results/grader_invariance.csv"),
    )
    parser.add_argument(
        "--human-key",
        type=Path,
        default=Path("results/adjudication_key.json"),
    )
    parser.add_argument(
        "--confusion-out",
        type=Path,
        default=Path("results/grader_audit_confusion.csv"),
    )
    parser.add_argument(
        "--human-labels-out",
        type=Path,
        default=Path("results/human_audit_labels.csv"),
    )
    args = parser.parse_args()

    with args.independent.open(encoding="utf-8", newline="") as handle:
        independent_rows = list(csv.DictReader(handle))
    key = json.loads(args.human_key.read_text(encoding="utf-8"))
    confusion, human_labels = confusion_rows(independent_rows, key)
    _write(args.confusion_out, confusion)
    _write(args.human_labels_out, human_labels)
    print(
        f"Wrote {args.confusion_out} and {args.human_labels_out} "
        f"({len(human_labels)} non-collision human-audit items)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
