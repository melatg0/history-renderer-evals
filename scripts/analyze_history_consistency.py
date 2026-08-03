#!/usr/bin/env python
"""Compute history-level directional consistency for controlled studies."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    non_ties = positive + negative
    if non_ties == 0:
        return 1.0
    tail = min(positive, negative)
    one_sided = sum(
        math.comb(non_ties, count) for count in range(tail + 1)
    ) / (2**non_ties)
    return min(1.0, 2 * one_sided)


def analyze_qwen(rows: list[dict[str, str]]) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["dataset"] == "history_robustness"
        and row["model"] == "Qwen3-30B"
    ]
    baseline_rows = [row for row in selected if row["condition"] == "baseline"]
    history_rows = [row for row in selected if row["condition"] == "history"]
    if len(baseline_rows) != 1 or len(history_rows) != 8:
        raise ValueError("Expected one Qwen baseline and eight history cells")

    baseline_harmful = int(baseline_rows[0]["harmful"])
    baseline_n = int(baseline_rows[0]["n"])
    baseline_rate = baseline_harmful / baseline_n
    differences = [
        int(row["harmful"]) / int(row["n"]) - baseline_rate
        for row in history_rows
    ]
    below = sum(difference < 0 for difference in differences)
    above = sum(difference > 0 for difference in differences)
    ties = sum(difference == 0 for difference in differences)
    return {
        "model": "Qwen3-30B",
        "comparison": "full benign history vs pooled no-history baseline",
        "baseline_harmful": baseline_harmful,
        "baseline_n": baseline_n,
        "baseline_rate": baseline_rate,
        "n_histories": len(history_rows),
        "histories_below_baseline": below,
        "histories_above_baseline": above,
        "ties": ties,
        "test": "two-sided exact sign test",
        "p_value": exact_two_sided_sign_p(below, above),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/paper/figure_counts.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper/history_consistency_tests.csv"),
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as handle:
        result = analyze_qwen(list(csv.DictReader(handle)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
