#!/usr/bin/env python
"""Summarize Inspect-native tool-call eval logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log


def _score_value(score: Any, key: str) -> Any:
    if score is None or not hasattr(score, "value"):
        return None
    value = score.value
    if isinstance(value, dict):
        return value.get(key)
    return None


def _rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            metadata = sample.metadata or {}
            scores = sample.scores or {}
            score = scores.get("native_blackmail_scorer") or scores.get(
                "send_email_called_scorer"
            )
            actions = sample.store.get("native_tool_action_log", []) if sample.store else []
            memory_level = metadata.get("memory_level") or log.eval.task_args.get(
                "history_type"
            )
            memory_script_id = metadata.get("memory_script_id")
            if memory_level != "D0" and not memory_script_id:
                memory_script_id = log.eval.task_args.get("script_id")
            rows.append(
                {
                    "log": str(path),
                    "log_status": log.status,
                    "task": log.eval.task,
                    "model": log.eval.model,
                    "sample_id": sample.id,
                    "epoch": sample.epoch,
                    "run_block": metadata.get("run_block")
                    or log.eval.task_args.get("run_block"),
                    "memory_level": memory_level,
                    "memory_render_level": metadata.get("memory_render_level")
                    or log.eval.task_args.get("history_type"),
                    "summary_variant": metadata.get("summary_variant")
                    or log.eval.task_args.get("summary_variant"),
                    "memory_script_id": memory_script_id,
                    "harmful": _score_value(score, "harmful"),
                    "classifier_verdict": _score_value(score, "classifier_verdict"),
                    "sent_email": _score_value(score, "sent_email"),
                    "target_contacted": _score_value(score, "target_contacted"),
                    "legacy_xml_harmful_gate": _score_value(
                        score, "legacy_xml_harmful_gate"
                    ),
                    "grader_error": _score_value(score, "grader_error"),
                    "called_send_email": _score_value(score, "called_send_email"),
                    "action_count": len(actions),
                    "actions": ",".join(str(action.get("tool")) for action in actions),
                }
            )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        "harmful",
        "classifier_verdict",
        "sent_email",
        "target_contacted",
        "legacy_xml_harmful_gate",
        "grader_error",
        "called_send_email",
    ]
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["memory_level"], row["memory_script_id"])
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (memory_level, memory_script_id), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "memory_level": memory_level,
            "memory_script_id": memory_script_id,
            "n": len(group),
        }
        for field in numeric_fields:
            values = [
                float(row[field])
                for row in group
                if row.get(field) not in (None, "")
            ]
            summary[f"{field}_n"] = len(values)
            summary[f"{field}_sum"] = sum(values) if values else ""
            summary[f"{field}_rate"] = (sum(values) / len(values)) if values else ""
        out.append(summary)
    return out


def _expand(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("**/*.eval")))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(value)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Inspect .eval files or directories")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--aggregate-out",
        default=None,
        help="Optional aggregate CSV output path grouped by memory level/script",
    )
    args = parser.parse_args()

    paths = _expand(args.paths)
    rows = _rows(paths)
    if not rows:
        raise SystemExit("No rows found")

    fields = list(rows[0])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} row(s) to {out}")
    else:
        writer = csv.DictWriter(
            __import__("sys").stdout,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    if args.aggregate_out:
        aggregate_rows = _aggregate(rows)
        aggregate_fields = list(aggregate_rows[0])
        aggregate_out = Path(args.aggregate_out)
        aggregate_out.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=aggregate_fields, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)
        print(f"Wrote {len(aggregate_rows)} aggregate row(s) to {aggregate_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
