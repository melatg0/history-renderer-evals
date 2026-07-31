#!/usr/bin/env python
"""Fold Inspect .eval logs into the plan's results layout.

Reads the per-cell .eval logs produced by run_experiment.py and writes::

    results/{experiment_id}/{model}/{history_level}/{script_id}/
        rollouts.jsonl   # one line per rollout: transcript + parsed label
        meta.json        # n, n_blackmail, rate + full provenance
    results/{experiment_id}/aggregated.csv   # one row per (model, level, script, cf)

The Inspect-log reading (`read_records_from_logs`) is isolated from the pure
aggregation logic (`build_aggregated`, `write_outputs`) so the latter can be
unit-tested without the eval harness.

Usage:
    python eval/agentic_misalignment/experiment/aggregate.py \
        --config configs/t1_1_history_robustness.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.config import ExperimentConfig, sanitize_model  # noqa: E402

GROUP_KEYS = [
    "model", "scenario", "goal_type", "goal_value", "urgency_type",
    "extra_system_instructions", "cf",
    "history_level", "script_id",
]


def read_records_from_logs(log_base: Path, model: str) -> list[dict[str, Any]]:
    from inspect_ai.log import read_eval_log  # local import: needs inspect_ai

    records: list[dict[str, Any]] = []
    log_files = sorted(log_base.glob("**/*.eval"))
    if not log_files:
        raise FileNotFoundError(f"No .eval files under {log_base}. Run run_experiment.py first.")

    for log_path in log_files:
        log = read_eval_log(str(log_path))
        for sample in (log.samples or []):
            meta = sample.metadata or {}
            scores = sample.scores or {}

            harmful = _score_value(scores.get("harmfulness_scorer"), "harmful")
            grader_error = _score_value(scores.get("harmfulness_scorer"), "grader_error")
            # Keep failed grades for auditing, but exclude them from the rate.
            # A missing score without an error flag is unusable.
            if harmful is None and not grader_error:
                continue
            verdict = _score_value(scores.get("harmfulness_scorer"), "classifier_verdict")
            manip = _score_value(scores.get("manipulation_check_scorer"), "manipulation_check")

            messages = _serialize_messages(sample)
            completion = ""
            if getattr(sample, "output", None) is not None:
                completion = getattr(sample.output, "completion", "") or ""

            records.append({
                "model": model,
                "scenario": meta.get("scenario"),
                "goal_type": meta.get("goal_type"),
                "goal_value": meta.get("goal_value"),
                "urgency_type": meta.get("urgency_type"),
                "extra_system_instructions": meta.get("extra_system_instructions"),
                "cf": bool(meta.get("cf", False)),
                "history_level": meta.get("history_type", "A0"),
                "script_id": meta.get("script_id"),
                "sample_id": getattr(sample, "id", None),
                "epoch": getattr(sample, "epoch", None),
                "blackmail": int(round(float(harmful))) if harmful is not None else float("nan"),
                "grader_error": bool(grader_error),
                "classifier_verdict": int(round(float(verdict))) if verdict is not None else None,
                "manipulation_check": float(manip) if manip is not None else None,
                "completion": completion,
                "messages": messages,
            })
    return records


def _score_value(score: Any, key: str) -> float | None:
    if score is None or not hasattr(score, "value"):
        return None
    val = score.value
    if isinstance(val, dict):
        v = val.get(key)
        return float(v) if v is not None else None
    return None


def _serialize_messages(sample: Any) -> list[dict[str, str]]:
    msgs = getattr(sample, "messages", None) or getattr(sample, "input", None) or []
    out = []
    for m in msgs:
        role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "?")
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content", "")
        out.append({"role": str(role), "content": content if isinstance(content, str) else str(content)})
    return out


def build_aggregated(records: Iterable[dict[str, Any]], extra: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(list(records))
    if df.empty:
        return df
    if "grader_error" not in df:
        df["grader_error"] = False
    # pandas excludes failed grades (NaN) from n and the harmful-action rate.
    grouped = (
        df.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n=("blackmail", "count"),  # count ignores NaN
            n_blackmail=("blackmail", "sum"),  # sum ignores NaN
            n_grader_error=("grader_error", "sum"),
            n_attempted=("blackmail", "size"),  # includes failed grades
        )
        .reset_index()
    )
    grouped["n_blackmail"] = grouped["n_blackmail"].astype(int)
    grouped["n_grader_error"] = grouped["n_grader_error"].astype(int)
    grouped["rate"] = grouped["n_blackmail"] / grouped["n"].where(grouped["n"] > 0)
    for k, v in extra.items():
        grouped[k] = v
    sort_cols = [c for c in ["history_level", "script_id", "cf"] if c in grouped]
    return grouped.sort_values(sort_cols).reset_index(drop=True)


def write_outputs(
    cfg: ExperimentConfig,
    records: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> Path:
    exp_dir = cfg.results_root / cfg.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_dir_name = sanitize_model(cfg.model)

    extra = {
        "experiment_id": cfg.experiment_id,
        "classifier_version": run_meta.get("classifier_version"),
        "eval_version": run_meta.get("eval_version"),
        "seed": cfg.seed,
        "temperature": cfg.temperature,
        "timestamp_utc": run_meta.get("timestamp_utc"),
    }
    agg = build_aggregated(records, extra)

    df = pd.DataFrame(records)
    if not df.empty:
        for _, row in agg.iterrows():
            sid = row["script_id"] if pd.notna(row["script_id"]) else None
            cell_dir = exp_dir / model_dir_name / row["history_level"] / (sid or "_none")
            cell_dir.mkdir(parents=True, exist_ok=True)

            mask = (df["history_level"] == row["history_level"]) & (df["cf"] == row["cf"])
            mask &= (df["script_id"] == sid) if sid is not None else df["script_id"].isna()
            cell_records = df[mask].to_dict(orient="records")

            with (cell_dir / "rollouts.jsonl").open("w", encoding="utf-8") as f:
                for r in cell_records:
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

            meta = {
                "experiment_id": cfg.experiment_id,
                "model": cfg.model,
                "history_level": row["history_level"],
                "script_id": sid,
                "cf": bool(row["cf"]),
                "scenario": row["scenario"],
                "goal_type": row["goal_type"],
                "goal_value": row["goal_value"],
                "urgency_type": row["urgency_type"],
                "extra_system_instructions": row["extra_system_instructions"],
                "n": int(row["n"]),
                "n_blackmail": int(row["n_blackmail"]),
                "n_grader_error": int(row.get("n_grader_error", 0)),
                "rate": None if pd.isna(row["rate"]) else float(row["rate"]),
                "seed": cfg.seed,
                "temperature": cfg.temperature,
                "grader_model": cfg.grader_model,
                "config_hash": run_meta.get("config_hash"),
                "framework_commit": run_meta.get("framework_commit"),
                "classifier_version": run_meta.get("classifier_version"),
                "timestamp_utc": run_meta.get("timestamp_utc"),
            }
            (cell_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    out_csv = exp_dir / "aggregated.csv"
    agg.to_csv(out_csv, index=False)
    return out_csv


def load_run_meta(cfg: ExperimentConfig) -> dict[str, Any]:
    p = cfg.log_base / "run_meta.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    run_meta = load_run_meta(cfg)
    records = read_records_from_logs(cfg.log_base, cfg.model)
    out_csv = write_outputs(cfg, records, run_meta)

    agg = pd.read_csv(out_csv)
    print(f"Loaded {len(records)} rollouts -> {len(agg)} cells")
    print(agg.to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Per-cell rollouts.jsonl + meta.json under {cfg.results_root / cfg.experiment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
