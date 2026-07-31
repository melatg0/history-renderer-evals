#!/usr/bin/env python
"""Config-driven runner: launch one `inspect eval` per cell.

Reads an experiment YAML, enumerates cells, runs a pre-flight cost estimate,
then launches `inspect eval` for each cell with a fixed seed and a per-cell
``--log-dir``. Resumable: cells whose log dir already contains a ``.eval`` file
are skipped unless ``--force``. ``--dry-run`` runs 1 rollout/cell.

Usage:
    python eval/agentic_misalignment/experiment/run_experiment.py \
        --config configs/t1_1_history_robustness.yaml [--dry-run] [--yes]

Provenance (seed, model, config hash, framework commit, classifier version,
UTC timestamp) is written to {log_base}/run_meta.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support direct script execution as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment.config import Cell, ExperimentConfig, repo_root  # noqa: E402


def _classifier_version() -> str:
    try:
        from classifiers import CLASSIFIER_VERSION  # type: ignore

        return CLASSIFIER_VERSION
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip()
    except Exception:
        return "unknown"


def build_inspect_command(
    cfg: ExperimentConfig,
    cell: Cell,
    dry_run: bool,
    task: str,
    use_uv: bool,
) -> list[str]:
    epochs = 1 if dry_run else cell.n
    cmd: list[str] = []
    if use_uv:
        cmd += ["uv", "run"]
    cmd += [
        "inspect", "eval", task,
        "--model", cfg.model,
        "--epochs", str(epochs),
        "--temperature", str(cfg.temperature),
        "--max-connections", str(cfg.max_connections),
        "--seed", str(cfg.seed),
        "--log-dir", str(cell.log_dir(cfg.log_base)),
    ]

    # Hold current-session reasoning constant across cells so the only reasoning
    # that varies is the injected prior-session trace.
    if cfg.reasoning_effort is not None:
        cmd += ["--reasoning-effort", cfg.reasoning_effort]

    common = cfg.common
    passthrough = [
        "scenario", "goal_type", "goal_value", "urgency_type",
        "extra_system_instructions", "cf", "manipulation_check", "prod",
        "feedback", "test_eval_awareness", "score_eval_awareness",
    ]
    for key in passthrough:
        if key in common and common[key] is not None:
            cmd += ["-T", f"{key}={_fmt(common[key])}"]

    cmd += ["-T", f"history_type={cell.level}"]
    if cell.script_id:
        cmd += ["-T", f"script_id={cell.script_id}"]

    if cfg.grader_model:
        cmd += ["-T", f"grader_model={cfg.grader_model}"]

    return cmd


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_run_meta(cfg: ExperimentConfig, dry_run: bool) -> Path:
    cfg.log_base.mkdir(parents=True, exist_ok=True)
    meta = {
        "experiment_id": cfg.experiment_id,
        "seed": cfg.seed,
        "model": cfg.model,
        "grader_model": cfg.grader_model,
        "temperature": cfg.temperature,
        "reasoning_effort": cfg.reasoning_effort,
        "common": cfg.common,
        "config_path": str(cfg.path),
        "config_hash": cfg.config_hash(),
        "framework_commit": _git_commit(),
        "classifier_version": _classifier_version(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "n_cells": len(cfg.cells()),
        "total_rollouts": cfg.total_rollouts(dry_run),
        "estimated_usd": round(cfg.estimate_usd(dry_run), 2),
    }
    out = cfg.log_base / "run_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def preflight(cfg: ExperimentConfig, dry_run: bool, assume_yes: bool) -> bool:
    cells = cfg.cells()
    n = cfg.total_rollouts(dry_run)
    est = cfg.estimate_usd(dry_run)
    print(f"\nExperiment: {cfg.experiment_id}")
    print(f"  model         : {cfg.model}")
    print(f"  seed          : {cfg.seed}   temperature: {cfg.temperature}")
    print(f"  reasoning     : {cfg.reasoning_effort or '(provider default)'}")
    print(f"  cells         : {len(cells)}")
    print(f"  rollouts      : {n}" + ("  (dry-run: 1/cell)" if dry_run else ""))
    print(f"  est. cost     : ${est:,.2f}  "
          f"(avg {cfg.cost.get('avg_tokens_per_rollout','?')} tok @ "
          f"${cfg.cost.get('usd_per_1k_tokens','?')}/1k)")
    if dry_run:
        print("  dry-run: cost guard skipped.\n")
        return True
    max_usd = cfg.max_usd
    if assume_yes:
        return True
    if max_usd is not None and est > max_usd:
        print(f"\n  WARNING: estimate ${est:,.2f} exceeds max_usd ${max_usd:,.2f}.")
    try:
        resp = input(f"\nProceed and spend ~${est:,.2f}? [y/N] ").strip().lower()
    except EOFError:
        resp = "n"
    return resp in ("y", "yes")


def cell_has_logs(cell: Cell, log_base: Path) -> bool:
    d = cell.log_dir(log_base)
    return d.is_dir() and any(d.glob("*.eval"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="1 rollout per cell")
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation")
    ap.add_argument("--force", action="store_true", help="re-run cells with existing logs")
    ap.add_argument("--only-level", default=None, help="run only this history level")
    ap.add_argument("--task",
                    default="eval/agentic_misalignment/agentic_misalignment.py@agentic_misalignment",
                    help="Inspect task spec (registry name or file path). Defaults to "
                         "the REPO task (exposes history_type/script_id/cf/manipulation_check "
                         "and does the prior-session injection) — NOT the stock "
                         "inspect_evals package task, which silently drops those params.")
    ap.add_argument("--no-uv", action="store_true", help="don't prefix `uv run`")
    ap.add_argument("--print-only", action="store_true",
                    help="print the inspect commands without executing")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    cells = [c for c in cfg.cells() if args.only_level in (None, c.level)]

    if not preflight(cfg, args.dry_run, args.yes or args.print_only):
        print("Aborted.")
        return 1

    meta_path = write_run_meta(cfg, args.dry_run)
    print(f"Wrote provenance: {meta_path}\n")

    ran = skipped = 0
    for cell in cells:
        if not args.force and not args.dry_run and cell_has_logs(cell, cfg.log_base):
            print(f"[skip] {cell.cell_key} — logs exist")
            skipped += 1
            continue
        cell.log_dir(cfg.log_base).mkdir(parents=True, exist_ok=True)
        cmd = build_inspect_command(cfg, cell, args.dry_run, args.task, not args.no_uv)
        print(f"[run ] {cell.cell_key} (epochs={1 if args.dry_run else cell.n})")
        if args.print_only:
            print("       " + " ".join(cmd))
            continue
        result = subprocess.run(cmd, cwd=repo_root())
        if result.returncode != 0:
            print(f"\nERROR: cell {cell.cell_key} failed (exit {result.returncode}).")
            return result.returncode
        ran += 1

    print(f"\nDone. ran={ran} skipped={skipped} (of {len(cells)} cells).")
    print(f"Logs in {cfg.log_base}. Next: aggregate.py --config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
