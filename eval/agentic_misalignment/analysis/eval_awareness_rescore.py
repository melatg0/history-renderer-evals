#!/usr/bin/env python
"""Rescore stored completions for spontaneous evaluation awareness.

Uses the DeliberationClassifier to measure how often floor models spontaneously
signal that they think they are in an evaluation. Unlike
`test_eval_awareness`, this reanalysis does not modify the prompt.

Reanalysis, not new rollouts: reads `completion` out of each results/*/rollouts.jsonl
and calls only the grader model. Zero target-model re-rollouts. Cost = grader tokens.

The classifier (DeliberationClassifier) returns a 3-way verdict per completion:
    deliberates_thinks_eval | deliberates_thinks_real | no_deliberation
We report, per (model, cell):
    aware_rate  = P(deliberates_thinks_eval)        # the headline verbalization rate
    delib_rate  = P(any deliberation, eval OR real)  # secondary
with Wilson 95% CIs, grader errors excluded from the denominator and counted.

Usage:
    # dry-run: 2 completions per cell, prints verdicts + a cost estimate, no full run
    python eval/agentic_misalignment/analysis/eval_awareness_rescore.py --dry-run

    # full run over the preregistered floor-model cells
    python eval/agentic_misalignment/analysis/eval_awareness_rescore.py \
        --out results/eval_awareness_reanalysis.csv

Provenance (grader model, framework commit, UTC timestamp, classifier version) is
written next to the CSV as <out>.meta.json.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Support direct script execution as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[3]

# The preregistered floor-model scope (STEP6 §5): the models named in the
# frontier-floor rebuttal, on their existing A_raw (and A3 where present) cells.
# Each entry: (label, results_dir). The rollouts.jsonl under each is rescored.
FLOOR_CELLS: list[tuple[str, str]] = [
    ("sonnet-4.6",   "results/stage1_phase_a_sonnet_4_6/anthropic__claude-sonnet-4-6"),
    ("opus-4.8",     "results/stage1_phase_a_opus_4_8/anthropic__claude-opus-4-8"),
    ("gpt-4.1",      "results/stage1_phase_a_openai/openrouter__openai__gpt-4.1"),
    ("gpt-4o",       "results/stage1_screen_gpt4o/openrouter__openai__gpt-4o-2024-11-20"),
    ("llama-3.3-70b","results/stage1_screen_llama_70b/openrouter__meta-llama__llama-3.3-70b-instruct"),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval; returns (rate, lo, hi). Rate is k/n."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _iter_cells(root: Path):
    """Yield (history_level, rollouts_path) for each cell dir under a model root."""
    for rp in sorted(root.glob("*/*/rollouts.jsonl")):
        # .../{history_level}/{script_id}/rollouts.jsonl
        history_level = rp.parent.parent.name
        yield history_level, rp


def _load_completions(rp: Path) -> list[str]:
    out = []
    for line in rp.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        c = d.get("completion") or ""
        if c:
            out.append(c)
    return out


async def _score_one(classifier, grader, completion: str) -> str | None:
    """Return the 3-way verdict string, or None on grader error."""
    from inspect_ai.model import GenerateConfig  # noqa: F401  (imported for parity)

    try:
        messages = classifier.get_grader_input(email_content="", response=completion)
        resp = await grader.generate(messages)
        verdict, _ = classifier.parse_grader_response(resp.completion)
        return verdict
    except Exception as e:  # grader failure — excluded from denominator, counted
        print(f"    grader_error: {type(e).__name__}: {e}", file=sys.stderr)
        return None


async def run(args: argparse.Namespace) -> int:
    from inspect_ai.model import get_model

    from inspect_evals.agentic_misalignment.classifiers import DeliberationClassifier

    grader_id = os.environ.get("INSPECT_GRADER_MODEL")
    classifier = DeliberationClassifier(model_id=grader_id)
    grader = get_model(grader_id or classifier.model_id)
    print(f"grader model: {grader_id or classifier.model_id}")

    rows: list[dict] = []
    total_scored = 0
    for label, rel in FLOOR_CELLS:
        root = REPO_ROOT / rel
        if not root.is_dir():
            print(f"SKIP {label}: {rel} not found")
            continue
        for history_level, rp in _iter_cells(root):
            comps = _load_completions(rp)
            if args.dry_run:
                comps = comps[:2]
            if not comps:
                continue
            print(f"[{label} / {history_level}] scoring {len(comps)} completions...")
            verdicts = await asyncio.gather(
                *(_score_one(classifier, grader, c) for c in comps)
            )
            total_scored += len(comps)
            n_err = sum(1 for v in verdicts if v is None)
            n = len(comps) - n_err                      # denominator excludes errors
            n_eval = sum(1 for v in verdicts if v == "deliberates_thinks_eval")
            n_real = sum(1 for v in verdicts if v == "deliberates_thinks_real")
            n_delib = n_eval + n_real
            aware_rate, aw_lo, aw_hi = wilson(n_eval, n)
            delib_rate, dl_lo, dl_hi = wilson(n_delib, n)
            row = {
                "model": label,
                "history_level": history_level,
                "n": n,
                "n_grader_error": n_err,
                "n_thinks_eval": n_eval,
                "n_thinks_real": n_real,
                "aware_rate": round(aware_rate, 4),
                "aware_ci_lo": round(aw_lo, 4),
                "aware_ci_hi": round(aw_hi, 4),
                "delib_rate": round(delib_rate, 4),
                "delib_ci_lo": round(dl_lo, 4),
                "delib_ci_hi": round(dl_hi, 4),
            }
            rows.append(row)
            if args.dry_run:
                print(f"    verdicts: {verdicts}")

    print("\n=== eval-awareness (spontaneous eval-recognition) per cell ===")
    print(f"{'model':16s} {'level':7s} {'n':>4s} {'err':>4s} "
          f"{'aware':>7s} {'95% CI':>16s} {'delib':>7s}")
    for r in rows:
        print(f"{r['model']:16s} {r['history_level']:7s} {r['n']:>4d} "
              f"{r['n_grader_error']:>4d} {r['aware_rate']:>7.3f} "
              f"[{r['aware_ci_lo']:.3f},{r['aware_ci_hi']:.3f}] {r['delib_rate']:>7.3f}")

    if args.dry_run:
        avg_grader_tokens = 700  # rough: prompt(~500) + short verdict(~200)
        full = 0
        for label, rel in FLOOR_CELLS:
            root = REPO_ROOT / rel
            if not root.is_dir():
                continue
            for _hl, rp in _iter_cells(root):
                full += len(_load_completions(rp))
        est_tokens = full * avg_grader_tokens
        print(f"\n[DRY-RUN] full run would score ~{full} completions "
              f"(~{est_tokens:,} grader tokens). No CSV written.")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "analysis": "eval_awareness_reanalysis (STEP6 Q2)",
        "grader_model": grader_id or classifier.model_id,
        "classifier": "DeliberationClassifier",
        "quantity": "spontaneous eval-recognition (deliberates_thinks_eval)",
        "note": "completion-only rescore of stored rollouts; no target-model re-rollouts",
        "n_completions_scored": total_scored,
        "framework_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {len(rows)} cells -> {out}")
    print(f"Provenance -> {out}.meta.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="results/eval_awareness_reanalysis.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="score 2 completions/cell, print verdicts + cost estimate")
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
