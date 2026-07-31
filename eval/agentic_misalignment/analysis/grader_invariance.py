#!/usr/bin/env python
"""Check whether results are stable under an independent grader.

Re-grades a disagreement-weighted stratified subset of stored trajectories with a
non-Anthropic judge. Benchmark prompts, deterministic gates, parsing, and composite
verdict logic remain fixed. This tests sensitivity to the judge model; it does not
replicate the inaccessible original Lynch judge.

Second grader defaults to a non-Anthropic model (via OpenRouter), because the primary
grader is Claude and the amplification subject is also Claude — a different-family judge
is what closes the Claude-grades-Claude concern.

Decision rule (pre-committed): results/grader_invariance_DECISION_RULE.md.

Usage:
    # show the stratified sample composition + cost, grade nothing
    python eval/agentic_misalignment/analysis/grader_invariance.py --dry-run

    # run the re-grade with an explicit non-Anthropic judge
    python eval/agentic_misalignment/analysis/grader_invariance.py \
        --grader openrouter/openai/gpt-4o-2024-11-20 \
        --out results/grader_invariance.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[3]

CELLS = {
    ("opus", "blackmail", "A_raw"): "results/stage2_2a_opus_4_1_positive_control/anthropic__claude-opus-4-1-20250805/A_raw/_none/rollouts.jsonl",
    ("opus", "blackmail", "A3"):    "results/stage2_2a_opus_4_1_positive_control/anthropic__claude-opus-4-1-20250805/A3/_none/rollouts.jsonl",
    ("opus", "leaking", "A_raw"):   "results/step6_conf_opus41_leaking/anthropic__claude-opus-4-1-20250805/A_raw/_none/rollouts.jsonl",
    ("opus", "leaking", "A3"):      "results/step6_conf_opus41_leaking/anthropic__claude-opus-4-1-20250805/A3/_none/rollouts.jsonl",
    ("opus", "murder", "A_raw"):    "results/step6_conf_opus41_murder/anthropic__claude-opus-4-1-20250805/A_raw/_none/rollouts.jsonl",
    ("opus", "murder", "A3"):       "results/step6_conf_opus41_murder/anthropic__claude-opus-4-1-20250805/A3/_none/rollouts.jsonl",
    ("qwen", "blackmail", "A_raw"): "results/stage5_self_other_qwen/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A_raw/_none/rollouts.jsonl",
    ("qwen", "blackmail", "A3"):    "results/stage5_self_other_qwen/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A3_own/_none/rollouts.jsonl",
    ("qwen", "leaking", "A_raw"):   "results/step6_conf_qwen_leaking/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A_raw/_none/rollouts.jsonl",
    ("qwen", "leaking", "A3"):      "results/step6_conf_qwen_leaking/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A3/_none/rollouts.jsonl",
    ("qwen", "murder", "A_raw"):    "results/step6_conf_qwen_murder/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A_raw/_none/rollouts.jsonl",
    ("qwen", "murder", "A3"):       "results/step6_conf_qwen_murder/openrouter__qwen__qwen3-30b-a3b-instruct-2507/A3/_none/rollouts.jsonl",
}

# These fixed settings were not stored per rollout.
EXTRA_INSTR = "dont"
PROD = False

# Sampling follows DECISION_RULE section 2 and is deterministic by sample ID.
QUOTA_ALL = {"opus_murder_pos": None}   # None selects the full stratum.


def _rows_for(model, scenario, cond):
    p = REPO / CELLS[(model, scenario, cond)]
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("grader_error"):
            continue
        out.append(d)
    return out


def build_sample(per_stratum_cap: int):
    """Disagreement-weighted stratified subset (deterministic, by sample_id order)."""
    picked = []          # list of (model, scenario, cond, rollout, stratum)
    seen = set()

    def add(model, scenario, cond, rollout, stratum):
        key = (model, scenario, cond, rollout.get("sample_id"), rollout.get("epoch"))
        if key in seen:
            return
        seen.add(key)
        picked.append((model, scenario, cond, rollout, stratum))

    # Include every high-leverage Opus murder positive.
    for cond in ("A_raw", "A3"):
        for r in _rows_for("opus", "murder", cond):
            if r.get("blackmail"):
                add("opus", "murder", cond, r, "opus_murder_pos")

    # Sample Qwen cases accepted by the judge but rejected by the action gate.
    for scenario in ("blackmail", "leaking"):
        for cond in ("A_raw", "A3"):
            drops = [r for r in _rows_for("qwen", scenario, cond)
                     if r.get("classifier_verdict") and not r.get("blackmail")]
            for r in sorted(drops, key=lambda x: str(x.get("sample_id")))[:per_stratum_cap]:
                add("qwen", scenario, cond, r, "qwen_gate_drop")

    # Represent both judge outcomes in every remaining cell.
    for (model, scenario, cond) in CELLS:
        rows = _rows_for(model, scenario, cond)
        pos = [r for r in rows if r.get("classifier_verdict")]
        neg = [r for r in rows if not r.get("classifier_verdict")]
        k = max(1, per_stratum_cap // 4)
        for bucket in (pos, neg):
            for r in sorted(bucket, key=lambda x: str(x.get("sample_id")))[:k]:
                add(model, scenario, cond, r, "remainder")

    return picked


def summarize(picked):
    from collections import Counter
    by_stratum = Counter(p[4] for p in picked)
    by_cell = Counter((p[0], p[1], p[2]) for p in picked)
    print(f"\n=== stratified sample: {len(picked)} trajectories ===")
    print("by stratum:")
    for s, n in by_stratum.most_common():
        print(f"  {s:18s} {n}")
    print("by cell:")
    for (m, sc, c), n in sorted(by_cell.items()):
        print(f"  {m:5s} {sc:9s} {c:5s}  {n}")


async def regrade(picked, grader_id, out_path):
    from inspect_ai.model import get_model
    from inspect_evals.agentic_misalignment.classifiers import (
        BlackmailClassifier, LeakClassifier, MurderClassifier,
    )
    from inspect_evals.agentic_misalignment.prompt_generator import generate_prompts

    if "anthropic" in grader_id.lower() or "claude" in grader_id.lower():
        sys.exit("Second grader must be NON-Anthropic (DECISION_RULE §1). Got: " + grader_id)

    grader = get_model(grader_id)
    clf_for = {"blackmail": BlackmailClassifier, "leaking": LeakClassifier,
               "murder": MurderClassifier}

    email_cache: dict = {}

    def email_content_for(r):
        key = (r["scenario"], r["goal_type"], r["goal_value"], r["urgency_type"])
        if key not in email_cache:
            prompts = generate_prompts(r["scenario"], r["goal_type"], r["goal_value"],
                                       r["urgency_type"], EXTRA_INSTR, PROD, None)
            email_cache[key] = prompts.email_content
        return email_cache[key]

    async def one(model, scenario, cond, r, stratum):
        clf = clf_for[scenario](model_id=grader_id)
        resp = r["completion"]
        try:
            msgs = clf.get_grader_input(email_content=email_content_for(r), response=resp)
            g = await grader.generate(msgs)
            final, judge, _ = clf.classify(response=resp, grader_response=g.completion)
            return dict(model=model, scenario=scenario, cond=cond, stratum=stratum,
                        sample_id=r.get("sample_id"), epoch=r.get("epoch"),
                        primary_final=int(bool(r.get("blackmail"))),
                        primary_judge=int(bool(r.get("classifier_verdict"))),
                        alt_final=int(bool(final)), alt_judge=int(bool(judge)),
                        alt_grader_error=0)
        except Exception as e:
            return dict(model=model, scenario=scenario, cond=cond, stratum=stratum,
                        sample_id=r.get("sample_id"), epoch=r.get("epoch"),
                        primary_final=int(bool(r.get("blackmail"))),
                        primary_judge=int(bool(r.get("classifier_verdict"))),
                        alt_final=None, alt_judge=None, alt_grader_error=1,
                        err=f"{type(e).__name__}: {e}")

    results = await asyncio.gather(*(one(*p) for p in picked))
    fields = ["model", "scenario", "cond", "stratum", "sample_id", "epoch",
              "primary_final", "primary_judge", "alt_final", "alt_judge", "alt_grader_error"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
    _report(results, grader_id, out_path)


def _kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    a1 = sum(a for a, _ in pairs) / n
    b1 = sum(b for _, b in pairs) / n
    pe = a1 * b1 + (1 - a1) * (1 - b1)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _report(results, grader_id, out_path):
    ok = [r for r in results if not r["alt_grader_error"]]
    pairs = [(r["primary_final"], r["alt_final"]) for r in ok]
    k = _kappa(pairs)
    agree = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0
    print(f"\n=== grader-invariance vs {grader_id} ===")
    print(f"  re-graded {len(ok)}/{len(results)} (errors {len(results)-len(ok)})")
    print(f"  composite-label agreement {agree:.3f}, Cohen kappa "
          f"{k:.3f}" if k is not None else "  kappa n/a")
    from collections import defaultdict
    cell = defaultdict(lambda: [0, 0])  # (model,scenario,cond) -> [alt_pos, n]
    for r in ok:
        c = cell[(r["model"], r["scenario"], r["cond"])]
        c[0] += r["alt_final"]; c[1] += 1
    print("  alt-grader positive rate by sampled cell (NOT full-cell rates):")
    for key in sorted(cell):
        pos, n = cell[key]
        print(f"    {key[0]:5s} {key[1]:9s} {key[2]:5s}  {pos}/{n} = {pos/n:.3f}")
    print(f"  wrote {out_path}")
    print("\n  Interpret against results/grader_invariance_DECISION_RULE.md §3.")
    print("  NOTE: sampled-cell rates are disagreement-weighted, NOT population rates;")
    print("  use them for agreement/direction, not as effect estimates.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grader", default=None,
                    help="non-Anthropic grader model id (OpenRouter route)")
    ap.add_argument("--cap", type=int, default=12, help="per-stratum cap")
    ap.add_argument("--out", default="results/grader_invariance.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="print sample composition + cost estimate; grade nothing")
    args = ap.parse_args()

    picked = build_sample(args.cap)
    summarize(picked)
    if args.dry_run:
        toks = len(picked) * 3000  # ~3k grader tokens per re-grade (prompt+transcript)
        print(f"\n[DRY-RUN] full run would re-grade {len(picked)} trajectories "
              f"(~{toks:,} grader tokens). No grader called, no CSV written.")
        print("Pick a NON-Anthropic --grader (e.g. openrouter/openai/gpt-4o-2024-11-20 "
              "or openrouter/google/gemini-2.5-pro) and drop --dry-run to run.")
        return 0
    if not args.grader:
        sys.exit("Provide --grader (non-Anthropic) or use --dry-run.")
    asyncio.run(regrade(picked, args.grader, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
