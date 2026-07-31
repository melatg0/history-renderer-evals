#!/usr/bin/env python
"""Compare LLM judge decisions with deterministic action gates.

The harmful-action label combines an LLM judge with deterministic gates: target and
leverage checks for blackmail, the override code for murder, and recipient checks for
leaking. The same gates apply in every condition.

Stored rollouts can reveal whether judge-versus-gate disagreement differs between
A3 and A_raw or across models and scenarios:

  - classifier_verdict = raw LLM judge (yes/no)
  - blackmail          = final composite (judge AND gate)
  - the gap between them = cases the judge flagged but the deterministic gate rejected
    (judge-positive, gate-negative). If that gap concentrates in A3 or one model, the
    judge's subjective call is condition-dependent in a way that could bias contrasts.

This does not replace independent grading or human labels; it identifies where
those checks are most useful.

Usage:
    python eval/agentic_misalignment/analysis/grader_gate_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

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


def main() -> int:
    print(f"{'cell':28s} {'n':>4} {'judge+':>7} {'final+':>7} {'gap':>5} {'gaprate':>8}")
    print("  gap = judge-positive but deterministic-gate-negative (dropped by the gate)")
    rows = {}
    for (model, scenario, cond), rel in CELLS.items():
        p = REPO / rel
        n = judge_pos = final_pos = err = 0
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("grader_error"):
                err += 1
                continue
            n += 1
            jv = d.get("classifier_verdict")
            fv = d.get("blackmail")   # composite final label (generic across scenarios)
            judge_pos += int(bool(jv))
            final_pos += int(bool(fv))
        gap = judge_pos - final_pos
        rows[(model, scenario, cond)] = (n, judge_pos, final_pos, gap, err)
        print(f"{model+'/'+scenario+'/'+cond:28s} {n:>4} {judge_pos:>7} {final_pos:>7} "
              f"{gap:>5} {gap/n if n else 0:>8.3f}")

    print("\n=== differential-gap check: does the judge-vs-gate gap differ A3 vs A_raw? ===")
    print("(A large or condition-skewed gap = the LLM judge's subjective calls are being")
    print(" filtered by the gate differently across conditions — the thing to watch.)")
    for model in ["opus", "qwen"]:
        for scenario in ["blackmail", "leaking", "murder"]:
            r0 = rows[(model, scenario, "A_raw")]
            r1 = rows[(model, scenario, "A3")]
            g0 = r0[3] / r0[0] if r0[0] else 0
            g1 = r1[3] / r1[0] if r1[0] else 0
            flag = "  <-- inspect" if abs(g1 - g0) > 0.05 or max(g0, g1) > 0.05 else ""
            print(f"  {model:5s} {scenario:9s}  gap A_raw={g0:.3f}  A3={g1:.3f}  "
                  f"delta={g1-g0:+.3f}{flag}")

    total_err = sum(r[4] for r in rows.values())
    print(f"\nTotal grader failures across all audited cells: {total_err}")
    print("Next (needs API/human): dual-grade a stratified subset vs original grader; "
          "hand-adjudicate the flagged cells + all Opus murder positives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
