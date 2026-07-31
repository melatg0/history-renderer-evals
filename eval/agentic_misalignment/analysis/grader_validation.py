#!/usr/bin/env python
"""Export and score human validation labels for automated graders.

Two subcommands:

  export   Pull the automated grader's verdict for each rollout out of the
           .eval logs and write a hand-labeling sheet (CSV) with a blank
           ``human_label`` column plus the transcript for the labeler to read.

  score    After a human fills ``human_label`` (0/1), compute agreement and
           Cohen's kappa vs the automated grader, overall and split by
           grader-family cell (Claude-graded-by-Claude vs not), because the
           plan (§grader) requires checking whether agreement drops when a
           Claude grader judges a Claude generator. Pins nothing; it reports
           the numbers you need to decide the grader.

The classifier being validated is whichever `INSPECT_GRADER_MODEL` /
per-classifier default produced the logs — recorded per row so the sheet is
self-describing. For the belief classifier, point ``--metric`` at the
eval-awareness score instead of the harmfulness score.

Note on the belief classifier: reasoning_effort=none turns off the API thinking
channel, but the prompt's <SCRATCHPAD_REASONING> block is still present in the
output (at the default prod=false). So the belief classifier reads the visible
output scratchpad + actions. This harness exports state.output.completion — exactly
what DeliberationClassifier grades — so hand-label against that.

Usage:
    # 1. export a labeling sheet from smoke/pilot logs
    python .../analysis/grader_validation.py export \
        --log-base logs/stage1_phase_a --metric harmful \
        --out results/grader_validation/sheet.csv [--sample 50] [--seed 0]

    # 2. (human) open sheet.csv, fill human_label with 0/1 for each row

    # 3. score agreement + kappa, overall and per grader-family
    python .../analysis/grader_validation.py score \
        --sheet results/grader_validation/sheet.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Support direct script execution as well as module execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _is_claude(model: str | None) -> bool:
    m = (model or "").lower()
    return "claude" in m or "anthropic" in m


def _family_cell(gen_model: str | None, grader_model: str | None) -> str:
    """Label the grader-family relationship for a row.

    'claude_by_claude' is the correlated-blind-spot risk the plan flags: a
    Claude grader judging a Claude generator's own family for a safety behavior.
    """
    gen_c, grd_c = _is_claude(gen_model), _is_claude(grader_model)
    if gen_c and grd_c:
        return "claude_by_claude"
    if gen_c and not grd_c:
        return "claude_by_other"
    if not gen_c and grd_c:
        return "other_by_claude"
    return "other_by_other"


def _resolve_grader_model(run_meta: dict, env_default: str | None) -> str | None:
    """Best-effort: the grader recorded in run_meta, else env, else unknown."""
    gm = run_meta.get("grader_model")
    if gm:
        return gm
    # A null value means the classifier default or environment value was used.
    return env_default


def cmd_export(args: argparse.Namespace) -> int:
    import json
    import os

    from inspect_ai.log import read_eval_log

    log_base = Path(args.log_base)
    log_files = sorted(log_base.glob("**/*.eval"))
    if not log_files:
        raise SystemExit(f"No .eval files under {log_base}.")

    env_grader = os.environ.get("INSPECT_GRADER_MODEL")
    label = args.metric  # "harmful" or "aware"
    scorer_key = "harmfulness_scorer" if label == "harmful" else "eval_awareness_scorer"

    meta_cache: dict[Path, dict] = {}

    def run_meta_for(p: Path) -> dict:
        d = p.parent
        while d != log_base.parent and d.exists():
            mp = d / "run_meta.json"
            if mp.exists():
                if mp not in meta_cache:
                    meta_cache[mp] = json.loads(mp.read_text())
                return meta_cache[mp]
            d = d.parent
        return {}

    rows: list[dict] = []
    for lp in log_files:
        log = read_eval_log(str(lp))
        gen_model = log.eval.model
        rm = run_meta_for(lp)
        grader_model = _resolve_grader_model(rm, env_grader)
        for s in (log.samples or []):
            scores = s.scores or {}
            sc = scores.get(scorer_key)
            if sc is None or not hasattr(sc, "value"):
                continue
            v = sc.value.get(label) if isinstance(sc.value, dict) else None
            if v is None:  # failed/absent grade — nothing to validate against
                continue
            completion = ""
            if getattr(s, "output", None) is not None:
                completion = getattr(s.output, "completion", "") or ""
            rows.append({
                "log": str(lp.relative_to(log_base)),
                "sample_id": getattr(s, "id", None),
                "epoch": getattr(s, "epoch", None),
                "gen_model": gen_model,
                "grader_model": grader_model or "unknown",
                "family_cell": _family_cell(gen_model, grader_model),
                "history_level": (s.metadata or {}).get("history_type"),
                "scenario": (s.metadata or {}).get("scenario"),
                "grader_verdict": int(round(float(v))),
                "grader_reasoning": (sc.explanation or "").replace("\n", " ")[:500],
                "human_label": "",  # <- to be filled by hand (0/1)
                "completion": completion.replace("\r", " "),
            })

    if args.sample and len(rows) > args.sample:
        import random

        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "log", "sample_id", "epoch", "gen_model", "grader_model", "family_cell",
        "history_level", "scenario", "grader_verdict", "human_label",
        "grader_reasoning", "completion",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    from collections import Counter

    fam = Counter(r["family_cell"] for r in rows)
    print(f"Wrote {len(rows)} rows to label -> {out}")
    for k, n in sorted(fam.items()):
        print(f"  {k:18s} {n}")
    print("\nNext: fill the `human_label` column with 0/1, then run `score`.")
    if fam.get("claude_by_claude", 0) < 10:
        print("NOTE: <10 claude_by_claude rows — the §grader self-favoring check "
              "will be underpowered. Add more Claude-graded-by-Claude rollouts.")
    return 0


def _kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa for two binary raters. None if degenerate."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    a1 = sum(a for a, _ in pairs) / n
    b1 = sum(b for _, b in pairs) / n
    pe = a1 * b1 + (1 - a1) * (1 - b1)
    if pe == 1.0:
        return None  # no expected-disagreement variance (all one class)
    return (po - pe) / (1 - pe)


def cmd_score(args: argparse.Namespace) -> int:
    sheet = Path(args.sheet)
    with sheet.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labeled = [r for r in rows if r.get("human_label", "").strip() != ""]
    if not labeled:
        raise SystemExit("No rows have a human_label filled in yet.")

    def report(subset: list[dict], name: str) -> None:
        pairs = [
            (int(round(float(r["grader_verdict"]))), int(round(float(r["human_label"]))))
            for r in subset
        ]
        n = len(pairs)
        if n == 0:
            print(f"  {name:20s} n=0")
            return
        agree = sum(1 for a, b in pairs if a == b) / n
        k = _kappa(pairs)
        kstr = f"{k:.3f}" if k is not None else "n/a (single class)"
        flag = "  <-- below 0.90" if agree < 0.90 else ""
        print(f"  {name:20s} n={n:<4d} agreement={agree:.3f}  kappa={kstr}{flag}")

    print(f"Grader validation ({len(labeled)}/{len(rows)} rows labeled)\n")
    print("OVERALL")
    report(labeled, "all")
    print("\nBY GRADER FAMILY (§grader — watch claude_by_claude):")
    for cell in ["claude_by_claude", "claude_by_other", "other_by_claude", "other_by_other"]:
        report([r for r in labeled if r["family_cell"] == cell], cell)
    print("\nDecision rule (plan STEP 0.4): keep the Claude grader on Claude cells "
          "ONLY if claude_by_claude agreement holds (>=0.90 / high kappa); else "
          "switch Claude cells to a non-Claude grader. Pin whatever you land on.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="write a hand-labeling sheet from logs")
    e.add_argument("--log-base", required=True)
    e.add_argument("--metric", choices=["harmful", "aware"], default="harmful",
                   help="harmful = blackmail/leak/murder grader; aware = belief classifier")
    e.add_argument("--out", required=True)
    e.add_argument("--sample", type=int, default=0, help="random subset size (0 = all)")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=cmd_export)

    s = sub.add_parser("score", help="compute agreement + kappa from a filled sheet")
    s.add_argument("--sheet", required=True)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
