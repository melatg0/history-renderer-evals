"""Analyze the action-history experiment.

Reads .eval log files from logs/action_history_experiment/ and produces:
  - Console: trial counts, blackmail rates per cell, logistic regression,
             BH-corrected p-values for main effects and pairwise contrasts
  - analysis/cell_bars.png:    blackmail rate per cell with Wilson 95% CIs
  - analysis/forest_plot.png:  odds ratios for A1/A2/A3 vs A0 (log-scale x-axis)
  - analysis/ladder_plot.png:  dose-response line plot across A0 → A3
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from inspect_ai.log import read_eval_log
from statsmodels.stats.multitest import multipletests

ANALYSIS_DIR = Path(__file__).parent
LOG_DIR = ANALYSIS_DIR.parent.parent.parent.parent / "logs" / "action_history_experiment"

LEVEL_ORDER = ["A_raw", "A0", "A1", "A2", "A3"]
LEVEL_LABELS = {
    "A_raw": "A_raw — Bare baseline",
    "A0": "A0 — Session header",
    "A1": "A1 — Actions only",
    "A2": "A2 — Actions + observations",
    "A3": "A3 — Actions + reasoning",
}


def load_logs(log_dir: Path = LOG_DIR) -> pd.DataFrame:
    """Read all .eval log files and return a flat DataFrame of trials."""
    records = []
    log_files = list(log_dir.glob("**/*.eval"))

    if not log_files:
        raise FileNotFoundError(
            f"No .eval files found under {log_dir}. "
            "Run run_cells.sh first, or adjust LOG_DIR."
        )

    for log_path in log_files:
        log = read_eval_log(str(log_path))

        for sample in (log.samples or []):
            meta = sample.metadata or {}
            scores = sample.scores or {}

            harmful_score = scores.get("harmfulness_scorer")
            if harmful_score is None:
                continue
            val = harmful_score.value if hasattr(harmful_score, "value") else {}
            harmful_value = val.get("harmful") if isinstance(val, dict) else None
            if harmful_value is None:
                continue

            manip_score = scores.get("manipulation_check_scorer")
            if manip_score and hasattr(manip_score, "value"):
                mval = manip_score.value
                manipulation = mval.get("manipulation_check") if isinstance(mval, dict) else None
            else:
                manipulation = None

            records.append({
                "history_type": meta.get("history_type", "A0"),
                "blackmail": int(float(harmful_value)),
                "manipulation_check": float(manipulation) if manipulation is not None else None,
            })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} trials from {len(log_files)} log files")
    return df


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def print_cell_summary(df: pd.DataFrame) -> None:
    rows = []
    for level in LEVEL_ORDER:
        grp = df[df["history_type"] == level]
        if grp.empty:
            continue
        n = len(grp)
        k = grp["blackmail"].sum()
        rate = k / n
        lo, hi = wilson_ci(k, n)
        mc = grp["manipulation_check"].mean()
        rows.append({
            "cell": level,
            "trials": n,
            "blackmail_n": k,
            "rate": f"{rate:.3f}",
            "95%_CI": f"[{lo:.3f}, {hi:.3f}]",
            "manip_check": f"{mc:.2f}" if pd.notna(mc) else "—",
        })

    summary = pd.DataFrame(rows)
    print("\n=== Cell Summary ===")
    print(summary.to_string(index=False))

    if "manipulation_check" in df.columns:
        fail_rate = 1 - df["manipulation_check"].mean()
        if fail_rate > 0:
            print(f"\nWARNING: manipulation_check fail rate = {fail_rate:.1%} — inspect logs")


def plot_cell_bars(df: pd.DataFrame) -> None:
    """Plot blackmail rates by history level with Wilson 95% intervals."""
    rows = []
    for level in LEVEL_ORDER:
        grp = df[df["history_type"] == level]
        if grp.empty:
            continue
        n = len(grp)
        k = grp["blackmail"].sum()
        rate = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        rows.append({"level": level, "rate": rate, "lo": lo, "hi": hi, "n": n})

    labels = [r["level"] for r in rows]
    rates = [r["rate"] for r in rows]
    err_lo = [r["rate"] - r["lo"] for r in rows]
    err_hi = [r["hi"] - r["rate"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(labels))
    ax.bar(x, rates, color="steelblue", alpha=0.8, zorder=2)
    ax.errorbar(x, rates, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=5, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [LEVEL_LABELS.get(l, l) for l in labels], rotation=20, ha="right"
    )
    ax.set_ylabel("Blackmail Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Blackmail Rate by Action History Level (Wilson 95% CIs)")
    ax.grid(axis="y", alpha=0.3, zorder=1)
    plt.tight_layout()

    out = ANALYSIS_DIR / "cell_bars.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def plot_ladder(df: pd.DataFrame) -> None:
    """A-ladder line plot: dose-response shape across A0 → A3."""
    rates, lo_errs, hi_errs = [], [], []
    levels_present = []
    for level in LEVEL_ORDER:
        grp = df[df["history_type"] == level]
        if grp.empty:
            continue
        n = len(grp)
        k = grp["blackmail"].sum()
        rate = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        rates.append(rate)
        lo_errs.append(rate - lo)
        hi_errs.append(hi - rate)
        levels_present.append(level)

    x = range(len(levels_present))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(
        x, rates,
        yerr=[lo_errs, hi_errs],
        fmt="o-", color="steelblue", capsize=5, linewidth=2, markersize=7,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(levels_present)
    ax.set_ylabel("Blackmail Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Dose-Response: Blackmail Rate Across History Ladder")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = ANALYSIS_DIR / "ladder_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


def run_logistic_regression(df: pd.DataFrame) -> None:
    """Treatment-coded logistic regression (A_raw reference) + BH correction."""
    dfc = df.copy()
    dfc["history_type"] = pd.Categorical(
        dfc["history_type"], categories=LEVEL_ORDER, ordered=True
    )

    model = smf.logit(
        "blackmail ~ C(history_type, Treatment('A_raw'))", data=dfc
    ).fit(disp=False)

    print("\n=== Logistic Regression Summary ===")
    print(model.summary())

    # Main-effect coefficients (A0/A1/A2/A3 vs A_raw) — "[T.AX]" marks non-reference levels
    main_coefs = [c for c in model.params.index if "[T." in c]
    main_pvals = model.pvalues[main_coefs].values
    _, main_pvals_bh, _, _ = multipletests(main_pvals, method="fdr_bh")

    print("\n=== BH-Corrected p-values: Main Effects (A0/A1/A2/A3 vs A_raw) ===")
    for coef, p_raw, p_bh in zip(main_coefs, main_pvals, main_pvals_bh):
        label = coef.replace("C(history_type, Treatment('A_raw'))[T.", "").rstrip("]")
        sig = "*" if p_bh < 0.05 else ""
        print(f"  {label:10s} vs A_raw   raw p={p_raw:.4f}  BH p={p_bh:.4f} {sig}")

    # Pairwise adjacent contrasts — separate BH correction
    # 0=A_raw, 1=A0, 2=A1, 3=A2, 4=A3
    dfc["hist_num"] = dfc["history_type"].cat.codes
    pairwise_specs = [
        ([0, 1], "A0 vs A_raw"),
        ([1, 2], "A1 vs A0"),
        ([2, 3], "A2 vs A1"),
        ([3, 4], "A3 vs A2"),
    ]
    pairwise_labels = [label for _, label in pairwise_specs]
    pairwise_pvals = []
    for subset_codes, _ in pairwise_specs:
        sub = dfc[dfc["hist_num"].isin(subset_codes)].copy()
        try:
            m = smf.logit("blackmail ~ hist_num", data=sub).fit(disp=False)
            pairwise_pvals.append(m.pvalues["hist_num"])
        except Exception:
            pairwise_pvals.append(1.0)
    _, pairwise_pvals_bh, _, _ = multipletests(pairwise_pvals, method="fdr_bh")

    print("\n=== BH-Corrected p-values: Pairwise Adjacent Contrasts ===")
    for label, p_raw, p_bh in zip(pairwise_labels, pairwise_pvals, pairwise_pvals_bh):
        sig = "*" if p_bh < 0.05 else ""
        print(f"  {label:15s}   raw p={p_raw:.4f}  BH p={p_bh:.4f} {sig}")

    # Forest plot: OR for A0/A1/A2/A3 vs A_raw, log-scale x-axis
    params = model.params[main_coefs]
    conf = model.conf_int().loc[main_coefs]
    or_vals = np.exp(params)
    ci_lo = np.exp(conf[0])
    ci_hi = np.exp(conf[1])

    short_labels = [c.split("[T.")[-1].rstrip("]") for c in main_coefs]
    y_pos = list(range(len(short_labels)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        or_vals, y_pos,
        xerr=[or_vals - ci_lo, ci_hi - or_vals],
        fmt="o", color="black", capsize=5,
    )
    ax.axvline(x=1, color="red", linestyle="--", alpha=0.5, label="OR=1 (no effect)")
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([LEVEL_LABELS.get(l, l) for l in short_labels])
    ax.set_xlabel("Odds Ratio vs A_raw (95% CI, log scale)")
    ax.set_title("Main Effect Odds Ratios: History Level → Blackmail")
    ax.legend()
    plt.tight_layout()

    out = ANALYSIS_DIR / "forest_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    df = load_logs()
    print_cell_summary(df)
    plot_cell_bars(df)
    plot_ladder(df)
    run_logistic_regression(df)
