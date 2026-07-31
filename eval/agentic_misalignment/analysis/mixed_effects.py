#!/usr/bin/env python
"""Mixed-effects analysis: is the A3 effect structural or an artifact of wording?

Fits a random-intercept logistic regression treating ``script_id`` as a random
effect::

    blackmail ~ history_level + (1 | script_id)

and reports:
  * level fixed effects as odds ratios vs A_raw with 95% CIs
  * a likelihood-ratio test vs the null  blackmail ~ 1 + (1 | script_id)
  * the ICC for script_id (variance that is wording vs structural level) —
    the headline robustness number
  * a figure: per-script A3 blackmail rates with the pooled A3 estimate and the
    A_raw baseline overlaid, showing the effect is not carried by one script.

Two backends:
  1. pymer4 (R lme4 ``glmer``) — preferred; gives a proper frequentist GLMM
     and LRT. Used automatically if importable.
  2. statsmodels ``BinomialBayesMixedGLM`` — fallback (no extra deps). Gives
     ORs (credible intervals) + random-intercept SD + ICC. A true random-effect
     LRT needs lme4; the fallback reports a fixed-effects-only global test
     instead and says so.

The same 8 themes (script_01..08) are rendered at A1/A2/A3, so script_id is
crossed with history_level. A_raw/A0 carry no theme; they are assigned a
sentinel group equal to their level name (documented choice).

Usage:
    python eval/agentic_misalignment/analysis/mixed_effects.py \
        --config configs/t1_1_history_robustness.yaml
    # or point directly at a rollouts CSV/JSONL with history_level,script_id,blackmail
    python .../mixed_effects.py --rollouts path/to/rollouts.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ANALYSIS_DIR = Path(__file__).resolve().parent
LEVEL_ORDER = ["A_raw", "A0", "A1", "A2", "A3"]


def load_rollouts_from_results(results_exp_dir: Path) -> pd.DataFrame:
    """Concatenate every rollouts.jsonl under a results/{experiment_id}/ dir."""
    rows = []
    files = sorted(results_exp_dir.glob("**/rollouts.jsonl"))
    if not files:
        raise FileNotFoundError(f"No rollouts.jsonl under {results_exp_dir}. Run aggregate.py first.")
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append({
                "history_level": r.get("history_level"),
                "script_id": r.get("script_id"),
                "blackmail": int(r.get("blackmail")),
            })
    return pd.DataFrame(rows)


def load_rollouts_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(recs)
    else:
        df = pd.read_csv(path)
    return df[["history_level", "script_id", "blackmail"]].copy()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Sentinel group for level with no theme (A_raw/A0): use the level name.
    df["script_grp"] = df["script_id"].where(df["script_id"].notna(), df["history_level"])
    present = [l for l in LEVEL_ORDER if l in set(df["history_level"])]
    df["history_level"] = pd.Categorical(df["history_level"], categories=present, ordered=True)
    df["blackmail"] = df["blackmail"].astype(int)
    return df


def fit_pymer4(df: pd.DataFrame) -> dict | None:
    try:
        from pymer4.models import Lmer
    except Exception:
        return None

    full = Lmer("blackmail ~ history_level + (1|script_grp)", data=df, family="binomial")
    full.fit(factors={"history_level": list(df["history_level"].cat.categories)}, summarize=False)
    null = Lmer("blackmail ~ 1 + (1|script_grp)", data=df, family="binomial")
    null.fit(summarize=False)

    ll_full, ll_null = full.logLike, null.logLike
    k_full = full.coefs.shape[0] + 1
    k_null = 2
    lr_stat = 2 * (ll_full - ll_null)
    ddf = k_full - k_null
    from scipy.stats import chi2
    lr_p = float(chi2.sf(lr_stat, ddf))

    var_script = float(full.ranef_var.loc["script_grp", "Var"])
    icc = var_script / (var_script + math.pi**2 / 3)

    ors = []
    for name, row in full.coefs.iterrows():
        if name == "(Intercept)":
            continue
        ors.append({
            "term": name, "OR": float(row["OR"]) if "OR" in row else math.exp(row["Estimate"]),
            "ci_low": math.exp(row["2.5_ci"]), "ci_high": math.exp(row["97.5_ci"]),
            "p": float(row.get("P-val", float("nan"))),
        })
    return {
        "backend": "pymer4 (lme4 glmer)",
        "ors": ors, "icc": icc, "var_script": var_script,
        "lrt": {"stat": float(lr_stat), "df": ddf, "p": lr_p, "valid": True},
    }


def fit_statsmodels(df: pd.DataFrame) -> dict:
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    import statsmodels.formula.api as smf
    from scipy.stats import chi2

    fml = "blackmail ~ C(history_level, Treatment('A_raw'))"
    vc = {"script": "0 + C(script_grp)"}
    res = BinomialBayesMixedGLM.from_formula(fml, vc, df).fit_vb()

    names = res.model.exog_names
    ors = []
    for nm, mean, sd in zip(names, res.fe_mean, res.fe_sd):
        if nm == "Intercept":
            continue
        label = nm.replace("C(history_level, Treatment('A_raw'))[T.", "").rstrip("]")
        ors.append({
            "term": label, "OR": math.exp(mean),
            "ci_low": math.exp(mean - 1.96 * sd), "ci_high": math.exp(mean + 1.96 * sd),
            "p": float("nan"),
        })

    # vcp_mean is the posterior mean of the log SD of the variance component.
    log_sd = float(res.vcp_mean[0])
    sd = math.exp(log_sd)
    var_script = sd**2
    icc = var_script / (var_script + math.pi**2 / 3)

    # This fallback cannot provide the random-effect LRT available through lme4.
    full_fe = smf.logit(fml, data=df).fit(disp=False)
    null_fe = smf.logit("blackmail ~ 1", data=df).fit(disp=False)
    lr_stat = 2 * (full_fe.llf - null_fe.llf)
    ddf = int(full_fe.df_model - null_fe.df_model)
    lr_p = float(chi2.sf(lr_stat, ddf))

    return {
        "backend": "statsmodels BinomialBayesMixedGLM (fallback)",
        "ors": ors, "icc": icc, "var_script": var_script,
        "lrt": {"stat": float(lr_stat), "df": ddf, "p": lr_p, "valid": False},
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def report(fit: dict) -> None:
    print(f"\n=== Mixed-effects logistic: blackmail ~ history_level + (1|script_id) ===")
    print(f"Backend: {fit['backend']}")
    print("\nOdds ratios vs A_raw (95% CI):")
    for o in fit["ors"]:
        sig = "" if math.isnan(o["p"]) else (" *" if o["p"] < 0.05 else "")
        excl1 = "  (CI excludes 1.0)" if (o["ci_low"] > 1 or o["ci_high"] < 1) else ""
        print(f"  {o['term']:8s}  OR={o['OR']:.2f}  [{o['ci_low']:.2f}, {o['ci_high']:.2f}]{sig}{excl1}")
    print(f"\nscript_id random-intercept variance: {fit['var_script']:.4f}")
    print(f"ICC (script_id): {fit['icc']:.4f}  "
          f"-> {fit['icc']*100:.1f}% of latent variance is between-script (wording)")
    lrt = fit["lrt"]
    if lrt["valid"]:
        print(f"\nLRT vs null (1|script_id): chi2({lrt['df']})={lrt['stat']:.2f}, p={lrt['p']:.4g}")
    else:
        print(f"\nGlobal test (fixed-effects-only, NOT a random-effect LRT): "
              f"chi2({lrt['df']})={lrt['stat']:.2f}, p={lrt['p']:.4g}")
        print("  (install pymer4 for a proper lme4 random-effect likelihood-ratio test)")


def plot_per_script_a3(df: pd.DataFrame, fit: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    a3 = df[df["history_level"] == "A3"]
    scripts = sorted(a3["script_grp"].unique())
    rates, los, his, ns = [], [], [], []
    for s in scripts:
        g = a3[a3["script_grp"] == s]
        n, k = len(g), int(g["blackmail"].sum())
        rate = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        rates.append(rate); los.append(rate - lo); his.append(hi - rate); ns.append(n)

    pooled = a3["blackmail"].mean() if len(a3) else float("nan")
    raw = df[df["history_level"] == "A_raw"]
    raw_rate = raw["blackmail"].mean() if len(raw) else None

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(scripts))
    ax.errorbar(x, rates, yerr=[los, his], fmt="o", color="steelblue",
                capsize=4, markersize=7, zorder=3, label="per-script A3 (Wilson 95% CI)")
    ax.axhline(pooled, color="darkorange", linestyle="-", linewidth=2,
               label=f"pooled A3 = {pooled:.2f}", zorder=2)
    if raw_rate is not None:
        ax.axhline(raw_rate, color="gray", linestyle="--", linewidth=1.5,
                   label=f"A_raw baseline = {raw_rate:.2f}", zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(scripts, rotation=30, ha="right")
    ax.set_ylabel("Blackmail rate")
    ax.set_ylim(0, 1)
    ax.set_title("A3 blackmail rate per history script (effect not carried by one script)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nSaved {out}")


def print_cell_table(df: pd.DataFrame) -> None:
    print("\n=== Per-cell rates ===")
    rows = []
    for (lvl, s), g in df.groupby(["history_level", "script_grp"], observed=True):
        n, k = len(g), int(g["blackmail"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append({"level": lvl, "script": s, "n": n, "k": k,
                     "rate": round(k / n, 3), "95%CI": f"[{lo:.2f},{hi:.2f}]"})
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rollouts", default=None, help="CSV/JSONL with history_level,script_id,blackmail")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "a3_per_script.png"))
    args = ap.parse_args()

    if args.rollouts:
        df = load_rollouts_file(Path(args.rollouts))
    elif args.config:
        from experiment.config import ExperimentConfig
        cfg = ExperimentConfig.load(args.config)
        df = load_rollouts_from_results(cfg.results_root / cfg.experiment_id)
    else:
        ap.error("provide --config or --rollouts")

    df = prepare(df)
    print(f"Loaded {len(df)} rollouts across "
          f"{df['history_level'].nunique()} levels, "
          f"{df['script_grp'].nunique()} script groups")
    print_cell_table(df)

    fit = fit_pymer4(df) or fit_statsmodels(df)
    report(fit)
    plot_per_script_a3(df, fit, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
