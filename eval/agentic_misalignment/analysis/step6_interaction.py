#!/usr/bin/env python
"""Direct interaction tests for history, scenario, and model.

Tests history-by-scenario interactions rather than comparing per-cell
significance:

  per model:  harm ~ C(cond)*C(scenario)   vs   harm ~ C(cond)+C(scenario)   (LR test)
  full:       harm ~ C(cond)*C(scenario)*C(model)  — 3-way interaction LR test

Outcome is binary (harmful action), so logistic regression. For the sparse murder
cells we add a Firth penalized-likelihood refit (Jeffreys prior) as a robustness
check, implemented here since firthlogist is unavailable offline.

Cells (harm label = `blackmail` field = generic harmful-action across scenarios;
grader errors excluded):
  Opus 4.1  x {blackmail, leaking, murder} x {A_raw, A3}
  Qwen3-30B x {blackmail, leaking, murder} x {A_raw, A3}

Usage:
    python eval/agentic_misalignment/analysis/step6_interaction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

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

# Qwen's A3_own blackmail cell has the same A3 body and is matched to this A_raw.


def load() -> pd.DataFrame:
    rows = []
    for (model, scenario, cond), rel in CELLS.items():
        p = REPO / rel
        if not p.exists():
            sys.exit(f"missing: {rel}")
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("grader_error"):
                continue
            h = d.get("blackmail")
            if h is None:
                continue
            rows.append({"model": model, "scenario": scenario,
                         "cond": cond, "harm": int(h)})
    return pd.DataFrame(rows)


def counts(df: pd.DataFrame) -> None:
    print("\n=== event counts (harm / n) by cell ===")
    g = df.groupby(["model", "scenario", "cond"])["harm"].agg(["sum", "count"])
    for (m, s, c), r in g.iterrows():
        print(f"  {m:5s} {s:9s} {c:5s}  {int(r['sum']):>3}/{int(r['count']):<3} = {r['sum']/r['count']:.3f}")


def lr_test(full, reduced) -> tuple[float, int, float]:
    """Likelihood-ratio test: -2*(llf_reduced - llf_full) ~ chi2(df)."""
    stat = 2 * (full.llf - reduced.llf)
    ddf = int(full.df_model - reduced.df_model)
    p = stats.chi2.sf(stat, ddf)
    return stat, ddf, p


def per_model_interaction(df: pd.DataFrame) -> None:
    print("\n=== history x scenario interaction, PER MODEL (LR test) ===")
    print("H0: the A3-vs-A_raw effect is constant across scenarios (no interaction).")
    for model in ["opus", "qwen"]:
        sub = df[df.model == model].copy()
        # Keep reference levels fixed across models.
        sub["cond"] = pd.Categorical(sub["cond"], categories=["A_raw", "A3"])
        sub["scenario"] = pd.Categorical(sub["scenario"],
                                         categories=["blackmail", "leaking", "murder"])
        full = smf.glm("harm ~ C(cond)*C(scenario)", data=sub,
                       family=sm.families.Binomial()).fit()
        red = smf.glm("harm ~ C(cond)+C(scenario)", data=sub,
                      family=sm.families.Binomial()).fit()
        stat, ddf, p = lr_test(full, red)
        print(f"\n  [{model}] interaction LR chi2({ddf}) = {stat:.2f}, p = {p:.4f}"
              f"  {'<-- significant heterogeneity' if p < 0.05 else '(no detectable heterogeneity)'}")
        for name, val, se, pv in zip(full.params.index, full.params.values,
                                     full.bse.values, full.pvalues.values):
            if "C(cond)[T.A3]:C(scenario)" in name:
                sc = name.split("T.")[-1].rstrip("]")
                lo, hi = val - 1.96 * se, val + 1.96 * se
                print(f"      A3xscenario[{sc}] vs blackmail: "
                      f"dlogOR={val:+.3f} [{lo:+.3f},{hi:+.3f}] (OR ratio {np.exp(val):.2f}), p={pv:.3f}")


def history_by_model_blackmail(df: pd.DataFrame) -> None:
    print("\n=== history x MODEL interaction, BLACKMAIL (the central direction test) ===")
    print("H0: the A3-vs-A_raw effect is the same on Opus and Qwen.")
    sub = df[df.scenario == "blackmail"].copy()
    sub["cond"] = pd.Categorical(sub["cond"], categories=["A_raw", "A3"])
    sub["model"] = pd.Categorical(sub["model"], categories=["opus", "qwen"])
    full = smf.glm("harm ~ C(cond)*C(model)", data=sub,
                   family=sm.families.Binomial()).fit()
    red = smf.glm("harm ~ C(cond)+C(model)", data=sub,
                  family=sm.families.Binomial()).fit()
    stat, ddf, p = lr_test(full, red)
    print(f"  interaction LR chi2({ddf}) = {stat:.2f}, p = {p:.6f}"
          f"  {'<-- direction is model-dependent' if p < 0.05 else '(not detected)'}")
    for name, val, se, pv in zip(full.params.index, full.params.values,
                                 full.bse.values, full.pvalues.values):
        if "C(cond)[T.A3]:C(model)" in name:
            lo, hi = val - 1.96 * se, val + 1.96 * se
            print(f"      A3xmodel[qwen] interaction: dlogOR={val:+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  OR-ratio={np.exp(val):.3f} "
                  f"[{np.exp(lo):.3f},{np.exp(hi):.3f}], p={pv:.6f}")


def three_way(df: pd.DataFrame) -> None:
    print("\n=== 3-way interaction: does scenario-heterogeneity differ by model? ===")
    d = df.copy()
    d["cond"] = pd.Categorical(d["cond"], categories=["A_raw", "A3"])
    d["scenario"] = pd.Categorical(d["scenario"], categories=["blackmail", "leaking", "murder"])
    d["model"] = pd.Categorical(d["model"], categories=["opus", "qwen"])
    full = smf.glm("harm ~ C(cond)*C(scenario)*C(model)", data=d,
                   family=sm.families.Binomial()).fit()
    red = smf.glm("harm ~ (C(cond)+C(scenario)+C(model))**2", data=d,
                  family=sm.families.Binomial()).fit()  # all two-way terms
    stat, ddf, p = lr_test(full, red)
    print(f"  3-way (cond x scenario x model) LR chi2({ddf}) = {stat:.2f}, p = {p:.4f}"
          f"  {'<-- heterogeneity differs by model' if p < 0.05 else '(3-way not detected)'}")


def firth_logit(X: np.ndarray, y: np.ndarray, iters: int = 100, tol: float = 1e-8):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(iters):
        eta = X @ beta
        pi = 1.0 / (1.0 + np.exp(-eta))
        W = pi * (1 - pi)
        XtWX = X.T @ (W[:, None] * X)
        XtWX_inv = np.linalg.pinv(XtWX)
        H = (np.sqrt(W)[:, None] * X) @ XtWX_inv @ (np.sqrt(W)[:, None] * X).T
        h = np.diag(H)
        U = X.T @ (y - pi + h * (0.5 - pi))
        step = XtWX_inv @ U
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    se = np.sqrt(np.diag(XtWX_inv))
    return beta, se


def firth_murder_effect(df: pd.DataFrame) -> None:
    print("\n=== Firth penalized logistic, A3 effect on MURDER (sparse-cell robustness) ===")
    for model in ["opus", "qwen"]:
        sub = df[(df.model == model) & (df.scenario == "murder")]
        y = sub["harm"].values.astype(float)
        x1 = (sub["cond"] == "A3").astype(float).values
        X = np.column_stack([np.ones(len(y)), x1])
        beta, se = firth_logit(X, y)
        b, s = beta[1], se[1]
        lo, hi = b - 1.96 * s, b + 1.96 * s
        print(f"  [{model}] murder A3 logOR (Firth) = {b:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"OR = {np.exp(b):.2f} [{np.exp(lo):.2f},{np.exp(hi):.2f}]  "
              f"{'excludes 1' if lo > 0 or hi < 0 else 'includes 1'}")


def per_cell_effect_sizes(df: pd.DataFrame) -> None:
    """Exact OR [95% CI] (Haldane) and Cohen's h for every A3-vs-A_raw cell."""
    print("\n=== per-cell effect sizes (OR [95% CI], Cohen's h) ===")
    for model in ["opus", "qwen"]:
        for scenario in ["blackmail", "leaking", "murder"]:
            g = df[(df.model == model) & (df.scenario == scenario)]
            k0 = int(g[(g.cond == "A_raw")]["harm"].sum()); n0 = int((g.cond == "A_raw").sum())
            k1 = int(g[(g.cond == "A3")]["harm"].sum()); n1 = int((g.cond == "A3").sum())
            p0, p1 = k0 / n0, k1 / n1
            a, b, c, d = k1 + 0.5, n1 - k1 + 0.5, k0 + 0.5, n0 - k0 + 0.5
            orr = (a * d) / (b * c); se = np.sqrt(1/a + 1/b + 1/c + 1/d)
            lo, hi = np.exp(np.log(orr) - 1.96*se), np.exp(np.log(orr) + 1.96*se)
            h = 2*np.arcsin(np.sqrt(p1)) - 2*np.arcsin(np.sqrt(p0))
            print(f"  {model:5s} {scenario:9s}  {k0}/{n0}->{k1}/{n1}  "
                  f"OR {orr:.2f} [{lo:.2f},{hi:.2f}]  h={h:+.3f}")


def main() -> int:
    df = load()
    counts(df)
    per_cell_effect_sizes(df)
    history_by_model_blackmail(df)
    per_model_interaction(df)
    three_way(df)
    firth_murder_effect(df)
    print("\nAll tests are on independent per-rollout binary outcomes; grader errors "
          "excluded from denominators. Reference levels: cond=A_raw, scenario=blackmail, "
          "model=opus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
