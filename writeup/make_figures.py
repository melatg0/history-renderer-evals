#!/usr/bin/env python
"""Publication figures for the action-history-scheming writeup.

Colorblind-safe (Okabe-Ito), light-mode, direct-labeled (identity never by color
alone), Wilson 95% CIs on every rate. Regenerates from committed results with no
live API calls. Outputs the figures used by the paper and README as 300dpi PNGs.
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "writeup" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito (canonical CVD-safe). Semantics: BLUE=raises (↑, concerning),
# GREEN=lowers (↓, safe), GRAY=flat/floor, AMBER=baseline reference.
BLUE, VERM, GREEN, AMBER, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#7F7F7F"
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
plt.rcParams.update({
    "figure.dpi": 300, "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
    "svg.hashsalt": "action-history-scheming",
})

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0, 0.0
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    m = (z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return p, max(0, c-m), min(1, c+m)

def err(rates):  # asymmetric yerr from (p,lo,hi)
    lo = [r[0]-r[1] for r in rates]; hi = [r[2]-r[0] for r in rates]
    return [lo, hi]

def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)

# ---------------------------------------------------------------- FIG 1
# Headline: the A3-own effect on Opus 4.1: certified, replicated, +18 points.
# Pool the certified-header A3 vs A_raw across the anchor config runs at the
# reconciled config (latent+restriction). Each run plus pooled.
def fig1_headline():
    # (label, A_raw k/n, A3 k/n): certified header, latent+restriction only
    runs = [
        ("Pilot\n(n=40)", (14,40), (25,40)),
        ("Positive control\n(n=120)", (53,120), (75,120)),
        ("Diagnostic\n(n=100)", (42,100), (60,100)),
    ]
    # pooled
    ar_k=sum(r[1][0] for r in runs); ar_n=sum(r[1][1] for r in runs)
    a3_k=sum(r[2][0] for r in runs); a3_n=sum(r[2][1] for r in runs)
    runs.append(("Pooled\n(n=260)", (ar_k,ar_n), (a3_k,a3_n)))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = range(len(runs)); w = 0.38
    raw = [wilson(*r[1]) for r in runs]; a3 = [wilson(*r[2]) for r in runs]
    b1 = ax.bar([i-w/2 for i in x], [r[0] for r in raw], w, yerr=err(raw),
                color=AMBER, capsize=3, ecolor=MUTED, label="None")
    b2 = ax.bar([i+w/2 for i in x], [r[0] for r in a3], w, yerr=err(a3),
                color=BLUE, capsize=3, ecolor=MUTED, label="+Reasoning")
    import decimal
    def d2(p):  # 2dp half-up, so 0.625 -> 0.63 (consistent with the §5 panel)
        return f"{decimal.Decimal(p).quantize(decimal.Decimal('0.01'), rounding=decimal.ROUND_HALF_UP)}"
    for i,r in enumerate(raw): ax.text(i-w/2, r[0]+0.02, d2(r[0]), ha="center", fontsize=8, color=MUTED)
    for i,r in enumerate(a3):  ax.text(i+w/2, r[0]+0.02, d2(r[0]), ha="center", fontsize=8, color=BLUE, weight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in runs])
    ax.set_ylim(0,1); ax.set_ylabel("Blackmail rate")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.axvline(2.5, color=GRID, lw=1, ls=":")
    save(fig, "fig1_headline_opus_effect")

# ---------------------------------------------------------------- FIG 2
# The cross-model reversal + floors: A3 delta per model.
def fig2_reversal():
    # (model, A_raw k/n, A3 k/n, tag): Phase-A pilot n=40, reconciled config
    data = [
        ("Opus 4.1",        (14,40),(25,40), "raises"),
        ("Qwen3-30B",       (23,40),(8,40),  "lowers"),
        ("Gemini 2.5 Pro*", (31,40),(12,40), "lowers"),
        ("GPT-4.1",         (0,40),(0,40),   "floor"),
        ("GPT-4o",          (0,40),(0,40),   "floor"),
        ("Llama-3.3-70B",   (0,40),(0,40),   "floor"),
        ("Sonnet 4.6",      (0,40),(0,40),   "floor"),
        ("Opus 4.8",        (0,40),(0,40),   "floor"),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ys = range(len(data))
    # Delta = difference of the DISPLAYED 2-dp rates (half-up), so figure & prose
    # match the report table (difference of the displayed rates).
    import decimal
    def _r2(k,n):
        return decimal.Decimal(k*100)/decimal.Decimal(n)  # in pp, exact
    def _disp(k,n):  # 2-dp rate, half-up, as the label prints it
        return float((_r2(k,n)/100).quantize(decimal.Decimal('0.01'), rounding=decimal.ROUND_HALF_UP))
    def dpp_disp(k1,n1,k0,n0):
        return round((_disp(k1,n1) - _disp(k0,n0))*100)
    for i,(m,raw,a3,tag) in enumerate(data):
        pr,_,_ = wilson(*raw); pa,la,ha = wilson(*a3)
        d = pa - pr
        col = BLUE if tag=="raises" else (GREEN if tag=="lowers" else GRAY)
        ax.plot([pr,pa],[i,i], color=col, lw=2, zorder=2, solid_capstyle="round")
        ax.scatter([pr],[i], color=AMBER, s=45, zorder=3, edgecolor="white", lw=1)
        ax.scatter([pa],[i], color=col, s=70, zorder=4, edgecolor="white", lw=1, marker=">" if d>0 else ("<" if d<0 else "o"))
        dpp = dpp_disp(a3[0],a3[1],raw[0],raw[1])
        lbl = (f"{dpp:+d} pts" if tag!="floor" else "floor (0%)")
        ax.text(max(pr,pa)+0.03 if tag!="floor" else 0.03, i, lbl, va="center", fontsize=8.5,
                color=col if tag!="floor" else MUTED, weight="bold")
    ax.set_yticks(list(ys)); ax.set_yticklabels([d[0] for d in data]); ax.invert_yaxis()
    ax.set_xlim(0,1); ax.set_xlabel("Blackmail rate  (●  None  →  ▶/◀  +Reasoning)")
    leg = [Patch(color=BLUE,label="raises harmful action"),
           Patch(color=GREEN,label="lowers harmful action"),
           Patch(color=GRAY,label="near zero (0/40)")]
    ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=8.5)
    fig.text(0.5, -0.02,
             "*Gemini runs minimal-thinking (reasoning control); the reversal is corroborated on non-reasoning Qwen.",
             ha="center", fontsize=7.5, color=MUTED)
    save(fig, "fig2_cross_model_reversal")

# ---------------------------------------------------------------- FIG 3
# Mechanism (Qwen, clean): own vs other vs A_raw. Attribution doesn't matter.
def fig3_mechanism():
    cells = [("None",(121,250),GRAY), ("+Reasoning\n(own)",(61,250),GREEN),
             ("+Reasoning\n(other AI)",(60,250),GREEN)]
    fig, ax = plt.subplots(figsize=(6.0,4.2))
    x=range(len(cells)); rates=[wilson(*c[1]) for c in cells]
    ax.bar(x,[r[0] for r in rates],0.6,yerr=err(rates),
           color=[c[2] for c in cells],capsize=4,ecolor=MUTED)
    for i,r in enumerate(rates): ax.text(i,r[0]+0.025,f"{r[0]:.2f}",ha="center",fontsize=10,weight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylim(0,0.7); ax.set_ylabel("Blackmail rate")
    ax.annotate("", xy=(1,0.30), xytext=(0,0.30), arrowprops=dict(arrowstyle="<->",color=MUTED))
    ax.text(0.5,0.32,"−24 pts",ha="center",fontsize=9,color=GREEN,weight="bold")
    save(fig, "fig3_qwen_mechanism")

# ---------------------------------------------------------------- FIG 4
# Baseline-independence (Opus): A3 delta stable as cf shifts baseline (within-config).
def fig4_baseline_independence():
    # cf off / cf on, within latent+restriction, n=180 each
    pts = [("Baseline",(78,180),(114,180)), ("With counterfactual\nprompt",(54,180),(86,180))]
    fig, ax = plt.subplots(figsize=(6.4,4.2))
    x=range(len(pts)); w=0.38
    raw=[wilson(*p[1]) for p in pts]; a3=[wilson(*p[2]) for p in pts]
    ax.bar([i-w/2 for i in x],[r[0] for r in raw],w,yerr=err(raw),color=AMBER,capsize=3,ecolor=MUTED,label="None")
    ax.bar([i+w/2 for i in x],[r[0] for r in a3],w,yerr=err(a3),color=BLUE,capsize=3,ecolor=MUTED,label="+Reasoning")
    for i,p in enumerate(pts):
        a3w=wilson(*p[2]); d=a3w[0]-wilson(*p[1])[0]
        ax.text(i,a3w[2]+0.03,f"Δ = {d:+.2f}",ha="center",fontsize=9,weight="bold",color=BLUE)
    ax.set_xticks(list(x)); ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylim(0,0.8); ax.set_ylabel("Blackmail rate")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "fig4_baseline_independence")

# ---------------------------------------------------------------- FIG 5
# Richness ladder (Opus 4.1, pilot n=180/cell): A_raw -> A1 -> A2 -> A3.
# Shows the "reasoning traces are the potent part" claim: A1 (actions) and A2
# (actions+observations) do not clear A_raw after correction; only A3 (reasoning)
# separates. Data: results/results_latent_n180.csv (hackathon pilot).
def fig5_ladder():
    # k/n per level (n=180), from the pilot latent+restriction run.
    levels = [
        ("None", (78, 180), GRAY),
        ("Actions", (97, 180), GRAY),
        ("Actions\n+ Obs", (88, 180), GRAY),
        ("+Reasoning", (114, 180), BLUE),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = range(len(levels))
    rates = [wilson(*l[1]) for l in levels]
    ax.bar(x, [r[0] for r in rates], 0.62, yerr=err(rates),
           color=[l[2] for l in levels], capsize=4, ecolor=MUTED)
    for i, r in enumerate(rates):
        ax.text(i, r[2] + 0.02, f"{r[0]:.2f}", ha="center", fontsize=10,
                weight="bold" if levels[i][2] == BLUE else "normal",
                color=BLUE if levels[i][2] == BLUE else MUTED)
    # dashed reference line at the None baseline (no label text / marker)
    ax.axhline(rates[0][0], color=GRID, lw=1, ls="--", zorder=1)
    ax.set_xticks(list(x)); ax.set_xticklabels([l[0] for l in levels])
    ax.set_ylim(0, 0.82); ax.set_ylabel("Blackmail rate")
    save(fig, "fig_richness_ladder")

# ---------------------------------------------------------------- FIG 6
# Cross-scenario forest plot: A3-vs-A_raw effect (risk difference, pp) per
# model x scenario, grouped by model, with 95% CIs and exact x/n. Replaces the
# Section 5 table. Odds ratios reported in the appendix/text; risk differences
# shown here to match the "percentage points" narrative.
def fig6_forest():
    # (model, scenario, A_raw k/n, A3 k/n)
    rows = [
        ("Opus 4.1", "blackmail", (53, 120), (75, 120)),
        ("Opus 4.1", "leaking",   (19, 120), (22, 120)),
        ("Opus 4.1", "murder",    (4, 120),  (11, 120)),
        ("Qwen3-30B", "blackmail", (121, 250), (61, 250)),
        ("Qwen3-30B", "leaking",   (115, 200), (82, 200)),
        ("Qwen3-30B", "murder",    (119, 200), (82, 200)),
    ]

    def rd_ci(k1, n1, k0, n0):  # risk difference + 95% CI (pp)
        p1, p0 = k1 / n1, k0 / n0
        se = math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        d = p1 - p0
        return d * 100, (d - 1.96 * se) * 100, (d + 1.96 * se) * 100

    # layout: group by model with a gap between groups; top-to-bottom
    ys, labels, pts, colors, markers, annots = [], [], [], [], [], []
    y = 0
    last_model = None
    for model, scenario, araw, a3 in rows:
        if last_model is not None and model != last_model:
            y -= 1  # gap between model groups
        d, lo, hi = rd_ci(*a3, *araw)
        ys.append(y); labels.append(f"{model}  ·  {scenario}")
        pts.append((d, lo, hi))
        # Redundant encoding (color AND shape) so the figure reads in grayscale:
        # upward triangle = raises, downward triangle = lowers.
        colors.append(BLUE if d > 0 else GREEN)
        markers.append("^" if d > 0 else "v")
        annots.append(f"{a3[0]}/{a3[1]} vs {araw[0]}/{araw[1]}")
        last_model = model
        y -= 1

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axvline(0, color=INK, lw=1, zorder=1)
    for yi, (d, lo, hi), c, mk in zip(ys, pts, colors, markers):
        ax.plot([lo, hi], [yi, yi], color=c, lw=2, zorder=2)
        ax.plot([d], [yi], mk, color=c, ms=9, zorder=3)
    for yi, (d, lo, hi), a in zip(ys, pts, annots):
        ax.text(46, yi, a, va="center", ha="left", fontsize=7.5, color=MUTED)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-40, 62); ax.set_xlabel("+Reasoning − None  (percentage points, 95% CI)")
    leg = [Line2D([0], [0], marker="^", color=BLUE, lw=0, ms=9,
                  label="raises harmful action (Opus)"),
           Line2D([0], [0], marker="v", color=GREEN, lw=0, ms=9,
                  label="lowers harmful action (Qwen)")]
    ax.legend(handles=leg, frameon=False, loc="center left",
              bbox_to_anchor=(0.02, 0.5), fontsize=8.5)
    save(fig, "fig_cross_scenario_forest")

# ---------------------------------------------------------------- FIG: artifact variance (NEW HEADLINE)
# The paper's thesis figure: 8 independently authored traces per model vs baseline.
# Qwen every trace suppresses (tight); Opus traces straddle baseline, one outlier (wide).
def fig_artifact_variance():
    # blackmail A3 rate per trace, n=120 (blackmail). From step7 paraphrase runs + baselines.
    qwen = {  # /120 unless noted
        "01": (61,250), "02": (37,120), "03": (21,120), "04": (28,120),
        "05": (17,120), "06": (29,120), "07": (20,120), "08": (31,120)}
    opus = {  # trace 01 shown as OpenRouter route control (route-matched to paraphrases)
        "01": (83,120), "02": (50,121), "03": (57,121), "04": (68,121),
        "05": (45,120), "06": (52,120), "07": (56,120), "08": (48,113)}
    qwen_base = (121,250)   # 48.4%
    opus_base = (53,120)    # 44.2% (native); route-matched baseline 46% (n=150) shown as band note
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    def scatter(model_x, cells, base, color, marker):
        bp = base[0]/base[1]*100
        ax.axhline(bp, xmin=(model_x-0.35)/2 if False else 0, color=color, lw=1, ls=":", alpha=0.0)
        pts=[]
        for i,(s,(k,n)) in enumerate(sorted(cells.items())):
            p,lo,hi=wilson(k,n); p*=100
            x = model_x + (i-3.5)*0.055
            ax.errorbar([x],[p], yerr=[[p-lo*100],[hi*100-p]], fmt=marker, color=color,
                        ms=6, lw=1, capsize=2, ecolor=color, alpha=0.85, zorder=3)
            pts.append(p)
        return bp, pts
    qb,qp = scatter(1, qwen, qwen_base, GREEN, "o")
    ob,op = scatter(2, opus, opus_base, BLUE, "s")
    # baseline bars (short horizontal) per model
    for x,bp,c in [(1,qb,GREEN),(2,ob,BLUE)]:
        ax.plot([x-0.28,x+0.28],[bp,bp], color=AMBER, lw=2.2, zorder=2)
        ax.text(x+0.30, bp, "None\nbaseline", va="center", ha="left", fontsize=7.5, color=MUTED)
    # annotate spread
    import statistics as st
    ax.text(1, 6, f"8 traces\nSD {st.pstdev(qp):.0f} pts\nall suppress", ha="center", fontsize=8.5, color=GREEN)
    ax.text(2, 82, f"8 traces\nSD {st.pstdev(op):.0f} pts\nstraddle baseline", ha="center", fontsize=8.5, color=BLUE)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Qwen3-30B","Opus 4.1"])
    ax.set_xlim(0.5,2.6); ax.set_ylim(0,90)
    ax.set_ylabel("Blackmail rate (%), +Reasoning cell")
    ax.set_title("One trace is not a propensity: per-trace rates vs. None baseline", fontsize=10)
    save(fig, "fig_artifact_variance")

# ---------------------------------------------------------------- FIG: replicated action-history suppression
# Lead paper figure: full benign histories across independent instantiations on
# the two model families with usable baseline headroom.
def fig_history_robustness():
    qwen = {
        "01": (61,250), "02": (37,120), "03": (21,120), "04": (28,120),
        "05": (17,120), "06": (29,120), "07": (20,120), "08": (31,120)}
    # Prefer the n=120 confirmatory cells where available; the remaining cells
    # come from the pre-committed n=40 eight-history screen.
    mistral = {
        "01": (47,120), "02": (51,120), "03": (13,40), "04": (53,120),
        "05": (13,40), "06": (18,40), "07": (36,120), "08": (12,40)}
    panels = [
        ("Qwen3-30B", qwen, (121,250), "22.4% pooled\nOR 0.31 [0.23, 0.41]"),
        ("Mistral-Large", mistral, (88,160), "38.0% pooled\nOR 0.50 [0.35, 0.71]"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    for ax, (title, cells, baseline, summary) in zip(axes, panels):
        base = baseline[0] / baseline[1] * 100
        ax.axhline(base, color=AMBER, lw=2.2, zorder=1)
        for i, (script, (k, n)) in enumerate(sorted(cells.items()), start=1):
            p, lo, hi = wilson(k, n)
            p, lo, hi = p * 100, lo * 100, hi * 100
            ax.errorbar(
                i, p, yerr=[[p - lo], [hi - p]], fmt="o", color=GREEN,
                ms=5.5, capsize=2, lw=1, ecolor=GREEN, zorder=3,
            )
        ax.text(8.15, base, "no-history\nbaseline", va="center", ha="left",
                fontsize=7.2, color=MUTED)
        ax.text(4.5, 5, summary, ha="center", va="bottom", fontsize=8.2,
                color=GREEN, weight="bold")
        ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels([f"h{i}" for i in range(1, 9)], fontsize=8)
        ax.set_xlim(0.5, 9.25)
        ax.set_xlabel("independent benign history")
    axes[0].set_ylim(0, 72)
    axes[0].set_ylabel("Harmful-action rate (%)")
    fig.suptitle("Benign prior-session histories suppress across independent instantiations",
                 fontsize=10.2, y=1.01)
    save(fig, "fig_history_robustness")

# ---------------------------------------------------------------- FIG: Opus artifact-level failure case
def fig_opus_history_variance():
    cells = {
        "01": (83,120), "02": (50,121), "03": (57,121), "04": (68,121),
        "05": (45,120), "06": (52,120), "07": (56,120), "08": (48,113)}
    baseline = 53 / 120 * 100

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.axhline(baseline, color=AMBER, lw=2.2, zorder=1)
    for i, (script, (k, n)) in enumerate(sorted(cells.items()), start=1):
        p, lo, hi = wilson(k, n)
        p, lo, hi = p * 100, lo * 100, hi * 100
        color = VERM if script == "01" else BLUE
        marker = "D" if script == "01" else "s"
        ax.errorbar(
            i, p, yerr=[[p - lo], [hi - p]], fmt=marker, color=color,
            ms=6, capsize=2, lw=1, ecolor=color, zorder=3,
        )
    ax.annotate(
        "original history\n62.5% confirmatory;\n69.2% route-matched rerun",
        xy=(1, cells["01"][0] / cells["01"][1] * 100), xytext=(2.25, 82),
        fontsize=7.6, color=VERM, ha="center",
        arrowprops=dict(arrowstyle="->", color=VERM, lw=1),
    )
    ax.text(8.15, baseline, "no-history\nbaseline (44.2%)", va="center",
            ha="left", fontsize=7.2, color=MUTED)
    ax.text(5.1, 24, "histories 02-08 pooled: 45.0%\nOR 1.03 [0.70, 1.52]",
            ha="center", fontsize=8.2, color=BLUE, weight="bold")
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"h{i}" for i in range(1, 9)])
    ax.set_xlim(0.5, 9.25)
    ax.set_ylim(15, 90)
    ax.set_xlabel("independent benign history")
    ax.set_ylabel("Blackmail rate (%)")
    ax.set_title("Opus 4.1: a repeated single-history effect does not generalize",
                 fontsize=10)
    save(fig, "fig_opus_history_variance")

# ---------------------------------------------------------------- FIG: Qwen ladder (reasoning not required)
def fig_qwen_ladder():
    # pooled s03+s05, n per pooled cell. baseline None=48.4% (121/250)
    cells = [("None",(121,250),AMBER), ("Actions",(56,242),GREEN),
             ("Actions\n+Obs",(56,242),GREEN), ("+Reasoning",(38,240),GREEN)]
    fig, ax = plt.subplots(figsize=(6.2,4.0))
    x=range(len(cells)); rates=[wilson(*c[1]) for c in cells]
    ax.bar(x,[r[0]*100 for r in rates],0.62,
           yerr=[[ (r[0]-r[1])*100 for r in rates],[(r[2]-r[0])*100 for r in rates]],
           color=[c[2] for c in cells],capsize=4,ecolor=MUTED)
    for i,r in enumerate(rates): ax.text(i,r[0]*100+2,f"{r[0]*100:.0f}%",ha="center",fontsize=10,weight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in cells])
    ax.set_ylim(0,60); ax.set_ylabel("Blackmail rate (%)")
    ax.set_title("Qwen3-30B: a bare action log carries the suppression", fontsize=10)
    ax.annotate("", xy=(1,30), xytext=(0,30), arrowprops=dict(arrowstyle="->",color=INK))
    ax.text(0.5,32,"action log alone\n-25 pts",ha="center",fontsize=8.5,color=GREEN,weight="bold")
    ax.annotate("", xy=(3,20), xytext=(2,20), arrowprops=dict(arrowstyle="->",color=MUTED))
    ax.text(2.5,22,"reasoning:\n-7 pts (marginal)",ha="center",fontsize=8,color=MUTED)
    save(fig, "fig_qwen_ladder")

# ---------------------------------------------------------------- FIG: cross-scenario grid (trace 01)
# Confirmatory grid visualized: risk difference (+Reasoning - None) by model x scenario.
def fig_cross_scenario():
    # (model, scenario, None k/n, +Reas k/n)
    rows=[("Opus 4.1","blackmail",(53,120),(75,120)),
          ("Opus 4.1","leaking",(19,120),(22,120)),
          ("Opus 4.1","murder",(4,120),(11,120)),
          ("Qwen3-30B","blackmail",(121,250),(61,250)),
          ("Qwen3-30B","leaking",(115,200),(82,200)),
          ("Qwen3-30B","murder",(119,200),(82,200))]
    def rd_ci(k1,n1,k0,n0):
        p1,p0=k1/n1,k0/n0; se=math.sqrt(p1*(1-p1)/n1+p0*(1-p0)/n0)
        return (p1-p0)*100,(p1-p0-1.96*se)*100,(p1-p0+1.96*se)*100
    fig,ax=plt.subplots(figsize=(7.0,3.8))
    ys=list(range(len(rows)))[::-1]
    for y,(m,sc,none,reas) in zip(ys,rows):
        d,lo,hi=rd_ci(*reas,*none)
        up=d>0; col=BLUE if up else GREEN; mk="^" if up else "v"
        ax.plot([lo,hi],[y,y],color=col,lw=2,zorder=2)
        ax.plot([d],[y],mk,color=col,ms=9,zorder=3)
        ax.text(hi+1.5 if up else lo-1.5, y, f"{m} · {sc}", va="center",
                ha="left" if up else "right", fontsize=8.5, color=INK)
    ax.axvline(0,color=INK,lw=1)
    ax.set_yticks([]); ax.set_xlim(-42,47); ax.set_ylim(-0.8, len(rows)-0.2)
    ax.set_xlabel("+Reasoning − None risk difference (pts, 95% CI), trace 01")
    ax.set_title("Trace-01 effect by model and scenario", fontsize=10)
    leg=[Line2D([0],[0],marker="^",color=BLUE,lw=0,ms=9,label="Opus (raises)"),
         Line2D([0],[0],marker="v",color=GREEN,lw=0,ms=9,label="Qwen (suppresses)")]
    ax.legend(handles=leg,frameon=False,loc="upper left",fontsize=8.5,
              bbox_to_anchor=(0.02,0.55))
    save(fig,"fig_cross_scenario")

# ---------------------------------------------------------------- FIG: frontier floor panel
# 8-model screen: where the effect is measurable. None vs +Reasoning per model.
def fig_frontier_floor():
    # (model, None, +Reas, kind)
    data=[("Opus 4.1",0.35,0.63,"move"),("Qwen3-30B",0.58,0.20,"move"),
          ("Gemini 2.5 Pro*",0.78,0.30,"move"),("GPT-4.1",0.0,0.0,"floor"),
          ("GPT-4o",0.0,0.0,"floor"),("Llama-3.3-70B",0.0,0.0,"floor"),
          ("Sonnet 4.6",0.0,0.0,"floor"),("Opus 4.8",0.0,0.0,"floor")]
    fig,ax=plt.subplots(figsize=(7.2,4.0))
    ys=list(range(len(data)))[::-1]
    for y,(m,none,reas,kind) in zip(ys,data):
        if kind=="floor":
            ax.plot([0],[y],"o",color=GRAY,ms=7)
            ax.text(0.03,y,f"{m}: floor (0/40)",va="center",fontsize=8.5,color=MUTED)
        else:
            col=BLUE if reas>none else GREEN
            ax.annotate("",xy=(reas,y),xytext=(none,y),
                        arrowprops=dict(arrowstyle="->",color=col,lw=2))
            ax.plot([none],[y],"o",color=AMBER,ms=7,zorder=3)
            ax.text(max(none,reas)+0.02,y,m,va="center",fontsize=8.5,color=INK)
    ax.set_yticks([]); ax.set_xlim(0,1.05); ax.set_ylim(-0.6,len(data)-0.4)
    ax.set_xlabel("Blackmail rate (screen, $n$=40): None $\\rightarrow$ +Reasoning")
    ax.set_title("Where the effect is measurable: 5 of 8 models floor", fontsize=10)
    ax.annotate("*Gemini thinking cannot be disabled; corroborating-only.",
                xy=(0,0), xytext=(0.98,-0.45), fontsize=7.5, color=MUTED, ha="right")
    save(fig,"fig_frontier_floor")

# ---------------------------------------------------------------- FIG: Mistral breadth (NEW)
# Second family corroborates both halves: a single-trace screen "amplification" (+12pp)
# collapses under the full paraphrase set, which instead SUPPRESSES. Per-trace A3 rates
# (confirmatory n=120 for s01/02/04/07; screen n=40 for s03/05/06/08) vs pooled baseline.
# Data: results/step12_mistral_paraphrase + step12b_mistral_confirmatory.
def fig_breadth_mistral():
    # per-trace A3 (k, n): prefer confirmatory n=120 where available
    traces = {
        "01": (47,120), "02": (51,120), "03": (13,40), "04": (53,120),
        "05": (13,40),  "06": (18,40),  "07": (36,120), "08": (12,40)}
    base = (88,160)          # pooled A_raw 55.0%
    screen_s01 = (21,40)     # the step8/screen single-trace "amplify" read (52-68%)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bp = base[0]/base[1]*100
    ax.plot([-0.5, 7.5], [bp, bp], color=AMBER, lw=2.2, zorder=2)
    ax.text(7.55, bp, "None\nbaseline\n(55%)", va="center", ha="left", fontsize=7.5, color=MUTED)
    ps=[]
    for i,(s,(k,n)) in enumerate(sorted(traces.items())):
        p,lo,hi=wilson(k,n); p*=100
        ax.errorbar([i],[p], yerr=[[p-lo*100],[hi*100-p]], fmt="s", color=GREEN,
                    ms=6, lw=1, capsize=2, ecolor=GREEN, alpha=0.85, zorder=3)
        ps.append(p)
    import statistics as st
    ax.text(3.5, 12, f"8 traces · pooled 38%\nall below baseline\nOR 0.50 [0.35, 0.71]",
            ha="center", fontsize=8.5, color=GREEN, weight="bold")
    # mark the single-trace screen artifact
    ax.annotate("single-trace screen\nsuggested +12pp (amplify);\ndid not reproduce",
                xy=(0, 39.2), xytext=(1.4, 72), fontsize=7.5, color=VERM,
                ha="center", arrowprops=dict(arrowstyle="->", color=VERM, lw=1))
    ax.set_xticks(range(8)); ax.set_xticklabels([f"s0{i}" for i in range(1,9)])
    ax.set_xlim(-0.6, 8.4); ax.set_ylim(0, 80)
    ax.set_ylabel("Blackmail rate (%), +Reasoning cell")
    ax.set_title("Mistral-Large: a second family suppresses; the screen 'amplification' was an artifact",
                 fontsize=9.5)
    save(fig, "fig_breadth_mistral")

if __name__ == "__main__":
    fig_artifact_variance()
    fig_history_robustness()
    fig_opus_history_variance()
    fig_qwen_ladder()
    fig_frontier_floor()
    print("all figures ->", FIG)
