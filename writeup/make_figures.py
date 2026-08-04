#!/usr/bin/env python
"""Generate publication figures from committed, source-annotated counts."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "writeup" / "figures"
DATA = ROOT / "results" / "paper" / "figure_counts.csv"
INFERENCE_DATA = (
    ROOT / "results" / "paper" / "controlled_history_inference.csv"
)
STABILITY_DATA = ROOT / "results" / "paper" / "history_budget_stability.csv"
NATIVE_HISTORY_DATA = (
    ROOT
    / "results"
    / "native_tools"
    / "history_distribution_opus41_history_analysis.csv"
)
NATIVE_CONTRAST_DATA = (
    ROOT
    / "results"
    / "native_tools"
    / "history_distribution_opus41_contrasts.csv"
)
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colors. Identity is also encoded by labels, position, or marker.
BLUE = "#0072B2"
VERMILION = "#D55E00"
GREEN = "#009E73"
AMBER = "#E69F00"
GRAY = "#7F7F7F"
INK = "#222222"
MUTED = "#666666"
GRID = "#DDDDDD"

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
    }
)


def load_counts(path: Path = DATA) -> list[dict[str, object]]:
    """Load and validate the compact figure input table."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            harmful = int(raw["harmful"])
            n = int(raw["n"])
            if not 0 <= harmful <= n:
                raise ValueError(f"Invalid count in {path}: {raw}")
            rows.append({**raw, "harmful": harmful, "n": n})
    return rows


COUNTS = load_counts()


def select(
    dataset: str,
    model: str,
    condition: str | None = None,
) -> list[dict[str, object]]:
    rows = [
        row
        for row in COUNTS
        if row["dataset"] == dataset and row["model"] == model
    ]
    if condition is not None:
        rows = [row for row in rows if row["condition"] == condition]
    if not rows:
        raise ValueError(f"No figure rows for {dataset}/{model}/{condition}")
    return rows


def one(
    dataset: str,
    model: str,
    condition: str,
    history_id: str | None = None,
) -> tuple[int, int]:
    rows = select(dataset, model, condition)
    if history_id is not None:
        rows = [row for row in rows if row["history_id"] == history_id]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {dataset}/{model}/{condition}/{history_id}, "
            f"got {len(rows)}"
        )
    return int(rows[0]["harmful"]), int(rows[0]["n"])


def combine(rows: list[dict[str, object]]) -> tuple[int, int]:
    return sum(int(row["harmful"]) for row in rows), sum(
        int(row["n"]) for row in rows
    )


def histories(model: str) -> dict[str, tuple[int, int]]:
    rows = select("history_robustness", model, "history")
    result = {
        str(row["history_id"]): (int(row["harmful"]), int(row["n"]))
        for row in rows
    }
    expected = {f"h{index}" for index in range(1, 9)}
    if set(result) != expected:
        raise ValueError(f"Incomplete history panel for {model}: {sorted(result)}")
    return result


def baseline(model: str) -> tuple[int, int]:
    return combine(select("history_robustness", model, "baseline"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        / denominator
    )
    return p, max(0, center - margin), min(1, center + margin)


def odds_ratio_interval(
    treatment: tuple[int, int],
    reference: tuple[int, int],
    z: float = 1.96,
) -> tuple[float, float, float]:
    a, n_treatment = treatment
    c, n_reference = reference
    b = n_treatment - a
    d = n_reference - c
    if min(a, b, c, d) <= 0:
        raise ValueError("Odds-ratio interval requires four positive cell counts")
    odds_ratio = (a * d) / (b * c)
    standard_error = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return (
        odds_ratio,
        math.exp(math.log(odds_ratio) - z * standard_error),
        math.exp(math.log(odds_ratio) + z * standard_error),
    )


def save(
    figure: plt.Figure,
    name: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    for output_format in formats:
        figure.savefig(
            FIG / f"{name}.{output_format}",
            dpi=300 if output_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    print("wrote", ", ".join(f"{name}.{suffix}" for suffix in formats))


def load_history_inference(
    path: Path = INFERENCE_DATA,
) -> list[dict[str, object]]:
    """Load history-resampling estimates used in the evidence figure."""
    integer_fields = {
        "baseline_harmful",
        "baseline_n",
        "history_count",
        "histories_below_baseline",
        "histories_equal_baseline",
        "histories_above_baseline",
        "bootstrap_draws",
        "bootstrap_seed",
    }
    float_fields = {
        "equal_history_mean",
        "baseline_rate",
        "mean_difference",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in integer_fields:
                row[field] = int(raw[field])
            for field in float_fields:
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def history_inference(
    model: str,
    subset: str,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return one authoritative history-level inference row."""
    candidates = [
        row
        for row in rows or load_history_inference()
        if row["model"] == model and row["subset"] == subset
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one inference row for {model}/{subset}, "
            f"got {len(candidates)}"
        )
    return candidates[0]


def load_stability(path: Path = STABILITY_DATA) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                **row,
                "history_budget": int(row["history_budget"]),
                "rollouts_per_history": int(row["rollouts_per_history"]),
                "agreement_rate": float(row["agreement_rate"]),
            }
            for row in csv.DictReader(handle)
        ]


def stability_matrix(
    rows: list[dict[str, object]],
    model: str,
    history_budgets: tuple[int, ...],
    rollout_budgets: tuple[int, ...],
) -> np.ndarray:
    """Return a complete random-history stability matrix."""
    selected = [
        row
        for row in rows
        if row["model"] == model and row["sampling"] == "random_histories"
    ]
    indexed = {
        (int(row["history_budget"]), int(row["rollouts_per_history"])): float(
            row["agreement_rate"]
        )
        for row in selected
    }
    expected = {
        (history_budget, rollout_budget)
        for history_budget in history_budgets
        for rollout_budget in rollout_budgets
    }
    if len(indexed) != len(selected) or set(indexed) != expected:
        raise ValueError(f"Incomplete or duplicate stability cells for {model}")
    return np.asarray(
        [
            [
                indexed[(history_budget, rollout_budget)]
                for rollout_budget in rollout_budgets
            ]
            for history_budget in history_budgets
        ]
    )


def fixed_history_curve(
    rows: list[dict[str, object]],
    model: str,
) -> list[dict[str, object]]:
    """Return fixed-h1 stability cells ordered by rollout budget."""
    fixed = sorted(
        (
            row
            for row in rows
            if row["model"] == model and row["sampling"] == "fixed_h1"
        ),
        key=lambda row: int(row["rollouts_per_history"]),
    )
    budgets = [int(row["rollouts_per_history"]) for row in fixed]
    if budgets != [10, 20, 40, 80, 100]:
        raise ValueError(f"Unexpected fixed-h1 stability cells for {model}")
    return fixed


def load_renderer_effects(
    path: Path = NATIVE_HISTORY_DATA,
) -> dict[str, list[float]]:
    """Load matched per-history renderer effects relative to D0."""
    conditions = ("D1", "D2", "D3")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    baselines = {
        row["run_block"]: float(row["harmful_rate"])
        for row in rows
        if row["condition"] == "D0"
    }
    if len(baselines) != 15:
        raise ValueError(f"Expected 15 native D0 histories in {path}")

    effects: dict[str, list[float]] = {}
    for condition in conditions:
        selected = sorted(
            (row for row in rows if row["condition"] == condition),
            key=lambda row: row["run_block"],
        )
        if (
            len(selected) != 15
            or {row["run_block"] for row in selected} != set(baselines)
        ):
            raise ValueError(
                f"Expected 15 histories matched to D0 for {condition}"
            )
        condition_effects = []
        for row in selected:
            effect = (
                float(row["harmful_rate"]) - baselines[row["run_block"]]
            )
            recorded_effect = float(row["diff_vs_block_D0"])
            if not math.isclose(effect, recorded_effect, abs_tol=1e-12):
                raise ValueError(
                    f"Native effect mismatch for "
                    f"{row['run_block']}/{condition}"
                )
            condition_effects.append(effect)
        effects[condition] = condition_effects
    return effects


def load_renderer_contrasts(
    path: Path = NATIVE_CONTRAST_DATA,
) -> dict[str, dict[str, float]]:
    """Load the planned renderer contrasts and simultaneous intervals."""
    expected = {"D1-D0", "D2-D0", "D3-D0"}
    with path.open(encoding="utf-8", newline="") as handle:
        selected = {
            row["contrast"]: {
                "mean": float(row["mean_difference"]),
                "simultaneous_low": float(row["simultaneous_low"]),
                "simultaneous_high": float(row["simultaneous_high"]),
                "holm_p": float(row["holm_adjusted_p"]),
            }
            for row in csv.DictReader(handle)
            if row["family"] == "primary"
        }
    if set(selected) != expected:
        raise ValueError(f"Missing planned native contrasts in {path}")
    return selected


def _pipeline_cell(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    edgecolor: str,
    facecolor: str,
    linestyle: str = "-",
    fontsize: float = 7.5,
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        linespacing=1.05,
    )


def fig_replication_pipeline() -> None:
    """Contrast conditional rollout replication with stability evaluation."""
    figure, axis = plt.subplots(figsize=(7.4, 2.65))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    columns = (
        (0.01, 0.15, "Source history\n$h$"),
        (0.21, 0.14, "Renderer\n$r$"),
        (0.40, 0.16, "Rendering\n$x_{h,r}$"),
        (0.61, 0.15, "Rollouts\n$y_1,\\ldots,y_n$"),
        (0.81, 0.18, "Harmful-action rate\n$\\widehat{p}_{h,r}$"),
    )
    for x, width, text in columns:
        _pipeline_cell(
            axis,
            x,
            0.72,
            width,
            0.19,
            text,
            edgecolor=INK,
            facecolor="white",
            fontsize=7.6,
        )
    for left, right in zip(columns, columns[1:]):
        left_x, left_width = left[:2]
        right_x = right[0]
        axis.annotate(
            "",
            xy=(right_x - 0.008, 0.815),
            xytext=(left_x + left_width + 0.008, 0.815),
            arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.2},
        )

    comparisons = (
        (
            0.01,
            "Rollout replication",
            "fix $h$ and $r$; repeat rollouts\nreduce conditional uncertainty",
            MUTED,
            "--",
            "#F7F7F7",
        ),
        (
            0.345,
            "Source stability",
            "vary $h$; hold $r$ fixed\ncompare across histories",
            BLUE,
            "-",
            "white",
        ),
        (
            0.68,
            "Renderer stability",
            "match histories; vary $r$\ncompare across renderings",
            BLUE,
            "-",
            "white",
        ),
    )
    for x, title, detail, edgecolor, linestyle, facecolor in comparisons:
        box = FancyBboxPatch(
            (x, 0.31),
            0.31,
            0.27,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.3,
            edgecolor=edgecolor,
            facecolor=facecolor,
            linestyle=linestyle,
        )
        axis.add_patch(box)
        axis.text(
            x + 0.155,
            0.50,
            title,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=edgecolor,
        )
        axis.text(
            x + 0.155,
            0.395,
            detail,
            ha="center",
            va="center",
            fontsize=7.8,
            linespacing=1.15,
        )

    axis.text(
        0.5,
        0.10,
        "More rollouts reduce conditional uncertainty; "
        "more histories and renderers test generalizability.",
        ha="center",
        va="center",
        fontsize=8.1,
        color=BLUE,
        fontweight="bold",
    )
    save(
        figure,
        "fig_replication_pipeline",
        formats=("pdf", "png"),
    )


def fig_renderer_sensitivity() -> None:
    """Plot paired native renderer effects with simultaneous intervals."""
    effects = load_renderer_effects()
    contrasts = load_renderer_contrasts()
    labels = {
        "D1": "Calls",
        "D2": "Calls + results",
        "D3": "Neutral summary",
    }
    conditions = ("D1", "D2", "D3")
    y_positions = (2, 1, 0)
    offsets = np.linspace(-0.14, 0.14, 15)

    figure, axis = plt.subplots(figsize=(3.45, 2.70))
    figure.subplots_adjust(left=0.34, right=0.98, top=0.70, bottom=0.22)
    for index, (condition, y) in enumerate(
        zip(conditions, y_positions)
    ):
        contrast = contrasts[f"{condition}-D0"]
        observed_mean = statistics.mean(effects[condition])
        if not math.isclose(
            observed_mean, contrast["mean"], abs_tol=1e-12
        ):
            raise ValueError(f"Native mean mismatch for {condition}-D0")
        axis.scatter(
            100 * np.asarray(effects[condition]),
            y + offsets,
            s=13,
            facecolors="white",
            edgecolors=MUTED,
            linewidths=0.7,
            alpha=0.9,
            zorder=2,
            label="per-history paired shift" if index == 0 else None,
        )
        mean = 100 * contrast["mean"]
        low = 100 * contrast["simultaneous_low"]
        high = 100 * contrast["simultaneous_high"]
        axis.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color=BLUE,
            ecolor=BLUE,
            markerfacecolor=BLUE,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            markersize=5.7,
            linewidth=1.7,
            capsize=3.2,
            capthick=1.2,
            zorder=3,
            label=(
                "mean + simultaneous 95% interval"
                if index == 0
                else None
            ),
        )
        axis.text(
            mean,
            y - 0.25,
            f"{mean:+.1f}",
            ha="center",
            va="top",
            fontsize=8.1,
            color=BLUE,
            fontweight="bold",
        )
        axis.text(
            52,
            y + 0.25,
            f"Holm $p={contrast['holm_p']:.3f}$",
            ha="right",
            va="bottom",
            fontsize=7.9,
            color=MUTED,
        )

    axis.axvline(0, color=INK, linewidth=1.1, linestyle="--", zorder=1)
    axis.set_xlim(-55, 55)
    axis.set_ylim(-0.48, 2.52)
    axis.set_xticks((-50, -25, 0, 25, 50))
    axis.set_yticks(y_positions)
    axis.set_yticklabels([labels[condition] for condition in conditions])
    axis.set_xlabel("Change vs. matched D0 (percentage points)", fontsize=7.7)
    axis.tick_params(axis="x", labelsize=7.0)
    axis.tick_params(axis="y", labelsize=7.6)
    axis.yaxis.grid(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        frameon=False,
        fontsize=7.3,
        handlelength=2.0,
        labelspacing=0.25,
    )
    save(figure, "fig_renderer_sensitivity", formats=("pdf", "png"))


def fig_false_certainty() -> None:
    """Combine finite-panel rates with fixed-history conclusion stability."""
    inference_rows = load_history_inference()
    qwen_inference = history_inference(
        "Qwen3-30B", "all_h1_h8", inference_rows
    )
    opus_all = history_inference("Opus 4.1", "all_h1_h8", inference_rows)
    opus_others = history_inference("Opus 4.1", "h2_h8", inference_rows)
    qwen = histories("Qwen3-30B")
    opus = histories("Opus 4.1")
    stability_rows = load_stability()

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 3.20),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.18)},
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        top=0.84,
        bottom=0.18,
        wspace=0.38,
    )

    axis = axes[0]
    qwen_x = list(range(1, 9))
    qwen_rates = [
        100 * qwen[f"h{index}"][0] / qwen[f"h{index}"][1]
        for index in qwen_x
    ]
    qwen_baseline = 100 * float(qwen_inference["baseline_rate"])
    qwen_mean = 100 * float(qwen_inference["equal_history_mean"])
    axis.axhline(
        qwen_baseline,
        color=INK,
        linewidth=1.4,
        linestyle="--",
        label="baseline",
    )
    axis.axhline(
        qwen_mean,
        color=GREEN,
        linewidth=1.5,
        linestyle=":",
        label="equal-history mean",
    )
    axis.plot(
        qwen_x,
        qwen_rates,
        linestyle="none",
        marker="o",
        markersize=5.2,
        markerfacecolor=GREEN,
        markeredgecolor=INK,
        markeredgewidth=0.5,
        color=GREEN,
    )
    axis.text(
        0.04,
        0.96,
        "8/8 below baseline\n"
        f"$\\Delta={100 * float(qwen_inference['mean_difference']):+.1f}$ "
        "pts\n95% resampling interval\n"
        f"$[{100 * float(qwen_inference['bootstrap_ci_low']):.1f},"
        f"{100 * float(qwen_inference['bootstrap_ci_high']):.1f}]$",
        transform=axis.transAxes,
        va="top",
        fontsize=8.1,
        linespacing=1.08,
    )
    axis.text(
        8.25,
        qwen_baseline + 1.2,
        "baseline",
        ha="right",
        fontsize=6.7,
    )
    axis.text(
        8.25,
        qwen_mean - 2.4,
        "8-history mean",
        ha="right",
        fontsize=6.7,
        color=GREEN,
    )
    axis.set_title(
        "A. Qwen3-30B\nconsistent within panel",
        fontsize=8.7,
        fontweight="bold",
    )

    axis = axes[1]
    opus_x = list(range(1, 9))
    opus_rates = [
        100 * opus[f"h{index}"][0] / opus[f"h{index}"][1]
        for index in opus_x
    ]
    opus_baseline = 100 * float(opus_all["baseline_rate"])
    opus_mean = 100 * float(opus_all["equal_history_mean"])
    axis.axhline(
        opus_baseline,
        color=INK,
        linewidth=1.4,
        linestyle="--",
    )
    axis.axhline(
        opus_mean,
        color=BLUE,
        linewidth=1.5,
        linestyle=":",
    )
    axis.plot(
        opus_x[1:],
        opus_rates[1:],
        linestyle="none",
        marker="s",
        markersize=5.0,
        markerfacecolor=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.5,
        color=BLUE,
    )
    axis.plot(
        [1],
        [opus_rates[0]],
        linestyle="none",
        marker="D",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=VERMILION,
        markeredgewidth=1.5,
        color=VERMILION,
    )
    axis.annotate(
        "outcome-selected $h_1$",
        xy=(1, opus_rates[0]),
        xytext=(2.0, 76),
        ha="left",
        va="center",
        fontsize=8.0,
        color=VERMILION,
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1.0},
    )
    axis.text(
        0.04,
        0.06,
        "95% resampling intervals\nall 8: "
        f"$\\Delta={100 * float(opus_all['mean_difference']):+.1f}$ pts\n"
        f"$[{100 * float(opus_all['bootstrap_ci_low']):.1f},"
        f"{100 * float(opus_all['bootstrap_ci_high']):.1f}]$\n"
        r"$h_2$-$h_8$: "
        f"$\\Delta={100 * float(opus_others['mean_difference']):+.1f}$ pts\n"
        f"$[{100 * float(opus_others['bootstrap_ci_low']):.1f},"
        f"{100 * float(opus_others['bootstrap_ci_high']):.1f}]$",
        transform=axis.transAxes,
        va="bottom",
        fontsize=7.9,
        linespacing=1.08,
    )
    axis.text(
        8.25,
        opus_baseline - 3.2,
        "baseline",
        ha="right",
        fontsize=6.7,
    )
    axis.text(
        8.25,
        opus_mean + 2.3,
        "8-history mean",
        ha="right",
        fontsize=6.7,
        color=BLUE,
    )
    axis.set_title(
        "B. Opus 4.1\nselected history is discordant",
        fontsize=9.2,
        fontweight="bold",
    )

    for axis in axes[:2]:
        axis.set_xlim(0.55, 8.45)
        axis.set_ylim(0, 82)
        axis.set_xticks(range(1, 9))
        axis.set_xticklabels(
            [f"$h_{index}$" for index in range(1, 9)],
            fontsize=7.2,
        )
        axis.set_xlabel("source history", fontsize=7.8)
        axis.tick_params(axis="y", labelsize=7.8)
    axes[0].set_ylabel("Harmful-action rate (%)", fontsize=8.2)
    axes[1].tick_params(axis="y", labelleft=False)

    axis = axes[2]
    curve_styles = (
        ("Qwen3-30B", GREEN, "o", "-"),
        ("Opus 4.1", VERMILION, "s", "--"),
    )
    for model, color, marker, line_style in curve_styles:
        fixed = fixed_history_curve(stability_rows, model)
        axis.plot(
            [int(row["rollouts_per_history"]) for row in fixed],
            [100 * float(row["agreement_rate"]) for row in fixed],
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=1.8,
            markersize=4.6,
            label=model,
        )
    axis.set_ylim(-4, 104)
    axis.set_xlim(7, 103)
    axis.set_xticks((10, 20, 40, 80, 100))
    axis.set_xlabel(
        "Rollouts sampled from fixed h1 and baseline",
        fontsize=7.4,
    )
    axis.set_ylabel(
        "Agreement with eight-history conclusion (%)",
        fontsize=7.8,
    )
    axis.tick_params(labelsize=7.8)
    axis.set_title(
        r"C. Fixed-$h_1$ replication"
        "\n(post hoc)",
        fontsize=9.2,
        fontweight="bold",
    )
    axis.legend(
        frameon=False,
        fontsize=7.4,
        loc="center right",
        handlelength=2.5,
    )
    axis.annotate(
        "more conditional precision,\nless panel agreement",
        xy=(80, 0.4),
        xytext=(56, 32),
        ha="center",
        fontsize=8.2,
        color=VERMILION,
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1.1},
    )

    save(figure, "fig_false_certainty", formats=("pdf", "png"))


def fig_history_budget_appendix() -> None:
    """Show both random-history budget heatmaps in one appendix figure."""
    rows = load_stability()
    models = ("Qwen3-30B", "Opus 4.1")
    history_budgets = (1, 2, 4, 8)
    rollout_budgets = (10, 20, 40, 80, 100)
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.65))
    figure.subplots_adjust(
        left=0.08,
        right=0.89,
        top=0.88,
        bottom=0.18,
        wspace=0.28,
    )
    image = None
    for column, (axis, model) in enumerate(zip(axes, models)):
        values = stability_matrix(
            rows,
            model,
            history_budgets,
            rollout_budgets,
        )
        image = axis.imshow(
            values,
            vmin=0,
            vmax=1,
            cmap="cividis",
            aspect="auto",
            interpolation="nearest",
        )
        for row_index, _history_budget in enumerate(history_budgets):
            for column_index, _rollout_budget in enumerate(rollout_budgets):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="white" if value < 0.58 else INK,
                )
        axis.set_xticks(range(len(rollout_budgets)))
        axis.set_xticklabels(rollout_budgets, fontsize=7.5)
        axis.set_yticks(range(len(history_budgets)))
        axis.set_yticklabels(history_budgets, fontsize=7.5)
        axis.set_xlabel("rollouts per sampled history", fontsize=8.0)
        axis.set_ylabel("sampled histories", fontsize=8.0)
        axis.set_title(
            f"{chr(65 + column)}. {model}",
            fontsize=9.0,
            fontweight="bold",
        )
        axis.grid(False)
    assert image is not None
    colorbar = figure.colorbar(
        image,
        ax=list(axes),
        location="right",
        fraction=0.04,
        pad=0.04,
    )
    colorbar.set_label(
        "agreement with full-panel conclusion",
        fontsize=8.0,
    )
    colorbar.ax.tick_params(labelsize=7.2)
    save(
        figure,
        "fig_history_budget_appendix",
        formats=("pdf", "png"),
    )


def fig_artifact_variance() -> None:
    qwen = histories("Qwen3-30B")
    opus = histories("Opus 4.1")
    qwen_base = baseline("Qwen3-30B")
    opus_base = baseline("Opus 4.1")

    figure, axis = plt.subplots(figsize=(7.4, 4.3))

    def scatter(
        model_x: int,
        cells: dict[str, tuple[int, int]],
        color: str,
        marker: str,
    ) -> list[float]:
        values = []
        for index, (history_id, (harmful, n)) in enumerate(sorted(cells.items())):
            p, low, high = wilson(harmful, n)
            p *= 100
            x = model_x + (index - 3.5) * 0.055
            axis.errorbar(
                [x],
                [p],
                yerr=[[p - low * 100], [high * 100 - p]],
                fmt=marker,
                color=color,
                markersize=6,
                linewidth=1,
                capsize=2,
                ecolor=color,
                alpha=0.85,
                zorder=3,
            )
            values.append(p)
        return values

    qwen_values = scatter(1, qwen, GREEN, "o")
    opus_values = scatter(2, opus, BLUE, "s")
    for x, count in ((1, qwen_base), (2, opus_base)):
        rate = 100 * count[0] / count[1]
        axis.plot([x - 0.28, x + 0.28], [rate, rate], color=AMBER, linewidth=2.2)
        axis.text(
            x + 0.30,
            rate,
            "no-history\nbaseline",
            va="center",
            ha="left",
            fontsize=7.5,
            color=MUTED,
        )

    axis.text(
        1,
        6,
        f"8 histories\nSD {statistics.pstdev(qwen_values):.0f} pts\nall suppress",
        ha="center",
        fontsize=8.5,
        color=GREEN,
    )
    axis.text(
        2,
        82,
        f"8 histories\nSD {statistics.pstdev(opus_values):.0f} pts\nstraddle baseline",
        ha="center",
        fontsize=8.5,
        color=BLUE,
    )
    axis.set_xticks([1, 2])
    axis.set_xticklabels(["Qwen3-30B", "Opus 4.1"])
    axis.set_xlim(0.5, 2.6)
    axis.set_ylim(0, 90)
    axis.set_ylabel("Blackmail rate (%), full-history cell")
    axis.set_title(
        "One history is not a propensity: per-history rates vs. matched baseline",
        fontsize=10,
    )
    save(figure, "fig_artifact_variance", formats=("pdf", "png"))


def fig_history_robustness() -> None:
    panels = []
    for model in ("Qwen3-30B", "Mistral-Large"):
        cells = histories(model)
        base = baseline(model)
        pooled = combine(
            [
                {"harmful": harmful, "n": n}
                for harmful, n in cells.values()
            ]
        )
        odds_ratio, low, high = odds_ratio_interval(pooled, base)
        panels.append(
            (
                model,
                cells,
                base,
                f"{100 * pooled[0] / pooled[1]:.1f}% pooled\n"
                f"OR {odds_ratio:.2f} [{low:.2f}, {high:.2f}]",
            )
        )

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), sharey=True)
    for axis, (title, cells, base_count, summary) in zip(axes, panels):
        base_rate_raw, base_low, base_high = wilson(*base_count)
        base_rate = 100 * base_rate_raw
        axis.axhspan(
            100 * base_low,
            100 * base_high,
            color=AMBER,
            alpha=0.16,
            linewidth=0,
            zorder=0,
        )
        axis.axhline(base_rate, color=AMBER, linewidth=2.2, zorder=1)
        for index, (history_id, (harmful, n)) in enumerate(
            sorted(cells.items()), start=1
        ):
            p, low, high = wilson(harmful, n)
            p, low, high = p * 100, low * 100, high * 100
            axis.errorbar(
                index,
                p,
                yerr=[[p - low], [high - p]],
                fmt="o",
                color=GREEN,
                markersize=5.5,
                capsize=2,
                linewidth=1,
                ecolor=GREEN,
                zorder=3,
            )
        axis.text(
            8.9,
            base_rate,
            "baseline\n95% CI",
            va="center",
            ha="right",
            fontsize=7.2,
            color=MUTED,
        )
        axis.text(
            4.5,
            5,
            summary,
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=GREEN,
            fontweight="bold",
        )
        axis.set_title(title, fontsize=10, fontweight="bold")
        axis.set_xticks(range(1, 9))
        axis.set_xticklabels([f"h{i}" for i in range(1, 9)], fontsize=8)
        axis.set_xlim(0.5, 9.25)
        axis.set_xlabel("independent benign history")
    axes[0].set_ylim(0, 72)
    axes[0].set_ylabel("Harmful-action rate (%)")
    figure.suptitle(
        "Full-history rates across independent benign histories",
        fontsize=10.2,
        y=1.01,
    )
    save(figure, "fig_history_robustness")


def fig_opus_history_variance() -> None:
    cells = histories("Opus 4.1")
    route_baseline = baseline("Opus 4.1")
    native_h1 = one("opus_native_confirmation", "Opus 4.1", "history", "h1")
    pooled_others = combine(
        [
            {"harmful": harmful, "n": n}
            for history_id, (harmful, n) in cells.items()
            if history_id != "h1"
        ]
    )
    odds_ratio, low, high = odds_ratio_interval(
        pooled_others, route_baseline
    )
    baseline_rate = 100 * route_baseline[0] / route_baseline[1]

    figure, axis = plt.subplots(figsize=(6.5, 3.4))
    axis.axhline(baseline_rate, color=AMBER, linewidth=2.2, zorder=1)
    for index, (history_id, (harmful, n)) in enumerate(
        sorted(cells.items()), start=1
    ):
        p, ci_low, ci_high = wilson(harmful, n)
        p, ci_low, ci_high = p * 100, ci_low * 100, ci_high * 100
        color = VERMILION if history_id == "h1" else BLUE
        marker = "D" if history_id == "h1" else "s"
        axis.errorbar(
            index,
            p,
            yerr=[[p - ci_low], [ci_high - p]],
            fmt=marker,
            color=color,
            markersize=6,
            capsize=2,
            linewidth=1,
            ecolor=color,
            zorder=3,
        )
    axis.annotate(
        "original history\n"
        f"{100 * native_h1[0] / native_h1[1]:.1f}% native confirmation;\n"
        f"{100 * cells['h1'][0] / cells['h1'][1]:.1f}% route-matched rerun",
        xy=(1, 100 * cells["h1"][0] / cells["h1"][1]),
        xytext=(2.25, 82),
        fontsize=7.6,
        color=VERMILION,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1},
    )
    axis.text(
        8.15,
        baseline_rate,
        f"route-matched baseline\n({baseline_rate:.1f}%)",
        va="center",
        ha="left",
        fontsize=7.2,
        color=MUTED,
    )
    axis.text(
        5.1,
        24,
        f"histories h2-h8 pooled: {100 * pooled_others[0] / pooled_others[1]:.1f}%\n"
        f"OR {odds_ratio:.2f} [{low:.2f}, {high:.2f}]",
        ha="center",
        fontsize=8.2,
        color=BLUE,
        fontweight="bold",
    )
    axis.set_xticks(range(1, 9))
    axis.set_xticklabels([f"h{i}" for i in range(1, 9)])
    axis.set_xlim(0.5, 9.25)
    axis.set_ylim(15, 90)
    axis.set_xlabel("independent benign history")
    axis.set_ylabel("Blackmail rate (%)")
    axis.set_title(
        "Opus 4.1: a repeated single-history effect does not generalize",
        fontsize=10,
    )
    save(figure, "fig_opus_history_variance")


def fig_qwen_ladder() -> None:
    cells = [
        ("None", one("qwen_ladder", "Qwen3-30B", "none"), AMBER),
        ("Actions", one("qwen_ladder", "Qwen3-30B", "actions"), GREEN),
        (
            "Actions\n+Obs",
            one("qwen_ladder", "Qwen3-30B", "actions_results"),
            GREEN,
        ),
        ("Full\nhistory", one("qwen_ladder", "Qwen3-30B", "full"), GREEN),
    ]
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    x = range(len(cells))
    rates = [wilson(*cell[1]) for cell in cells]
    axis.bar(
        x,
        [rate[0] * 100 for rate in rates],
        0.62,
        yerr=[
            [(rate[0] - rate[1]) * 100 for rate in rates],
            [(rate[2] - rate[0]) * 100 for rate in rates],
        ],
        color=[cell[2] for cell in cells],
        capsize=4,
        ecolor=MUTED,
    )
    for index, rate in enumerate(rates):
        axis.text(
            index,
            rate[0] * 100 + 2,
            f"{rate[0] * 100:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_xticks(list(x))
    axis.set_xticklabels([cell[0] for cell in cells])
    axis.set_ylim(0, 60)
    axis.set_ylabel("Blackmail rate (%)")
    axis.set_title(
        "Qwen3-30B: a bare action log carries most of the suppression",
        fontsize=10,
    )
    axis.annotate(
        "",
        xy=(1, 30),
        xytext=(0, 30),
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    axis.text(
        0.5,
        32,
        "action log alone\n-25 pts",
        ha="center",
        fontsize=8.5,
        color=GREEN,
        fontweight="bold",
    )
    axis.annotate(
        "",
        xy=(3, 34),
        xytext=(2, 34),
        arrowprops={"arrowstyle": "->", "color": MUTED},
    )
    axis.text(
        2.5,
        36,
        "full rendering:\n-7 pts (not isolated)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    save(figure, "fig_qwen_ladder")


def fig_history_budget() -> None:
    rows = load_stability()
    models = ("Qwen3-30B", "Opus 4.1")
    history_budgets = (1, 2, 4, 8)
    rollout_budgets = (10, 20, 40, 80, 100)
    figure = plt.figure(figsize=(7.4, 5.1))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.72),
        hspace=0.42,
        wspace=0.24,
    )

    for column, model in enumerate(models):
        axis = figure.add_subplot(grid[0, column])
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["sampling"] == "random_histories"
        ]
        values = np.asarray(
            [
                [
                    next(
                        float(row["agreement_rate"])
                        for row in selected
                        if row["history_budget"] == history_budget
                        and row["rollouts_per_history"] == rollout_budget
                    )
                    for rollout_budget in rollout_budgets
                ]
                for history_budget in history_budgets
            ]
        )
        image = axis.imshow(
            values,
            vmin=0,
            vmax=1,
            cmap="cividis",
            aspect="auto",
            interpolation="nearest",
        )
        for row_index, history_budget in enumerate(history_budgets):
            for column_index, rollout_budget in enumerate(rollout_budgets):
                value = values[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value < 0.58 else INK,
                )
        axis.set_xticks(range(len(rollout_budgets)))
        axis.set_xticklabels(rollout_budgets, fontsize=8)
        axis.set_yticks(range(len(history_budgets)))
        axis.set_yticklabels(history_budgets, fontsize=8)
        axis.set_xlabel("rollouts per sampled history", fontsize=8.5)
        axis.set_ylabel("sampled histories", fontsize=8.5)
        axis.set_title(
            f"{chr(65 + column)}. {model}: random histories",
            fontsize=9.2,
            fontweight="bold",
        )
        axis.grid(False)

    colorbar = figure.colorbar(
        image,
        ax=[figure.axes[0], figure.axes[1]],
        location="right",
        fraction=0.035,
        pad=0.03,
    )
    colorbar.set_label(
        "agreement with full-panel conclusion", fontsize=8.5
    )
    colorbar.ax.tick_params(labelsize=8)

    axis = figure.add_subplot(grid[1, :])
    for model, color, marker in (
        ("Qwen3-30B", GREEN, "o"),
        ("Opus 4.1", VERMILION, "s"),
    ):
        fixed = sorted(
            (
                row
                for row in rows
                if row["model"] == model and row["sampling"] == "fixed_h1"
            ),
            key=lambda row: int(row["rollouts_per_history"]),
        )
        axis.plot(
            [int(row["rollouts_per_history"]) for row in fixed],
            [100 * float(row["agreement_rate"]) for row in fixed],
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
            label=model,
        )
    axis.set_ylim(-3, 103)
    axis.set_xlim(7, 103)
    axis.set_xticks(rollout_budgets)
    axis.set_xlabel("rollouts after the fixed original history $h_1$")
    axis.set_ylabel("agreement (%)")
    axis.set_title(
        "C. More rollouts lock in the panel-discordant, "
        "outcome-selected Opus $h_1$",
        fontsize=8.8,
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8.5, loc="center right")
    axis.annotate(
        "more precision,\nless panel agreement",
        xy=(80, 0.5),
        xytext=(58, 28),
        fontsize=7.8,
        color=VERMILION,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1},
    )
    save(figure, "fig_history_budget")


def fig_frontier_floor() -> None:
    measurable = ("Opus 4.1", "Qwen3-30B", "Gemini 2.5 Pro*")
    floor_rows = (
        ("Opus 4.8", "0/120 stronger"),
        ("Sonnet 4.6", "explicit refusals"),
        ("GPT-4.1", "judge = gate"),
        ("GPT-4o", "judge = gate"),
        ("Llama3.3-70B", "judge = gate"),
        ("Qwen2.5-72B", "two configurations"),
        ("Llama3.1-70B", "1/40 alternate"),
        ("Gemma2-27B", "explicit refusals"),
    )
    figure, axis = plt.subplots(figsize=(3.35, 4.62))
    figure.subplots_adjust(
        left=0.30,
        right=0.98,
        top=0.94,
        bottom=0.11,
    )

    measurable_y = (10.4, 9.3, 8.2)
    for y, model in zip(measurable_y, measurable):
        none = one("frontier_screen", model, "none")
        history = one("frontier_screen", model, "history")
        none_rate = none[0] / none[1]
        history_rate = history[0] / history[1]
        color = BLUE if history_rate > none_rate else GREEN
        axis.annotate(
            "",
            xy=(history_rate, y),
            xytext=(none_rate, y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
        )
        axis.plot(
            [none_rate],
            [y],
            marker="o",
            linestyle="none",
            color=AMBER,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            markersize=5.5,
            zorder=3,
        )
        axis.plot(
            [history_rate],
            [y],
            marker="s",
            linestyle="none",
            color=color,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            markersize=4.8,
            zorder=3,
        )

    floor_y = tuple(6.5 - 0.72 * index for index in range(len(floor_rows)))
    axis.axhspan(0.85, 7.05, color="#F4F4F4", zorder=0)
    for y, (_model, note) in zip(floor_y, floor_rows):
        axis.plot(
            [0],
            [y + 0.08],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=0.8,
            markersize=4.8,
            zorder=3,
        )
        axis.plot(
            [0],
            [y - 0.08],
            marker="s",
            linestyle="none",
            color=GREEN,
            markeredgecolor=INK,
            markeredgewidth=0.5,
            markersize=4.0,
            zorder=3,
        )
        axis.text(
            0.055,
            y,
            "0/40 both",
            fontsize=6.1,
            va="center",
            fontweight="bold",
        )
        axis.text(
            0.36,
            y,
            note,
            fontsize=5.9,
            va="center",
            color=MUTED,
        )

    axis.text(
        0,
        11.15,
        "Non-floor screens",
        fontsize=7.4,
        fontweight="bold",
        va="center",
    )
    axis.text(
        0,
        7.18,
        "Measurement-floor screens",
        fontsize=7.4,
        fontweight="bold",
        va="center",
    )
    axis.plot(
        [],
        [],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=INK,
        markersize=4.8,
        label="No history",
    )
    axis.plot(
        [],
        [],
        marker="s",
        linestyle="none",
        color=GREEN,
        markeredgecolor=INK,
        markeredgewidth=0.5,
        markersize=4.0,
        label="Full history",
    )
    axis.legend(
        frameon=False,
        fontsize=6.2,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
        borderaxespad=0.2,
    )

    labels = list(measurable) + [row[0] for row in floor_rows]
    positions = list(measurable_y) + list(floor_y)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=6.2)
    axis.set_xlim(-0.02, 1.0)
    axis.set_ylim(0.75, 11.55)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.tick_params(axis="x", labelsize=6.5)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("Harmful-action rate", fontsize=7.0)
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["top"].set_visible(False)
    save(figure, "fig_frontier_floor", formats=("pdf", "png"))


if __name__ == "__main__":
    fig_replication_pipeline()
    fig_renderer_sensitivity()
    fig_false_certainty()
    fig_history_budget_appendix()
    fig_artifact_variance()
    fig_history_robustness()
    fig_opus_history_variance()
    fig_qwen_ladder()
    fig_history_budget()
    fig_frontier_floor()
    print("all figures ->", FIG)
