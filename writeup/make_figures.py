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

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "writeup" / "figures"
DATA = ROOT / "results" / "paper" / "figure_counts.csv"
STABILITY_DATA = ROOT / "results" / "paper" / "history_budget_stability.csv"
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
            for source in raw["source"].split(";"):
                if not (ROOT / source).is_file():
                    raise FileNotFoundError(f"Missing figure source: {source}")
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


def save(figure: plt.Figure, name: str) -> None:
    figure.savefig(
        FIG / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print("wrote", name)


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
    save(figure, "fig_artifact_variance")


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
        "C. Conditional replication validates Qwen $h_1$ but locks in "
        "the unrepresentative Opus $h_1$",
        fontsize=9.2,
        fontweight="bold",
    )
    axis.legend(frameon=False, fontsize=8.5, loc="center right")
    axis.annotate(
        "more precision,\nless construct validity",
        xy=(80, 0.5),
        xytext=(58, 28),
        fontsize=7.8,
        color=VERMILION,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": VERMILION, "lw": 1},
    )
    save(figure, "fig_history_budget")


def fig_frontier_floor() -> None:
    order = [
        "Opus 4.1",
        "Qwen3-30B",
        "Gemini 2.5 Pro*",
        "GPT-4.1",
        "GPT-4o",
        "Llama-3.3-70B",
        "Sonnet 4.6",
        "Opus 4.8",
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    ys = list(range(len(order)))[::-1]
    for y, model in zip(ys, order):
        none = one("frontier_screen", model, "none")
        history = one("frontier_screen", model, "history")
        none_rate = none[0] / none[1]
        history_rate = history[0] / history[1]
        if none[0] == history[0] == 0:
            axis.plot([0], [y], "o", color=GRAY, markersize=7)
            axis.text(
                0.03,
                y,
                f"{model}: floor (0/{none[1]})",
                va="center",
                fontsize=8.5,
                color=MUTED,
            )
            continue
        color = BLUE if history_rate > none_rate else GREEN
        axis.annotate(
            "",
            xy=(history_rate, y),
            xytext=(none_rate, y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2},
        )
        axis.plot([none_rate], [y], "o", color=AMBER, markersize=7, zorder=3)
        axis.text(
            max(none_rate, history_rate) + 0.02,
            y,
            model,
            va="center",
            fontsize=8.5,
            color=INK,
        )
    axis.set_yticks([])
    axis.set_xlim(0, 1.05)
    axis.set_ylim(-0.6, len(order) - 0.4)
    axis.set_xlabel(
        "Blackmail rate (screen, $n$=40): None $\\rightarrow$ full history"
    )
    axis.set_title(
        "Where the effect is measurable: 5 of 8 models floor", fontsize=10
    )
    axis.annotate(
        "*Gemini thinking cannot be disabled; corroborating-only.",
        xy=(0, 0),
        xytext=(0.98, -0.45),
        fontsize=7.5,
        color=MUTED,
        ha="right",
    )
    save(figure, "fig_frontier_floor")


if __name__ == "__main__":
    fig_artifact_variance()
    fig_history_robustness()
    fig_opus_history_variance()
    fig_qwen_ladder()
    fig_history_budget()
    fig_frontier_floor()
    print("all figures ->", FIG)
