#!/usr/bin/env python
"""Plot the native Study 2 history-conditioned risk distribution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CONDITIONS = ("D0", "D1", "D2", "D3", "D3C")
MEMORY_CONDITIONS = ("D1", "D2", "D3", "D3C")
LABELS = {
    "D0": "Fresh\nD0",
    "D1": "Calls\nD1",
    "D2": "Calls + results\nD2",
    "D3": "Neutral\nD3",
    "D3C": "Cued\nD3C",
}
BLUE = "#0072B2"
VERMILION = "#D55E00"
AMBER = "#E69F00"
GRAY = "#6E6E6E"
LIGHT_GRAY = "#D9D9D9"
INK = "#222222"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(figure: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        prefix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def plot(
    history_rows: list[dict[str, str]],
    condition_rows: list[dict[str, str]],
    output_prefix: Path,
) -> None:
    by_condition = {row["condition"]: row for row in condition_rows}
    blocks = sorted({row["run_block"] for row in history_rows})
    by_cell = {
        (row["run_block"], row["condition"]): row for row in history_rows
    }
    if set(by_condition) != set(CONDITIONS):
        raise ValueError(f"Expected condition rows for {CONDITIONS}")
    if len(blocks) != 15 or len(by_cell) != 75:
        raise ValueError("Expected a complete 15-history by 5-condition analysis")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRAY,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "svg.hashsalt": "native-history-distribution-study2",
        }
    )
    figure = plt.figure(figsize=(12.0, 7.0))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.35),
        height_ratios=(1, 1.08),
        hspace=0.42,
        wspace=0.30,
    )
    mean_axis = figure.add_subplot(grid[0, 0])
    delta_axis = figure.add_subplot(grid[1, 0])
    heat_axis = figure.add_subplot(grid[:, 1])

    x = np.arange(len(CONDITIONS))
    means = np.asarray(
        [float(by_condition[condition]["history_mean_rate"]) for condition in CONDITIONS]
    )
    lows = np.asarray(
        [float(by_condition[condition]["history_mean_t_low"]) for condition in CONDITIONS]
    )
    highs = np.asarray(
        [float(by_condition[condition]["history_mean_t_high"]) for condition in CONDITIONS]
    )
    colors = [AMBER, BLUE, BLUE, BLUE, VERMILION]
    mean_axis.errorbar(
        x,
        means * 100,
        yerr=np.vstack(((means - lows) * 100, (highs - means) * 100)),
        fmt="none",
        ecolor=GRAY,
        elinewidth=1.3,
        capsize=4,
        zorder=2,
    )
    for index, condition in enumerate(CONDITIONS):
        values = np.asarray(
            [float(by_cell[(block, condition)]["harmful_rate"]) for block in blocks]
        )
        jitter = np.linspace(-0.14, 0.14, len(values))
        mean_axis.scatter(
            index + jitter,
            values * 100,
            s=15,
            color=colors[index],
            alpha=0.43,
            linewidths=0,
            zorder=1,
        )
        mean_axis.scatter(
            [index],
            [means[index] * 100],
            s=62,
            marker="D",
            color=colors[index],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        mean_axis.text(
            index,
            min(99, highs[index] * 100 + 4.2),
            f"{means[index] * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    mean_axis.set_xticks(x)
    mean_axis.set_xticklabels([LABELS[condition] for condition in CONDITIONS], fontsize=7.5)
    mean_axis.set_ylim(0, 100)
    mean_axis.set_ylabel("Harmful executed actions (%)")
    mean_axis.set_title(
        "A  Mean risk across 15 histories",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    mean_axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    mean_axis.set_axisbelow(True)

    delta_x = np.arange(len(MEMORY_CONDITIONS))
    for block_index, block in enumerate(blocks):
        values = [
            100 * float(by_cell[(block, condition)]["diff_vs_block_D0"])
            for condition in MEMORY_CONDITIONS
        ]
        delta_axis.plot(
            delta_x,
            values,
            color=GRAY,
            alpha=0.24,
            linewidth=0.75,
            zorder=1,
        )
        delta_axis.scatter(
            delta_x,
            values,
            color=GRAY,
            alpha=0.48,
            s=11,
            linewidths=0,
            zorder=2,
        )
    delta_means = [
        100 * float(by_condition[condition]["mean_diff_vs_D0"])
        for condition in MEMORY_CONDITIONS
    ]
    delta_axis.scatter(
        delta_x[:-1],
        delta_means[:-1],
        s=68,
        marker="D",
        color=BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        label="History mean",
    )
    delta_axis.scatter(
        [delta_x[-1]],
        [delta_means[-1]],
        s=68,
        marker="D",
        color=VERMILION,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    for index, value in enumerate(delta_means):
        delta_axis.text(
            index,
            value + (3.3 if value >= 0 else -3.3),
            f"{value:+.1f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )
    delta_axis.axhline(0, color=INK, linewidth=0.9)
    delta_axis.axhline(10, color=LIGHT_GRAY, linewidth=0.7, linestyle="--")
    delta_axis.axhline(-10, color=LIGHT_GRAY, linewidth=0.7, linestyle="--")
    delta_axis.set_xticks(delta_x)
    delta_axis.set_xticklabels(
        [LABELS[condition] for condition in MEMORY_CONDITIONS], fontsize=7.5
    )
    delta_axis.set_ylabel("Difference from block D0 (points)")
    delta_axis.set_title(
        "B  Within-history shifts",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    delta_axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    delta_axis.set_axisbelow(True)

    matrix = np.asarray(
        [
            [
                100 * float(by_cell[(block, condition)]["diff_vs_block_D0"])
                for condition in MEMORY_CONDITIONS
            ]
            for block in blocks
        ]
    )
    limit = max(10.0, float(np.ceil(np.max(np.abs(matrix)) / 10) * 10))
    image = heat_axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
    )
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            heat_axis.text(
                column_index,
                row_index,
                f"{value:+.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) >= 0.48 * limit else INK,
                fontweight="bold" if abs(value) >= 10 - 1e-9 else "normal",
            )
    heat_axis.set_xticks(np.arange(len(MEMORY_CONDITIONS)))
    heat_axis.set_xticklabels(
        [LABELS[condition] for condition in MEMORY_CONDITIONS], fontsize=8
    )
    heat_axis.set_yticks(np.arange(len(blocks)))
    heat_axis.set_yticklabels(blocks)
    heat_axis.set_ylabel("Independently generated prior history")
    heat_axis.set_title(
        "C  History-by-renderer risk shifts (percentage points)",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=heat_axis, fraction=0.045, pad=0.04)
    colorbar.set_label("Change from block-matched fresh start")
    colorbar.outline.set_linewidth(0.6)

    figure.text(
        0.02,
        0.01,
        "Diamonds are history means; bars are history-level t intervals. "
        "Heatmap labels show each 12-rollout cell minus its 12-rollout D0 block.",
        fontsize=7.5,
        color=GRAY,
    )
    _save(figure, output_prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    plot(
        _read(Path(args.history)),
        _read(Path(args.condition)),
        Path(args.out_prefix),
    )
    print(f"Wrote {args.out_prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
