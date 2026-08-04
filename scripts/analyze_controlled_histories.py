#!/usr/bin/env python
"""Analyze controlled studies with histories as the replication units."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "paper" / "figure_counts.csv"
SUMMARY_OUTPUT = (
    ROOT / "results" / "paper" / "controlled_history_inference.csv"
)
STABILITY_OUTPUT = (
    ROOT / "results" / "paper" / "history_budget_stability.csv"
)
FINDINGS_OUTPUT = (
    ROOT / "results" / "paper" / "controlled_history_inference_FINDINGS.md"
)

BOOTSTRAP_DRAWS = 200_000
STABILITY_DRAWS = 50_000
SEED = 20260803
PRACTICAL_THRESHOLD = 0.10
ROLLOUT_BUDGETS = (10, 20, 40, 80, 100)
HISTORY_BUDGETS = (1, 2, 4, 8)


@dataclass(frozen=True)
class Cell:
    history_id: str
    harmful: int
    n: int

    @property
    def rate(self) -> float:
        return self.harmful / self.n


@dataclass(frozen=True)
class Panel:
    model: str
    histories: tuple[Cell, ...]
    baseline_harmful: int
    baseline_n: int

    @property
    def baseline_rate(self) -> float:
        return self.baseline_harmful / self.baseline_n


def load_rows(path: Path = INPUT) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def load_panel(
    model: str,
    *,
    exclude_histories: frozenset[str] = frozenset(),
    rows: list[dict[str, str]] | None = None,
) -> Panel:
    source = rows if rows is not None else load_rows()
    selected = [
        row
        for row in source
        if row["dataset"] == "history_robustness"
        and row["model"] == model
    ]
    histories = tuple(
        sorted(
            (
                Cell(
                    history_id=row["history_id"],
                    harmful=int(row["harmful"]),
                    n=int(row["n"]),
                )
                for row in selected
                if row["condition"] == "history"
                and row["history_id"] not in exclude_histories
            ),
            key=lambda cell: cell.history_id,
        )
    )
    baselines = [row for row in selected if row["condition"] == "baseline"]
    if not histories or not baselines:
        raise ValueError(f"Incomplete controlled panel for {model}")
    if any(not 0 <= cell.harmful <= cell.n for cell in histories):
        raise ValueError(f"Invalid history count for {model}")
    baseline_harmful = sum(int(row["harmful"]) for row in baselines)
    baseline_n = sum(int(row["n"]) for row in baselines)
    if not 0 <= baseline_harmful <= baseline_n:
        raise ValueError(f"Invalid baseline count for {model}")
    return Panel(model, histories, baseline_harmful, baseline_n)


def classify_effect(
    effect: np.ndarray | float,
    threshold: float = PRACTICAL_THRESHOLD,
) -> np.ndarray:
    values = np.asarray(effect)
    return np.where(
        values <= -threshold,
        -1,
        np.where(values >= threshold, 1, 0),
    )


def history_bootstrap(
    panel: Panel,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict[str, float | int]:
    rates = np.asarray([cell.rate for cell in panel.histories])
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, len(rates), size=(draws, len(rates))
    )
    sampled_history_means = rates[sampled_indices].mean(axis=1)
    sampled_baselines = (
        rng.binomial(
            panel.baseline_n,
            panel.baseline_rate,
            size=draws,
        )
        / panel.baseline_n
    )
    effects = sampled_history_means - sampled_baselines
    low, high = np.quantile(effects, (0.025, 0.975))
    mean_rate = float(rates.mean())
    effect = mean_rate - panel.baseline_rate
    return {
        "history_count": len(rates),
        "equal_history_mean": mean_rate,
        "baseline_rate": panel.baseline_rate,
        "mean_difference": effect,
        "bootstrap_ci_low": float(low),
        "bootstrap_ci_high": float(high),
        "histories_below_baseline": int(
            np.sum(rates < panel.baseline_rate)
        ),
        "histories_equal_baseline": int(
            np.sum(rates == panel.baseline_rate)
        ),
        "histories_above_baseline": int(
            np.sum(rates > panel.baseline_rate)
        ),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def sample_panel_rates(
    panel: Panel,
    *,
    rollout_budget: int,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rollout_budget > panel.baseline_n:
        raise ValueError("Rollout budget exceeds baseline sample size")
    if any(rollout_budget > cell.n for cell in panel.histories):
        raise ValueError("Rollout budget exceeds a history sample size")
    history_rates = np.column_stack(
        [
            rng.hypergeometric(
                cell.harmful,
                cell.n - cell.harmful,
                rollout_budget,
                size=draws,
            )
            / rollout_budget
            for cell in panel.histories
        ]
    )
    baseline_rates = (
        rng.hypergeometric(
            panel.baseline_harmful,
            panel.baseline_n - panel.baseline_harmful,
            rollout_budget,
            size=draws,
        )
        / rollout_budget
    )
    return history_rates, baseline_rates


def stability_rows(
    panel: Panel,
    *,
    draws: int = STABILITY_DRAWS,
    seed: int = SEED,
    threshold: float = PRACTICAL_THRESHOLD,
) -> list[dict[str, object]]:
    if len(panel.histories) != 8:
        raise ValueError("Stability analysis expects eight histories")
    rates = np.asarray([cell.rate for cell in panel.histories])
    full_effect = float(rates.mean() - panel.baseline_rate)
    full_class = int(classify_effect(full_effect, threshold))
    output: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for rollout_budget in ROLLOUT_BUDGETS:
        history_rates, baseline_rates = sample_panel_rates(
            panel,
            rollout_budget=rollout_budget,
            draws=draws,
            rng=rng,
        )
        for history_budget in HISTORY_BUDGETS:
            random_order = np.argsort(
                rng.random((draws, len(panel.histories))), axis=1
            )
            sampled_indices = random_order[:, :history_budget]
            sampled_rates = np.take_along_axis(
                history_rates, sampled_indices, axis=1
            )
            effects = sampled_rates.mean(axis=1) - baseline_rates
            classes = classify_effect(effects, threshold)
            output.append(
                {
                    "model": panel.model,
                    "sampling": "random_histories",
                    "history_budget": history_budget,
                    "rollouts_per_history": rollout_budget,
                    "full_panel_effect": full_effect,
                    "full_panel_class": full_class,
                    "agreement_rate": float(np.mean(classes == full_class)),
                    "lower_rate": float(np.mean(classes == -1)),
                    "no_large_shift_rate": float(np.mean(classes == 0)),
                    "higher_rate": float(np.mean(classes == 1)),
                    "threshold": threshold,
                    "draws": draws,
                    "seed": seed,
                }
            )

        fixed_effects = history_rates[:, 0] - baseline_rates
        fixed_classes = classify_effect(fixed_effects, threshold)
        output.append(
            {
                "model": panel.model,
                "sampling": "fixed_h1",
                "history_budget": 1,
                "rollouts_per_history": rollout_budget,
                "full_panel_effect": full_effect,
                "full_panel_class": full_class,
                "agreement_rate": float(
                    np.mean(fixed_classes == full_class)
                ),
                "lower_rate": float(np.mean(fixed_classes == -1)),
                "no_large_shift_rate": float(np.mean(fixed_classes == 0)),
                "higher_rate": float(np.mean(fixed_classes == 1)),
                "threshold": threshold,
                "draws": draws,
                "seed": seed,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def main() -> int:
    rows = load_rows()
    panels = [
        ("Qwen3-30B", "all_h1_h8", load_panel("Qwen3-30B", rows=rows)),
        (
            "Mistral-Large",
            "all_h1_h8_posthoc",
            load_panel("Mistral-Large", rows=rows),
        ),
        ("Opus 4.1", "all_h1_h8", load_panel("Opus 4.1", rows=rows)),
        (
            "Opus 4.1",
            "h2_h8",
            load_panel(
                "Opus 4.1",
                exclude_histories=frozenset({"h1"}),
                rows=rows,
            ),
        ),
    ]
    summary_rows: list[dict[str, object]] = []
    for index, (model, subset, panel) in enumerate(panels):
        summary_rows.append(
            {
                "model": model,
                "subset": subset,
                "baseline_harmful": panel.baseline_harmful,
                "baseline_n": panel.baseline_n,
                **history_bootstrap(panel, seed=SEED + index),
            }
        )
    write_csv(SUMMARY_OUTPUT, summary_rows)

    stability: list[dict[str, object]] = []
    for index, model in enumerate(("Qwen3-30B", "Opus 4.1")):
        stability.extend(
            stability_rows(
                load_panel(model, rows=rows),
                seed=SEED + 100 + index,
            )
        )
    write_csv(STABILITY_OUTPUT, stability)

    by_key = {
        (str(row["model"]), str(row["subset"])): row
        for row in summary_rows
    }
    qwen = by_key[("Qwen3-30B", "all_h1_h8")]
    mistral = by_key[("Mistral-Large", "all_h1_h8_posthoc")]
    opus = by_key[("Opus 4.1", "h2_h8")]
    fixed_opus = [
        row
        for row in stability
        if row["model"] == "Opus 4.1" and row["sampling"] == "fixed_h1"
    ]
    fixed_start = next(
        row for row in fixed_opus if row["rollouts_per_history"] == 10
    )
    fixed_end = next(
        row for row in fixed_opus if row["rollouts_per_history"] == 100
    )
    FINDINGS_OUTPUT.write_text(
        "\n".join(
            [
                "# Controlled History-Level Inference",
                "",
                "Equal-history means give each source history one unit of "
                "weight. Percentile intervals resample histories as clusters "
                "and independently resample the shared baseline count.",
                "",
                f"- Qwen3-30B: {percent(float(qwen['equal_history_mean']))} "
                f"versus {percent(float(qwen['baseline_rate']))}; mean shift "
                f"{percent(float(qwen['mean_difference']))}, 95% bootstrap "
                f"CI [{percent(float(qwen['bootstrap_ci_low']))}, "
                f"{percent(float(qwen['bootstrap_ci_high']))}]. All 8/8 "
                "observed history rates are below baseline.",
                f"- Mistral-Large: "
                f"{percent(float(mistral['equal_history_mean']))} versus "
                f"{percent(float(mistral['baseline_rate']))}; mean shift "
                f"{percent(float(mistral['mean_difference']))}, 95% bootstrap "
                f"CI [{percent(float(mistral['bootstrap_ci_low']))}, "
                f"{percent(float(mistral['bootstrap_ci_high']))}]. The "
                "eight-history combination is post hoc because four histories "
                "were selected for additional sampling after the screen.",
                f"- Opus 4.1 h2-h8: "
                f"{percent(float(opus['equal_history_mean']))} versus "
                f"{percent(float(opus['baseline_rate']))}; mean shift "
                f"{percent(float(opus['mean_difference']))}, 95% bootstrap "
                f"CI [{percent(float(opus['bootstrap_ci_low']))}, "
                f"{percent(float(opus['bootstrap_ci_high']))}].",
                "",
                "## Finite-Panel Conclusion Stability",
                "",
                "The post hoc stability analysis classifies effects below "
                "-10 points as lower, above +10 points as higher, and effects "
                "between those thresholds as no large shift. It resamples "
                "without replacement from the saved outcomes and describes "
                "the finite eight-history panels.",
                "",
                "- For fixed Opus h1, agreement with the eight-history "
                f"conclusion falls from "
                f"{percent(float(fixed_start['agreement_rate']))} at 10 "
                f"rollouts to {percent(float(fixed_end['agreement_rate']))} "
                "at 100 rollouts. Additional conditional replication makes "
                "the unrepresentative h1 increase more precise.",
                "- Qwen is the positive control: its h1 and full panel have "
                "the same lower-risk classification, so additional rollouts "
                "increase agreement.",
                "",
                f"Bootstrap draws: {BOOTSTRAP_DRAWS:,}; stability draws per "
                f"cell: {STABILITY_DRAWS:,}; seed: {SEED}.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {SUMMARY_OUTPUT}")
    print(f"wrote {STABILITY_OUTPUT}")
    print(f"wrote {FINDINGS_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
