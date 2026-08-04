from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

from eval.agentic_misalignment_native_tools.analyze_history_distribution import (
    analyze,
    history_renderer_interaction,
    leave_one_history_out,
)
from scripts import analyze_controlled_histories
from scripts.summarize_grader_audits import confusion_rows
from writeup import make_figures

ROOT = Path(__file__).resolve().parents[1]


class FigureDataTests(unittest.TestCase):
    def test_history_panels_are_complete(self) -> None:
        for model in ("Qwen3-30B", "Mistral-Large", "Opus 4.1"):
            self.assertEqual(
                set(make_figures.histories(model)),
                {f"h{index}" for index in range(1, 9)},
            )

    def test_headline_pooled_counts(self) -> None:
        qwen = make_figures.combine(
            [
                {"harmful": harmful, "n": n}
                for harmful, n in make_figures.histories("Qwen3-30B").values()
            ]
        )
        mistral = make_figures.combine(
            [
                {"harmful": harmful, "n": n}
                for harmful, n in make_figures.histories(
                    "Mistral-Large"
                ).values()
            ]
        )
        self.assertEqual(qwen, (244, 1090))
        self.assertEqual(mistral, (243, 640))
        self.assertEqual(make_figures.baseline("Qwen3-30B"), (121, 250))
        self.assertEqual(make_figures.baseline("Mistral-Large"), (88, 160))

    def test_opus_uses_route_matched_baseline(self) -> None:
        opus = make_figures.histories("Opus 4.1")
        pooled_others = make_figures.combine(
            [
                {"harmful": harmful, "n": n}
                for history_id, (harmful, n) in opus.items()
                if history_id != "h1"
            ]
        )
        baseline = make_figures.baseline("Opus 4.1")
        odds_ratio, low, high = make_figures.odds_ratio_interval(
            pooled_others, baseline
        )
        self.assertEqual(pooled_others, (376, 836))
        self.assertEqual(baseline, (69, 150))
        self.assertAlmostEqual(odds_ratio, 0.9595463)
        self.assertAlmostEqual(low, 0.6769836)
        self.assertAlmostEqual(high, 1.3600464)

    def test_qwen_ladder_counts(self) -> None:
        self.assertEqual(
            make_figures.one("qwen_ladder", "Qwen3-30B", "none"),
            (121, 250),
        )
        self.assertEqual(
            make_figures.one("qwen_ladder", "Qwen3-30B", "actions"),
            (56, 242),
        )
        self.assertEqual(
            make_figures.one(
                "qwen_ladder", "Qwen3-30B", "actions_results"
            ),
            (56, 242),
        )
        self.assertEqual(
            make_figures.one("qwen_ladder", "Qwen3-30B", "full"),
            (38, 240),
        )

    def test_controlled_history_level_estimates(self) -> None:
        rows = analyze_controlled_histories.load_rows()
        expected = {
            "Qwen3-30B": (0.221125, 0.484, -0.262875),
            "Mistral-Large": (0.3697916667, 0.55, -0.1802083333),
        }
        for index, (model, values) in enumerate(expected.items()):
            panel = analyze_controlled_histories.load_panel(model, rows=rows)
            result = analyze_controlled_histories.history_bootstrap(
                panel, draws=20_000, seed=20260803 + index
            )
            self.assertAlmostEqual(result["equal_history_mean"], values[0])
            self.assertAlmostEqual(result["baseline_rate"], values[1])
            self.assertAlmostEqual(result["mean_difference"], values[2])
            self.assertLess(result["bootstrap_ci_high"], 0)

        opus = analyze_controlled_histories.load_panel(
            "Opus 4.1",
            exclude_histories=frozenset({"h1"}),
            rows=rows,
        )
        result = analyze_controlled_histories.history_bootstrap(
            opus, draws=20_000, seed=20260806
        )
        self.assertAlmostEqual(result["equal_history_mean"], 0.4494371075)
        self.assertAlmostEqual(result["mean_difference"], -0.0105628925)
        self.assertLess(result["bootstrap_ci_low"], 0)
        self.assertGreater(result["bootstrap_ci_high"], 0)

    def test_fixed_h1_budget_exposes_opus_failure(self) -> None:
        with (
            ROOT / "results/paper/history_budget_stability.csv"
        ).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fixed = {
            (row["model"], int(row["rollouts_per_history"])): float(
                row["agreement_rate"]
            )
            for row in rows
            if row["sampling"] == "fixed_h1"
        }
        self.assertGreater(
            fixed[("Qwen3-30B", 100)], fixed[("Qwen3-30B", 10)]
        )
        self.assertLess(
            fixed[("Opus 4.1", 100)], fixed[("Opus 4.1", 10)]
        )
        self.assertGreater(fixed[("Qwen3-30B", 100)], 0.99)
        self.assertEqual(fixed[("Opus 4.1", 100)], 0.0)

    def test_grader_audit_counts(self) -> None:
        with (
            ROOT / "results/grader_invariance.csv"
        ).open(encoding="utf-8", newline="") as handle:
            independent = list(csv.DictReader(handle))
        key = json.loads(
            (ROOT / "results/adjudication_key.json").read_text(
                encoding="utf-8"
            )
        )
        confusion, human_labels = confusion_rows(independent, key)
        counts = {
            (
                row["audit"],
                row["condition"],
                row["reference_label"],
                row["audit_label"],
            ): row["count"]
            for row in confusion
        }
        self.assertEqual(len(independent), 128)
        self.assertEqual(len(human_labels), 33)
        self.assertEqual(
            counts[("independent_gpt4o", "A_raw", 1, 0)], 5
        )
        self.assertEqual(counts[("independent_gpt4o", "A3", 1, 0)], 6)
        self.assertEqual(
            counts[("single_author_human", "A_raw", 0, 1)], 4
        )
        self.assertEqual(
            counts[("single_author_human", "A3", 0, 1)], 2
        )


class NativeStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        summary = (
            ROOT
            / "results/native_tools/history_distribution_opus41_summary.csv"
        )
        with summary.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        (
            cls.condition_rows,
            cls.history_rows,
            cls.contrast_rows,
            cls.omnibus,
        ) = analyze(rows)
        cls.leave_one_out_rows = leave_one_history_out(cls.history_rows)
        cls.interaction = history_renderer_interaction(cls.history_rows)

    def test_prespecified_gate_and_counts(self) -> None:
        by_condition = {
            row["condition"]: row for row in self.condition_rows
        }
        self.assertEqual(by_condition["D0"]["harmful"], 140)
        self.assertEqual(by_condition["D1"]["harmful"], 143)
        self.assertEqual(by_condition["D2"]["harmful"], 122)
        self.assertEqual(by_condition["D3"]["harmful"], 113)
        self.assertEqual(by_condition["D3C"]["harmful"], 111)
        self.assertTrue(self.omnibus["claim_success_gate"])
        self.assertAlmostEqual(self.omnibus["p_value"], 0.003109968900310997)

    def test_primary_contrasts(self) -> None:
        contrasts = {
            row["contrast"]: row for row in self.contrast_rows
        }
        self.assertAlmostEqual(
            contrasts["D2-D0"]["mean_difference"], -0.10
        )
        self.assertAlmostEqual(
            contrasts["D3-D0"]["mean_difference"], -0.15
        )
        self.assertTrue(contrasts["D2-D0"]["contrast_passes_gate"])
        self.assertTrue(contrasts["D3-D0"]["contrast_passes_gate"])

    def test_posthoc_direct_renderer_contrasts(self) -> None:
        contrasts = {
            row["contrast"]: row for row in self.contrast_rows
        }
        with (
            ROOT
            / "results/native_tools/history_distribution_opus41_contrasts.csv"
        ).open(encoding="utf-8", newline="") as handle:
            committed = {
                row["contrast"]: row for row in csv.DictReader(handle)
            }
        expected = {
            "D1-D2": (0.11666666666666665, 0.04736328125, 0.0947265625),
            "D1-D3": (0.16666666666666666, 0.005126953125, 0.015380859375),
            "D2-D3": (0.050000000000000024, 0.33984375, 0.33984375),
        }
        for contrast, (
            difference,
            raw_p,
            adjusted_p,
        ) in expected.items():
            row = contrasts[contrast]
            self.assertEqual(row["family"], "posthoc_direct_renderer")
            self.assertAlmostEqual(row["mean_difference"], difference)
            self.assertAlmostEqual(row["exact_sign_flip_p"], raw_p)
            self.assertAlmostEqual(row["holm_adjusted_p"], adjusted_p)
            self.assertEqual(row["contrast_passes_gate"], "")
            self.assertEqual(
                committed[contrast]["family"], "posthoc_direct_renderer"
            )
            self.assertAlmostEqual(
                float(committed[contrast]["mean_difference"]), difference
            )
            self.assertAlmostEqual(
                float(committed[contrast]["exact_sign_flip_p"]), raw_p
            )
            self.assertAlmostEqual(
                float(committed[contrast]["holm_adjusted_p"]), adjusted_p
            )

    def test_leave_one_history_out_sensitivity(self) -> None:
        by_contrast = {
            contrast: [
                row
                for row in self.leave_one_out_rows
                if row["contrast"] == contrast
            ]
            for contrast in ("D2-D0", "D3-D0")
        }
        self.assertEqual(len(self.leave_one_out_rows), 45)
        self.assertTrue(
            all(
                row["same_direction_as_full_estimate"]
                for row in by_contrast["D2-D0"]
                + by_contrast["D3-D0"]
            )
        )
        self.assertEqual(
            sum(
                row["holm_significant_0_05"]
                for row in by_contrast["D2-D0"]
            ),
            10,
        )
        self.assertEqual(
            sum(
                row["holm_significant_0_05"]
                for row in by_contrast["D3-D0"]
            ),
            4,
        )

    def test_history_renderer_interaction(self) -> None:
        self.assertAlmostEqual(
            self.interaction["statistic"], 58.2869733755507
        )
        self.assertEqual(self.interaction["degrees_of_freedom"], 42)
        self.assertEqual(self.interaction["extreme_draws"], 2799)
        self.assertAlmostEqual(
            self.interaction["p_value"], 0.1399930003499825
        )
        self.assertEqual(self.interaction["fit_failures"], 0)
        self.assertFalse(self.interaction["interaction_detected_0_05"])

    def test_manifest_records_excluded_analysis_plan(self) -> None:
        manifest_path = (
            ROOT
            / "results/native_tools/history_distribution_opus41_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIsNone(manifest["analysis_plan"])
        provenance = manifest["analysis_plan_provenance"]
        self.assertEqual(
            provenance["original_sha256"],
            "2de285301a606c2229f001c8b9c502a3b8f4b36b1293ce1abb899f86fe685eeb",
        )
        self.assertEqual(
            manifest["analysis_plan_sha256"],
            provenance["original_sha256"],
        )
        self.assertFalse(provenance["public_copy_included"])
        self.assertFalse(provenance["externally_timestamped"])


if __name__ == "__main__":
    unittest.main()
