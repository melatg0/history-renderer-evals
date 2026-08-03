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
from scripts.analyze_history_consistency import analyze_qwen
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

    def test_qwen_history_consistency(self) -> None:
        with (
            ROOT / "results/paper/figure_counts.csv"
        ).open(encoding="utf-8", newline="") as handle:
            result = analyze_qwen(list(csv.DictReader(handle)))
        self.assertEqual(result["histories_below_baseline"], 8)
        self.assertEqual(result["histories_above_baseline"], 0)
        self.assertEqual(result["ties"], 0)
        self.assertAlmostEqual(result["p_value"], 0.0078125)

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

    def test_manifest_hashes_current_analysis_plan(self) -> None:
        manifest_path = (
            ROOT
            / "results/native_tools/history_distribution_opus41_manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        plan = ROOT / str(manifest["analysis_plan"])
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["analysis_plan_sha256"])
        provenance = manifest["analysis_plan_provenance"]
        self.assertEqual(
            provenance["original_sha256"],
            "2de285301a606c2229f001c8b9c502a3b8f4b36b1293ce1abb899f86fe685eeb",
        )
        self.assertTrue(provenance["public_copy_relabeled_after_completion"])
        self.assertFalse(provenance["externally_timestamped"])


if __name__ == "__main__":
    unittest.main()
