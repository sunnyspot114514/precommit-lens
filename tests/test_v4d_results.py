from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results" / "trajectory_v4d_confirmatory" / "Qwen__Qwen3.5-4B"


class V4dResultTests(unittest.TestCase):
    def test_confirmatory_gate_is_valid_and_fails(self) -> None:
        analysis = json.loads((RUN_DIR / "v4d_analysis.json").read_text())
        self.assertEqual(analysis["trajectory_count"], 1056)
        self.assertEqual(analysis["gate"]["status"], "fail")
        self.assertTrue(analysis["gate"]["contrast_yield_pass"])
        self.assertEqual(analysis["gate"]["winning_checkpoints"], [])
        self.assertEqual(analysis["confirmatory_yield"]["test"]["mixed_prompt_count"], 8)

    def test_primary_curve_is_exactly_at_chance(self) -> None:
        analysis = json.loads((RUN_DIR / "v4d_analysis.json").read_text())
        rows = {
            row["checkpoint"]: row
            for row in analysis["checkpoint_results"]
            if row["checkpoint"] in [2, 4, 6, 8, 10, 12]
        }
        for row in rows.values():
            self.assertEqual(row["metrics"]["residual_layer_21"]["auc"], 0.5)
            self.assertEqual(row["metrics"]["visible_prefix_tfidf"]["auc"], 0.5)
            self.assertEqual(row["metrics"]["next_token_stats"]["auc"], 0.5)
            self.assertEqual(row["metrics"]["prefix_model_judge"]["auc"], 0.5)
            self.assertEqual(row["primary_comparison"]["delta"], 0.0)

    def test_identity_diagnostic_does_not_change_gate(self) -> None:
        diagnostic = json.loads((RUN_DIR / "v4d_prelanding_identity.json").read_text())
        self.assertFalse(diagnostic["gate_changed"])
        self.assertFalse(diagnostic["new_checkpoint_sampled"])
        self.assertEqual(diagnostic["cross_label_first_difference"]["at_landing"], 1014)
        self.assertEqual(diagnostic["cross_label_first_difference"]["before_landing"], 714)
        self.assertEqual(diagnostic["prelanding_lead_tokens"], {"2": 714})
        for row in diagnostic["checkpoint_rows"]:
            if row["checkpoint"] <= 16:
                self.assertEqual(
                    row["identical_all_layer_feature_prompts"], row["evaluable_test_prompts"]
                )

    def test_cost_outputs_are_paired_and_identical(self) -> None:
        cost = json.loads((RUN_DIR / "monitoring_cost_benchmark.json").read_text())
        self.assertEqual(cost["paired_runs"], 16)
        self.assertTrue(cost["all_outputs_identical"])

    def test_final_summary_enforces_stop(self) -> None:
        summary = json.loads((ROOT / "results" / "v4d_final_summary.json").read_text())
        self.assertTrue(summary["stopping"]["experimental_development_complete"])
        self.assertFalse(summary["stopping"]["derive_v4e"])


if __name__ == "__main__":
    unittest.main()
