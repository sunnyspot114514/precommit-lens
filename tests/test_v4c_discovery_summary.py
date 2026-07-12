from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_v4c_discovery import (  # noqa: E402
    candidate_objects,
    classify_output,
    lottery_diagnostic,
    outcome_state,
)


class V4CDiscoverySummaryTests(unittest.TestCase):
    def test_candidate_parser_and_structural_output_match(self) -> None:
        prompt = (
            'Context. Candidate A: {"status": "pending", "nested": {"x": 1}} '
            'Candidate B: {"status": "done"} Return one.'
        )
        candidate_a, candidate_b = candidate_objects(prompt)
        self.assertEqual(candidate_a["nested"], {"x": 1})
        self.assertEqual(
            classify_output(
                '```json\n{"status":"done"}\n```', candidate_a, candidate_b
            ),
            "candidate_b",
        )
        self.assertEqual(
            classify_output('{"status":"other"}', candidate_a, candidate_b), "other"
        )

    def test_mixed_state_is_broader_than_eligibility(self) -> None:
        self.assertEqual(outcome_state(0.0), "always_commit")
        self.assertEqual(outcome_state(1.0), "always_rollback")
        self.assertEqual(outcome_state(0.0625), "mixed")

    def test_lottery_diagnostic_separates_non_ties_and_ties(self) -> None:
        cases = {
            "a_high": {"template_index": 4},  # A=0.60
            "b_high": {"template_index": 3},  # A=0.40
            "tie": {"template_index": 1},  # A=0.50
        }
        choices = {
            "a_high": Counter(candidate_a=3, candidate_b=1),
            "b_high": Counter(candidate_a=1, candidate_b=3),
            "tie": Counter(candidate_a=3, candidate_b=1, other=1),
        }
        diagnostic = lottery_diagnostic(cases, choices)
        self.assertEqual(diagnostic["non_tie"]["higher_weight_choice_count"], 6)
        self.assertEqual(diagnostic["non_tie"]["exact_candidate_count"], 8)
        self.assertEqual(diagnostic["equal_weight"]["trajectory_count"], 5)
        self.assertEqual(diagnostic["equal_weight"]["exact_candidate_count"], 4)


if __name__ == "__main__":
    unittest.main()
