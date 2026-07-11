from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_v4c_candidates import RISKS, build_round  # noqa: E402
from evaluate_v4c_discovery import discovery_gate, select_capped_pool  # noqa: E402


class V4CCandidateFreezeTests(unittest.TestCase):
    def test_all_rounds_are_balanced_and_target_absent(self) -> None:
        combined = []
        for round_number in (1, 2, 3):
            rows = build_round(round_number)
            self.assertEqual(len(rows), 64)
            self.assertEqual(Counter(row["risk_type"] for row in rows), {risk: 16 for risk in RISKS})
            self.assertTrue(all(not row["target_term_hits"] for row in rows))
            combined.extend(rows)
        families = Counter(row["template_family"] for row in combined)
        self.assertEqual(len(families), 32)
        self.assertEqual(set(families.values()), {6})

    def test_family_cap_keeps_distinct_mechanisms(self) -> None:
        base = {"template_family": "family", "risk_type": "risk"}
        rows = [
            {**base, "case_id": "r1a", "candidate_mechanism": "equal_authority_conflict", "discovery_violation_rate": 0.3},
            {**base, "case_id": "r1b", "candidate_mechanism": "equal_authority_conflict", "discovery_violation_rate": 0.4},
            {**base, "case_id": "r2", "candidate_mechanism": "boundary_tradeoff", "discovery_violation_rate": 0.5},
            {**base, "case_id": "r3", "candidate_mechanism": "latent_lottery", "discovery_violation_rate": 0.5},
        ]
        selected = select_capped_pool(rows)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["candidate_mechanism"] for row in selected}, {"boundary_tradeoff", "latent_lottery"})

    def test_discovery_gate_passes_only_with_frozen_coverage(self) -> None:
        selected = []
        for risk_index, risk in enumerate(("r1", "r2", "r3")):
            for index in range(10):
                selected.append(
                    {
                        "case_id": f"{risk}-{index}",
                        "risk_type": risk,
                        "template_family": f"{risk}-family-{index}",
                        "candidate_mechanism": (
                            "equal_authority_conflict" if (risk_index + index) % 2 == 0 else "boundary_tradeoff"
                        ),
                    }
                )
        gate = discovery_gate(selected, completed_rounds=2)
        self.assertEqual(gate["status"], "pass")
        self.assertTrue(all(gate["checks"].values()))


if __name__ == "__main__":
    unittest.main()
