from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_v4b_collapse_direction import collapse_state, pairwise_auc  # noqa: E402


class CollapseDirectionTests(unittest.TestCase):
    def test_collapse_state(self) -> None:
        self.assertEqual(collapse_state(0.0), "always_commit")
        self.assertEqual(collapse_state(1.0), "always_rollback")
        self.assertEqual(collapse_state(0.5), "mixed")

    def test_pairwise_auc_counts_ties_as_half(self) -> None:
        positive = np.asarray([0.5, 0.9])
        negative = np.asarray([0.5, 0.1])
        self.assertEqual(pairwise_auc(positive, negative), 0.875)


if __name__ == "__main__":
    unittest.main()
