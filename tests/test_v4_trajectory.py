from __future__ import annotations

import hashlib
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_v4_trajectories import (  # noqa: E402
    fit_logistic,
    paired_envelope_difference,
    prompt_auc_map,
)


class ConfirmatoryFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = ROOT / "data" / "prompt_sets" / "trajectory_confirmatory_v4.jsonl"
        cls.rows = [json.loads(line) for line in cls.path.read_text(encoding="utf-8").splitlines()]
        cls.config = yaml.safe_load(
            (ROOT / "configs" / "trajectory_confirmatory_v4.yaml").read_text(encoding="utf-8")
        )
        cls.v4b_config = yaml.safe_load(
            (ROOT / "configs" / "trajectory_confirmatory_v4b.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_hash_and_prompt_count(self) -> None:
        self.assertEqual(len(self.rows), 34)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.config["dataset_sha256"])

    def test_template_families_do_not_cross_splits(self) -> None:
        by_family: dict[str, set[str]] = defaultdict(set)
        for row in self.rows:
            by_family[row["template_family"]].add(row["trajectory_split"])
        self.assertEqual(len(by_family), 24)
        self.assertTrue(all(len(splits) == 1 for splits in by_family.values()))

    def test_expected_split_counts(self) -> None:
        counts = defaultdict(int)
        for row in self.rows:
            counts[(row["risk_type"], row["trajectory_split"])] += 1
        self.assertEqual(counts[("early_spoiler", "train")], 10)
        self.assertEqual(counts[("hidden_fields", "validation")], 2)
        self.assertEqual(counts[("schema_bypass", "test")], 2)

    def test_v4b_reuses_frozen_population_and_depth_mapping(self) -> None:
        self.assertEqual(self.v4b_config["dataset_sha256"], self.config["dataset_sha256"])
        self.assertEqual(self.v4b_config["sampling"], {
            **self.config["sampling"],
            "max_seq_len": 256,
            "dtype": "float16",
        })
        self.assertEqual(self.v4b_config["features"]["layers"], [0, 8, 16, 23, 31, 35])
        self.assertEqual(self.v4b_config["features"]["primary_layer"], 23)
        self.assertFalse(self.v4b_config["discovery"]["model_specific_prompt_replacement"])


class PairwiseMetricTests(unittest.TestCase):
    def test_prompt_auc_ties_count_as_half(self) -> None:
        scores = np.asarray([0.5, 0.5, 0.8, 0.2])
        labels = np.asarray([1, 0, 1, 0])
        prompts = np.asarray(["a", "a", "b", "b"])
        risks = np.asarray(["r1", "r1", "r2", "r2"])
        splits = np.asarray(["test"] * 4)
        values = prompt_auc_map(
            scores, labels, prompts, risks, splits, np.ones(4, dtype=np.bool_), "test"
        )
        self.assertEqual(values["a"]["auc"], 0.5)
        self.assertEqual(values["b"]["auc"], 1.0)

    def test_envelope_bootstrap_is_deterministic(self) -> None:
        def values(a: float, b: float) -> dict[str, dict[str, object]]:
            return {
                "p1": {"auc": a, "risk_type": "r1"},
                "p2": {"auc": b, "risk_type": "r2"},
            }

        first = paired_envelope_difference(
            values(0.9, 0.8), values(0.5, 0.6), values(0.4, 0.7), samples=50, seed=7
        )
        second = paired_envelope_difference(
            values(0.9, 0.8), values(0.5, 0.6), values(0.4, 0.7), samples=50, seed=7
        )
        self.assertEqual(first, second)
        self.assertGreater(first["delta"], 0)

    def test_probe_reports_insufficient_training_contrast(self) -> None:
        features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        labels = np.asarray([0, 0, 0, 1])
        prompts = np.asarray(["train", "train", "test", "test"])
        splits = np.asarray(["train", "train", "test", "test"])
        scores, payload = fit_logistic(
            features,
            labels,
            prompts,
            splits,
            np.ones(4, dtype=np.bool_),
            standardize=True,
            epochs=1,
            device=torch.device("cpu"),
            seed=1,
        )
        self.assertEqual(payload["status"], "insufficient_training_contrast")
        self.assertTrue(np.isnan(scores).all())


if __name__ == "__main__":
    unittest.main()
