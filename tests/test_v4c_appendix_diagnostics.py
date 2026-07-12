from __future__ import annotations

import hashlib
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ollama_trajectory_sampling import (  # noqa: E402
    build_chat_payload,
    normalized_digest,
    sampling_options,
    summarize_prompt_outcomes,
    validate_model_metadata,
)
from summarize_v4c_appendix_diagnostics import (  # noqa: E402
    candidate_prompt_diagnostic,
    state_counts,
)


class V4CAppendixProtocolTests(unittest.TestCase):
    def test_frozen_config_matches_round_one_and_has_no_new_gate(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs/v4c_appendix_diagnostics.yaml").read_text(
                encoding="utf-8"
            )
        )
        cases_path = ROOT / config["source_cases"]["path"]
        digest = hashlib.sha256(cases_path.read_bytes()).hexdigest()
        self.assertEqual(digest, config["source_cases"]["sha256"])
        self.assertEqual(
            [row["temperature"] for row in config["temperature_diagnostic"]["conditions"]],
            [0.8, 1.2, 1.5],
        )
        self.assertEqual(
            [row["model_id"] for row in config["cross_model_diagnostic"]["models"]],
            ["gemma4:e2b", "qwen3.5:4b"],
        )
        self.assertTrue(config["stopping"]["no_cross_model_round2"])
        self.assertTrue(config["stopping"]["no_confirmatory_probe"])
        self.assertTrue(config["stopping"]["no_new_success_gate"])

    def test_ollama_payload_freezes_sampling_and_disables_thinking(self) -> None:
        args = Namespace(
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            num_ctx=512,
            max_new_tokens=48,
        )
        options = sampling_options(args)
        payload = build_chat_payload(
            "model:tag",
            {"system": "system", "prompt": "user"},
            seed=123,
            options=options,
            keep_alive="10m",
        )
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["options"]["seed"], 123)
        self.assertEqual(payload["options"]["top_k"], 50)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")

    def test_digest_and_quantization_must_match(self) -> None:
        metadata = {
            "digest": "abc123",
            "details": {"quantization_level": "Q4_K_M"},
        }
        validate_model_metadata(metadata, "sha256:abc123", "Q4_K_M")
        self.assertEqual(normalized_digest("sha256:ABC123"), "abc123")
        with self.assertRaises(ValueError):
            validate_model_metadata(metadata, "wrong", "Q4_K_M")
        with self.assertRaises(ValueError):
            validate_model_metadata(metadata, "abc123", "Q8_0")

    def test_prompt_outcomes_keep_mixed_and_eligible_distinct(self) -> None:
        prompts = [
            {"case_id": "a", "risk_type": "risk", "condition": "condition"},
            {"case_id": "b", "risk_type": "risk", "condition": "condition"},
        ]
        trajectories = []
        for index in range(16):
            trajectories.append(
                {
                    "case_id": "a",
                    "policy_validator": {
                        "decision": "rollback" if index == 0 else "commit"
                    },
                }
            )
            trajectories.append(
                {
                    "case_id": "b",
                    "policy_validator": {
                        "decision": "rollback" if index < 8 else "commit"
                    },
                }
            )
        summary, eligible = summarize_prompt_outcomes(
            prompts, trajectories, minimum_rate=0.2, maximum_rate=0.8
        )
        self.assertEqual(eligible, {"b"})
        self.assertEqual(state_counts(summary)["mixed"], 2)

    def test_candidate_prompt_diagnostic_requires_both_exact_outputs(self) -> None:
        from collections import Counter

        diagnostic = candidate_prompt_diagnostic(
            {
                "both": Counter(candidate_a=4, candidate_b=3, other=1),
                "one": Counter(candidate_a=8, candidate_b=0, other=2),
                "neither": Counter(candidate_a=0, candidate_b=0, other=16),
            }
        )
        self.assertEqual(diagnostic["both_exact_candidates_prompt_count"], 1)
        self.assertEqual(diagnostic["one_exact_candidate_prompt_count"], 1)
        self.assertEqual(diagnostic["no_exact_candidate_prompt_count"], 1)
        self.assertEqual(diagnostic["prompts_with_other_output_count"], 3)


if __name__ == "__main__":
    unittest.main()
