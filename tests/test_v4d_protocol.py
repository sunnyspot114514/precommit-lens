from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_dense_jlens_qwen import get_core_model, get_hidden_size, get_layers  # noqa: E402
from v4d_protocol import evaluate_stage1_gate  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class V4dProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = read_jsonl(
            ROOT / "data" / "prompt_sets" / "trajectory_candidates_v4c_round1.jsonl"
        )

    def test_protocol_freezes_final_branch_and_model(self) -> None:
        config = yaml.safe_load((ROOT / "configs" / "trajectory_v4d.yaml").read_text())
        protocol = (ROOT / "results" / "PREREGISTERED_V4D_PROTOCOL.md").read_text()
        self.assertEqual(config["model_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(config["stage2"]["primary_layer"], 21)
        self.assertTrue(config["stopping"]["final_experiment_version"])
        self.assertFalse(config["stopping"]["derive_v4e"])
        self.assertIn("No result authorizes a v4e", protocol)

    def test_qwen35_q4_motivation_passes_frozen_population_gate(self) -> None:
        summary = json.loads(
            (
                ROOT
                / "results"
                / "trajectory_v4c_appendix_qwen35_4b"
                / "qwen3.5__4b"
                / "sampling_summary.json"
            ).read_text()
        )
        gate = evaluate_stage1_gate(self.cases, summary["prompt_summary"])
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["selected_prompt_count"], 30)
        self.assertEqual(gate["selected_template_family_count"], 22)
        self.assertEqual(
            gate["supported_risks"], ["early_spoiler", "fake_commit", "hidden_fields"]
        )

    def test_qwen3_fp16_round_one_stops_under_same_gate(self) -> None:
        summary = json.loads(
            (
                ROOT
                / "results"
                / "trajectory_v4c_discovery_round1"
                / "Qwen__Qwen3-4B"
                / "sampling_summary.json"
            ).read_text()
        )
        gate = evaluate_stage1_gate(self.cases, summary["prompt_summary"])
        self.assertEqual(gate["status"], "stop")
        self.assertFalse(gate["stage2_triggered"])

    def test_qwen35_nested_text_model_is_supported(self) -> None:
        layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        language_model = SimpleNamespace(layers=layers, norm=torch.nn.Identity())
        model = SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        self.assertEqual(get_layers(model), list(layers))
        self.assertIs(get_core_model(model), language_model)
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2560))
        self.assertEqual(get_hidden_size(config), 2560)


if __name__ == "__main__":
    unittest.main()
