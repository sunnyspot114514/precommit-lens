"""Evaluate the pre-registered v4d FP16 Transformers transfer gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_v4c_discovery import candidate_objects, classify_output, outcome_state
from v4d_protocol import evaluate_stage1_gate


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CASES_SHA256 = "74f8d38d023c41fbeb3b790f39bbbf46f2e99d6814431bd576d7c13bbfcc114e"
EXPECTED_SEED_START = 33000000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen v4d stage-one gate.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_candidates_v4c_round1.jsonl"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/trajectory_v4d_stage1/Qwen__Qwen3.5-4B"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_run(
    cases_path: Path,
    cases: list[dict[str, Any]],
    sampling: dict[str, Any],
    trajectories: list[dict[str, Any]],
) -> None:
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_count": 64,
        "trajectory_count": 1024,
        "samples_per_prompt": 16,
        "seed_start": EXPECTED_SEED_START,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 48,
        "capture_checkpoints": False,
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"Frozen v4d field {key}: expected {value!r}, got {sampling.get(key)!r}")
    if sha256(cases_path) != CASES_SHA256:
        raise ValueError("v4d stage-one candidate SHA-256 mismatch")
    if len(cases) != 64 or len({str(row["case_id"]) for row in cases}) != 64:
        raise ValueError("v4d stage one requires 64 unique prompts")
    if len(trajectories) != 1024:
        raise ValueError("v4d stage one requires exactly 1,024 trajectories")
    ids = [str(row["trajectory_id"]) for row in trajectories]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate v4d trajectory ids")


def candidate_diagnostic(
    cases: list[dict[str, Any]], trajectories: list[dict[str, Any]]
) -> dict[str, Any]:
    candidate_by_case = {
        str(row["case_id"]): candidate_objects(str(row["prompt"])) for row in cases
    }
    choices: Counter[str] = Counter()
    by_case: dict[str, Counter[str]] = defaultdict(Counter)
    for row in trajectories:
        case_id = str(row["case_id"])
        candidate_a, candidate_b = candidate_by_case[case_id]
        choice = classify_output(str(row["output"]), candidate_a, candidate_b)
        choices[choice] += 1
        by_case[case_id][choice] += 1
    exact = choices["candidate_a"] + choices["candidate_b"]
    return {
        "candidate_a_count": choices["candidate_a"],
        "candidate_b_count": choices["candidate_b"],
        "other_count": choices["other"],
        "exact_candidate_count": exact,
        "exact_candidate_rate": exact / len(trajectories),
        "both_exact_candidates_prompt_count": sum(
            row["candidate_a"] > 0 and row["candidate_b"] > 0 for row in by_case.values()
        ),
    }


def main() -> None:
    args = parse_args()
    summary_path = args.run_dir / "sampling_summary.json"
    trajectories_path = args.run_dir / "trajectories.jsonl"
    cases = read_jsonl(args.cases)
    sampling = read_json(summary_path)
    trajectories = read_jsonl(trajectories_path)
    validate_frozen_run(args.cases, cases, sampling, trajectories)

    prompt_summary = sampling["prompt_summary"]
    cases_by_id = {str(row["case_id"]): row for row in cases}
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        case_id = str(row["case_id"])
        if case_id not in cases_by_id:
            raise ValueError(f"Unknown v4d case id {case_id}")
        rows_by_case[case_id].append(row)
    outcomes_by_id = {str(row["case_id"]): row for row in prompt_summary}
    states: Counter[str] = Counter()
    for case_id in cases_by_id:
        rows = rows_by_case[case_id]
        outcome = outcomes_by_id[case_id]
        if len(rows) != 16 or int(outcome["n"]) != 16:
            raise ValueError(f"Case {case_id} does not have 16 samples")
        violations = sum(row["policy_validator"]["decision"] == "rollback" for row in rows)
        if violations != int(outcome["violations"]):
            raise ValueError(f"Validator count mismatch for {case_id}")
        rate = violations / len(rows)
        if rate != float(outcome["violation_rate"]):
            raise ValueError(f"Violation rate mismatch for {case_id}")
        if bool(outcome["eligible"]) != (0.2 <= rate <= 0.8):
            raise ValueError(f"Eligibility mismatch for {case_id}")
        states[outcome_state(rate)] += 1

    gate = evaluate_stage1_gate(cases, prompt_summary)
    payload = {
        "analysis_status": "preregistered_v4d_stage1_gate",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "precision": "float16",
        "candidate_sha256": sha256(args.cases),
        "sampling_summary_sha256": sha256(summary_path),
        "trajectories_sha256": sha256(trajectories_path),
        "outcome_states": {
            "always_commit": states["always_commit"],
            "always_rollback": states["always_rollback"],
            "mixed": states["mixed"],
        },
        "candidate_choice": candidate_diagnostic(cases, trajectories),
        "cost": {
            "generated_tokens": sampling["total_generated_tokens"],
            "generation_seconds": sampling["generation_seconds"],
            "tokens_per_second": sampling["tokens_per_second"],
            "cuda_peak_allocated_gib": sampling["cuda_peak_allocated_gib"],
            "cuda_peak_reserved_gib": sampling["cuda_peak_reserved_gib"],
        },
        "gate": gate,
    }
    json_path = args.run_dir / "STAGE1_GATE.json"
    md_path = args.run_dir / "STAGE1_GATE.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    exact = payload["candidate_choice"]
    lines = [
        "# v4d Stage-One FP16 Transfer Gate",
        "",
        f"- status: **{gate['status'].upper()}**",
        f"- stage 2 triggered: `{gate['stage2_triggered']}`",
        f"- raw eligible prompts: `{gate['raw_eligible_count']}/64`",
        f"- supported risks: `{gate['supported_risks']}`",
        f"- selected prompts / families: `{gate['selected_prompt_count']}` / "
        f"`{gate['selected_template_family_count']}`",
        f"- grouped split: `{gate['split_counts']}`",
        f"- outcome states C / R / mixed: `{states['always_commit']} / "
        f"{states['always_rollback']} / {states['mixed']}`",
        f"- exact candidate outputs: `{exact['exact_candidate_count']}/1024` "
        f"(`{100 * exact['exact_candidate_rate']:.1f}%`)",
        f"- prompts switching between exact A/B: "
        f"`{exact['both_exact_candidates_prompt_count']}/64`",
        f"- generation throughput: `{sampling['tokens_per_second']:.3f}` tok/s",
        "",
        "## Frozen Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | `{passed}` |" for name, passed in gate["checks"].items())
    lines.extend(
        [
            "",
            "## Risk Support",
            "",
            "| risk | eligible prompts | eligible families |",
            "|---|---:|---:|",
        ]
    )
    for risk, count in gate["risk_counts"].items():
        lines.append(f"| `{risk}` | {count} | {gate['risk_family_counts'][risk]} |")
    lines.extend(
        [
            "",
            "A pass authorizes only the frozen v4d confirmatory stage. A stop identifies "
            "deployment-state dependence but does not isolate model generation, quantization, "
            "backend, or chat rendering.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"v4d stage-one gate: {gate['status']}")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
