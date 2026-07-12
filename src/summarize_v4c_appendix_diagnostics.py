"""Summarize the frozen v4c temperature and deployment-state appendix diagnostics."""

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


CASES_PATH = Path("data/prompt_sets/trajectory_candidates_v4c_round1.jsonl")
CONDITIONS = (
    {
        "condition_id": "qwen3_4b_t0p8",
        "group": "temperature",
        "result_dir": Path("results/trajectory_v4c_discovery_round1/Qwen__Qwen3-4B"),
        "precision": "float16",
    },
    {
        "condition_id": "qwen3_4b_t1p2",
        "group": "temperature",
        "result_dir": Path(
            "results/trajectory_v4c_appendix_temperature_t1p2/Qwen__Qwen3-4B"
        ),
        "precision": "float16",
    },
    {
        "condition_id": "qwen3_4b_t1p5",
        "group": "temperature",
        "result_dir": Path(
            "results/trajectory_v4c_appendix_temperature_t1p5/Qwen__Qwen3-4B"
        ),
        "precision": "float16",
    },
    {
        "condition_id": "gemma4_e2b_t0p8",
        "group": "deployment_model",
        "result_dir": Path(
            "results/trajectory_v4c_appendix_gemma4_e2b/gemma4__e2b"
        ),
        "precision": "Q4_K_M",
    },
    {
        "condition_id": "qwen35_4b_t0p8",
        "group": "deployment_model",
        "result_dir": Path(
            "results/trajectory_v4c_appendix_qwen35_4b/qwen3.5__4b"
        ),
        "precision": "Q4_K_M",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v4c appendix diagnostics.")
    parser.add_argument(
        "--out-json", type=Path, default=Path("results/v4c_appendix_diagnostics.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("results/V4C_APPENDIX_DIAGNOSTICS.md")
    )
    parser.add_argument(
        "--static-out-md",
        type=Path,
        default=Path("space_static/v4c_appendix_diagnostics.md"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_counts(prompt_summary: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(outcome_state(float(row["violation_rate"])) for row in prompt_summary)
    return {
        "always_commit": counts["always_commit"],
        "always_rollback": counts["always_rollback"],
        "mixed": counts["mixed"],
    }


def candidate_prompt_diagnostic(
    choices_by_case: dict[str, Counter[str]],
) -> dict[str, int]:
    return {
        "both_exact_candidates_prompt_count": sum(
            counts["candidate_a"] > 0 and counts["candidate_b"] > 0
            for counts in choices_by_case.values()
        ),
        "one_exact_candidate_prompt_count": sum(
            (counts["candidate_a"] > 0) ^ (counts["candidate_b"] > 0)
            for counts in choices_by_case.values()
        ),
        "no_exact_candidate_prompt_count": sum(
            counts["candidate_a"] == 0 and counts["candidate_b"] == 0
            for counts in choices_by_case.values()
        ),
        "prompts_with_other_output_count": sum(
            counts["other"] > 0 for counts in choices_by_case.values()
        ),
    }


def summarize_condition(
    cases_path: Path,
    result_dir: Path,
    *,
    condition_id: str,
    group: str,
    precision: str,
) -> dict[str, Any]:
    cases_rows = read_jsonl(cases_path)
    cases = {str(row["case_id"]): row for row in cases_rows}
    summary_path = result_dir / "sampling_summary.json"
    trajectories_path = result_dir / "trajectories.jsonl"
    sampling = read_json(summary_path)
    trajectories = read_jsonl(trajectories_path)
    prompt_summary = {str(row["case_id"]): row for row in sampling["prompt_summary"]}

    if len(cases) != 64 or len(cases) != len(cases_rows):
        raise ValueError(f"{condition_id}: expected 64 unique cases")
    if set(cases) != set(prompt_summary):
        raise ValueError(f"{condition_id}: prompt summary case ids differ")
    if len(trajectories) != 1024:
        raise ValueError(f"{condition_id}: expected 1,024 trajectories")
    trajectory_ids = [str(row["trajectory_id"]) for row in trajectories]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError(f"{condition_id}: duplicate trajectory ids")

    candidates = {
        case_id: candidate_objects(str(case["prompt"]))
        for case_id, case in cases.items()
    }
    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    choices_by_case: dict[str, Counter[str]] = defaultdict(Counter)
    choices: Counter[str] = Counter()
    for trajectory in trajectories:
        case_id = str(trajectory["case_id"])
        if case_id not in cases:
            raise ValueError(f"{condition_id}: unknown case id {case_id}")
        rows_by_case[case_id].append(trajectory)
        candidate_a, candidate_b = candidates[case_id]
        choice = classify_output(str(trajectory["output"]), candidate_a, candidate_b)
        choices[choice] += 1
        choices_by_case[case_id][choice] += 1

    by_risk: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "prompt_count": 0,
            "trajectory_count": 0,
            "violation_count": 0,
            "eligible_prompt_count": 0,
            "states": Counter(),
        }
    )
    ordered_prompt_summary = []
    for case_id, case in cases.items():
        outcome = prompt_summary[case_id]
        rows = rows_by_case[case_id]
        if len(rows) != 16 or int(outcome["n"]) != 16:
            raise ValueError(f"{condition_id}: case {case_id} does not have 16 samples")
        violations = sum(
            row["policy_validator"]["decision"] == "rollback" for row in rows
        )
        if violations != int(outcome["violations"]):
            raise ValueError(f"{condition_id}: validator mismatch for {case_id}")
        rate = violations / len(rows)
        if rate != float(outcome["violation_rate"]):
            raise ValueError(f"{condition_id}: rate mismatch for {case_id}")
        risk = str(case["risk_type"])
        risk_row = by_risk[risk]
        risk_row["prompt_count"] += 1
        risk_row["trajectory_count"] += len(rows)
        risk_row["violation_count"] += violations
        risk_row["eligible_prompt_count"] += int(bool(outcome["eligible"]))
        risk_row["states"][outcome_state(rate)] += 1
        ordered_prompt_summary.append(outcome)

    risk_payload = {}
    for risk, values in sorted(by_risk.items()):
        risk_payload[risk] = {
            "prompt_count": values["prompt_count"],
            "trajectory_count": values["trajectory_count"],
            "violation_count": values["violation_count"],
            "violation_rate": values["violation_count"] / values["trajectory_count"],
            "eligible_prompt_count": values["eligible_prompt_count"],
            "states": {
                "always_commit": values["states"]["always_commit"],
                "always_rollback": values["states"]["always_rollback"],
                "mixed": values["states"]["mixed"],
            },
        }

    exact = choices["candidate_a"] + choices["candidate_b"]
    model_details = (sampling.get("model_metadata") or {}).get("details") or {}
    return {
        "condition_id": condition_id,
        "group": group,
        "model_id": sampling["model_id"],
        "model_revision": sampling.get("model_revision"),
        "backend": sampling.get("sampling_backend", "transformers"),
        "precision": precision,
        "model_family": model_details.get("family"),
        "model_parameter_size": model_details.get("parameter_size"),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "seed_start": int(sampling["seed_start"]),
        "prompt_count": len(cases),
        "trajectory_count": len(trajectories),
        "eligible_prompt_count": int(sampling["eligible_prompt_count"]),
        "states": state_counts(ordered_prompt_summary),
        "candidate_choice": {
            "candidate_a_count": choices["candidate_a"],
            "candidate_b_count": choices["candidate_b"],
            "other_count": choices["other"],
            "exact_candidate_count": exact,
            "exact_candidate_rate": exact / len(trajectories),
            **candidate_prompt_diagnostic(choices_by_case),
        },
        "by_risk": risk_payload,
        "cost": {
            "generated_tokens": int(sampling["total_generated_tokens"]),
            "generation_seconds": float(sampling["generation_seconds"]),
            "tokens_per_second": float(sampling["tokens_per_second"]),
            "cuda_peak_allocated_gib": sampling.get("cuda_peak_allocated_gib"),
        },
        "source_sha256": {
            "cases_jsonl": sha256(cases_path),
            "sampling_summary_json": sha256(summary_path),
            "trajectories_jsonl": sha256(trajectories_path),
        },
    }


def format_states(states: dict[str, int]) -> str:
    return (
        f"{states['always_commit']} / {states['always_rollback']} / "
        f"{states['mixed']}"
    )


def build_report(payload: dict[str, Any]) -> str:
    by_id = {row["condition_id"]: row for row in payload["conditions"]}
    temperature_rows = [
        by_id["qwen3_4b_t0p8"],
        by_id["qwen3_4b_t1p2"],
        by_id["qwen3_4b_t1p5"],
    ]
    deployment_rows = [
        by_id["gemma4_e2b_t0p8"],
        by_id["qwen35_4b_t0p8"],
    ]
    lines = [
        "# v4c Post-Hoc Appendix Diagnostics",
        "",
        "These diagnostics were frozen after the v4c discovery result. They do not "
        "reopen the failed gate, select a new prompt pool, or authorize residual probes.",
        "",
        "## Temperature Sensitivity",
        "",
        "The same 64 round-one prompts were evaluated on the same Qwen3-4B FP16 "
        "model. The original T=0.8 run is reused; T=1.2 and T=1.5 use fresh seeds.",
        "",
        "| temperature | seed start | eligible | always C / R / mixed | "
        "candidate match | A/B-switch prompts | tok/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in temperature_rows:
        choice = row["candidate_choice"]
        lines.append(
            f"| {row['temperature']:.1f} | {row['seed_start']} | "
            f"{row['eligible_prompt_count']}/64 | {format_states(row['states'])} | "
            f"{choice['exact_candidate_count']}/1024 "
            f"({choice['exact_candidate_rate']:.1%}) | "
            f"{choice['both_exact_candidates_prompt_count']}/64 | "
            f"{row['cost']['tokens_per_second']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Deployment-State Model Controls",
            "",
            "These are Ollama Q4_K_M deployment artifacts with native chat renderers. "
            "They are descriptive model controls, not an unconfounded FP16 scale curve.",
            "",
            "| model | role | digest | eligible | always C / R / mixed | "
            "candidate match | A/B-switch prompts | tok/s |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    roles = {
        "gemma4_e2b_t0p8": "cross-family",
        "qwen35_4b_t0p8": "same-family newer generation",
    }
    for row in deployment_rows:
        choice = row["candidate_choice"]
        lines.append(
            f"| `{row['model_id']}` | {roles[row['condition_id']]} | "
            f"`{str(row['model_revision'])[:12]}` | "
            f"{row['eligible_prompt_count']}/64 | {format_states(row['states'])} | "
            f"{choice['exact_candidate_count']}/1024 "
            f"({choice['exact_candidate_rate']:.1%}) | "
            f"{choice['both_exact_candidates_prompt_count']}/64 | "
            f"{row['cost']['tokens_per_second']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Risk-Level Outcomes",
            "",
            "| condition | risk | violation rate | eligible | always C / R / mixed |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["conditions"]:
        for risk, risk_row in row["by_risk"].items():
            lines.append(
                f"| `{row['condition_id']}` | `{risk}` | "
                f"{risk_row['violation_rate']:.3f} | "
                f"{risk_row['eligible_prompt_count']} | "
                f"{format_states(risk_row['states'])} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            f"The eligible counts across Qwen3-4B temperatures 0.8, 1.2, and 1.5 "
            f"are `{temperature_rows[0]['eligible_prompt_count']}`, "
            f"`{temperature_rows[1]['eligible_prompt_count']}`, and "
            f"`{temperature_rows[2]['eligible_prompt_count']}`. Mixed counts are "
            f"`{temperature_rows[0]['states']['mixed']}`, "
            f"`{temperature_rows[1]['states']['mixed']}`, and "
            f"`{temperature_rows[2]['states']['mixed']}`.",
            "",
            f"At T=0.8, the deployment controls yield "
            f"`{deployment_rows[0]['eligible_prompt_count']}/64` eligible prompts for "
            f"Gemma and `{deployment_rows[1]['eligible_prompt_count']}/64` for Qwen3.5. "
            "These counts describe the frozen deployments only.",
            "",
            "A post-hoc output-fidelity audit finds both exact candidate A and B in "
            f"`{deployment_rows[0]['candidate_choice']['both_exact_candidates_prompt_count']}/64` "
            f"Gemma prompts and "
            f"`{deployment_rows[1]['candidate_choice']['both_exact_candidates_prompt_count']}/64` "
            "Qwen3.5 prompts. The Qwen3.5 contrast therefore cannot be explained only by "
            "its non-candidate outputs.",
            "",
            "No appendix result changes the v4c `DISCOVERY YIELD FAIL`. Temperature, "
            "backend, quantization, native rendering, and model generation prevent a "
            "universal determinism claim or a causal family comparison.",
            "",
            "See `results/PREREGISTERED_V4C_APPENDIX_DIAGNOSTICS.md` and "
            "`results/v4c_appendix_diagnostics.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    conditions = [
        summarize_condition(CASES_PATH, **condition) for condition in CONDITIONS
    ]
    payload = {
        "analysis_status": "posthoc_appendix_no_new_gate",
        "main_v4c_gate_unchanged": True,
        "confirmatory_probe_run": False,
        "source_cases_sha256": sha256(CASES_PATH),
        "conditions": conditions,
    }
    report = build_report(payload)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(report, encoding="utf-8")
    args.static_out_md.parent.mkdir(parents=True, exist_ok=True)
    args.static_out_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out_json}, {args.out_md}, and {args.static_out_md}")


if __name__ == "__main__":
    main()
