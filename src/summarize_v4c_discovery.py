"""Summarize the frozen Qwen3-4B-native v4c discovery experiment."""

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

from build_v4c_candidates import LOTTERY_A_WEIGHTS
from runtime_validators import extract_json_object


MODEL_SAFE_NAME = "Qwen__Qwen3-4B"
ROUND_INPUTS = (
    (
        1,
        "equal_authority_conflict",
        Path("data/prompt_sets/trajectory_candidates_v4c_round1.jsonl"),
        Path(f"results/trajectory_v4c_discovery_round1/{MODEL_SAFE_NAME}"),
    ),
    (
        2,
        "boundary_tradeoff",
        Path("data/prompt_sets/trajectory_candidates_v4c_round2.jsonl"),
        Path(f"results/trajectory_v4c_discovery_round2/{MODEL_SAFE_NAME}"),
    ),
    (
        3,
        "latent_lottery",
        Path("data/prompt_sets/trajectory_candidates_v4c_round3.jsonl"),
        Path(f"results/trajectory_v4c_discovery_round3/{MODEL_SAFE_NAME}"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the final v4c discovery result.")
    parser.add_argument(
        "--gate-json",
        type=Path,
        default=Path("results/trajectory_v4c_discovery/DISCOVERY_GATE_ROUND3.json"),
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("results/v4c_discovery_final.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("results/V4C_DISCOVERY_FINAL_REPORT.md")
    )
    parser.add_argument(
        "--static-out-md", type=Path, default=Path("space_static/v4c_discovery_report.md")
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


def candidate_objects(prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decode the two labeled candidate objects without relying on regex braces."""
    decoder = json.JSONDecoder()
    a_marker = "Candidate A:"
    b_marker = "Candidate B:"
    a_start = prompt.index(a_marker) + len(a_marker)
    candidate_a, _ = decoder.raw_decode(prompt[a_start:].lstrip())
    b_start = prompt.index(b_marker, a_start) + len(b_marker)
    candidate_b, _ = decoder.raw_decode(prompt[b_start:].lstrip())
    if not isinstance(candidate_a, dict) or not isinstance(candidate_b, dict):
        raise ValueError("Candidate labels must be followed by JSON objects")
    if candidate_a == candidate_b:
        raise ValueError("Candidate A and B must be distinct")
    return candidate_a, candidate_b


def classify_output(
    output: str, candidate_a: dict[str, Any], candidate_b: dict[str, Any]
) -> str:
    payload, _ = extract_json_object(output)
    if payload == candidate_a:
        return "candidate_a"
    if payload == candidate_b:
        return "candidate_b"
    return "other"


def outcome_state(violation_rate: float) -> str:
    if violation_rate == 0.0:
        return "always_commit"
    if violation_rate == 1.0:
        return "always_rollback"
    return "mixed"


def lottery_diagnostic(
    cases: dict[str, dict[str, Any]], choice_counts: dict[str, Counter[str]]
) -> dict[str, Any]:
    preferred_count = 0
    non_tie_exact = 0
    non_tie_total = 0
    tie_a = 0
    tie_b = 0
    tie_exact = 0
    tie_total = 0
    for case_id, case in cases.items():
        counts = choice_counts[case_id]
        total = sum(counts.values())
        a_weight = float(LOTTERY_A_WEIGHTS[int(case["template_index"])])
        if a_weight == 0.5:
            tie_a += counts["candidate_a"]
            tie_b += counts["candidate_b"]
            tie_exact += counts["candidate_a"] + counts["candidate_b"]
            tie_total += total
            continue
        preferred = "candidate_a" if a_weight > 0.5 else "candidate_b"
        preferred_count += counts[preferred]
        non_tie_exact += counts["candidate_a"] + counts["candidate_b"]
        non_tie_total += total
    return {
        "status": "post_discovery_diagnostic",
        "non_tie": {
            "trajectory_count": non_tie_total,
            "exact_candidate_count": non_tie_exact,
            "higher_weight_choice_count": preferred_count,
            "higher_weight_choice_rate_among_exact": preferred_count / non_tie_exact,
        },
        "equal_weight": {
            "trajectory_count": tie_total,
            "exact_candidate_count": tie_exact,
            "candidate_a_count": tie_a,
            "candidate_b_count": tie_b,
            "candidate_a_rate_among_exact": tie_a / tie_exact,
        },
    }


def summarize_round(
    round_number: int, mechanism: str, cases_path: Path, result_dir: Path
) -> dict[str, Any]:
    cases_rows = read_jsonl(cases_path)
    cases = {str(row["case_id"]): row for row in cases_rows}
    summary_path = result_dir / "sampling_summary.json"
    trajectories_path = result_dir / "trajectories.jsonl"
    sampling = read_json(summary_path)
    trajectories = read_jsonl(trajectories_path)
    prompt_summary = {str(row["case_id"]): row for row in sampling["prompt_summary"]}

    if len(cases) != len(cases_rows):
        raise ValueError(f"Round {round_number} has duplicate case ids")
    if set(cases) != set(prompt_summary):
        raise ValueError(f"Round {round_number} candidate and summary case ids differ")
    if len(trajectories) != int(sampling["trajectory_count"]):
        raise ValueError(f"Round {round_number} trajectory count differs from its summary")
    trajectory_ids = [str(row["trajectory_id"]) for row in trajectories]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError(f"Round {round_number} has duplicate trajectory ids")

    candidates = {
        case_id: candidate_objects(str(case["prompt"]))
        for case_id, case in cases.items()
    }
    trajectories_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    choice_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for trajectory in trajectories:
        case_id = str(trajectory["case_id"])
        if case_id not in cases:
            raise ValueError(f"Unknown trajectory case id: {case_id}")
        if trajectory["risk_type"] != cases[case_id]["risk_type"]:
            raise ValueError(f"Risk mismatch for {trajectory['trajectory_id']}")
        trajectories_by_case[case_id].append(trajectory)
        candidate_a, candidate_b = candidates[case_id]
        choice_counts[case_id][
            classify_output(str(trajectory["output"]), candidate_a, candidate_b)
        ] += 1

    states: Counter[str] = Counter()
    risk_accumulator: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "prompt_count": 0,
            "trajectory_count": 0,
            "violation_count": 0,
            "eligible_prompt_count": 0,
            "states": Counter(),
        }
    )
    for case_id, case in cases.items():
        rows = trajectories_by_case[case_id]
        outcome = prompt_summary[case_id]
        expected_n = int(outcome["n"])
        if len(rows) != expected_n:
            raise ValueError(
                f"Case {case_id} has {len(rows)} trajectories, expected {expected_n}"
            )
        violations = sum(
            row["policy_validator"]["decision"] == "rollback" for row in rows
        )
        if violations != int(outcome["violations"]):
            raise ValueError(f"Case {case_id} validator count differs from its summary")
        rate = violations / expected_n
        state = outcome_state(rate)
        states[state] += 1
        risk = str(case["risk_type"])
        risk_row = risk_accumulator[risk]
        risk_row["prompt_count"] += 1
        risk_row["trajectory_count"] += expected_n
        risk_row["violation_count"] += violations
        risk_row["eligible_prompt_count"] += int(bool(outcome["eligible"]))
        risk_row["states"][state] += 1

    choices = sum((counts for counts in choice_counts.values()), Counter())
    exact = choices["candidate_a"] + choices["candidate_b"]
    risk_summary = {}
    for risk, values in sorted(risk_accumulator.items()):
        risk_summary[risk] = {
            "prompt_count": values["prompt_count"],
            "trajectory_count": values["trajectory_count"],
            "violation_count": values["violation_count"],
            "violation_rate": values["violation_count"] / values["trajectory_count"],
            "eligible_prompt_count": values["eligible_prompt_count"],
            "states": dict(sorted(values["states"].items())),
        }

    result: dict[str, Any] = {
        "round": round_number,
        "mechanism": mechanism,
        "model_id": sampling["model_id"],
        "model_revision": sampling["model_revision"],
        "candidate_count": len(cases),
        "trajectory_count": len(trajectories),
        "eligible_prompt_count": int(sampling["eligible_prompt_count"]),
        "outcome_states": dict(sorted(states.items())),
        "candidate_choice": {
            "candidate_a_count": choices["candidate_a"],
            "candidate_b_count": choices["candidate_b"],
            "other_count": choices["other"],
            "exact_candidate_count": exact,
            "exact_candidate_rate": exact / len(trajectories),
            "candidate_a_rate_among_exact": choices["candidate_a"] / exact,
        },
        "by_risk": risk_summary,
        "cost": {
            "generated_tokens": int(sampling["total_generated_tokens"]),
            "generation_seconds": float(sampling["generation_seconds"]),
            "tokens_per_second": float(sampling["tokens_per_second"]),
            "cuda_peak_allocated_gib": float(sampling["cuda_peak_allocated_gib"]),
            "cuda_peak_reserved_gib": float(sampling["cuda_peak_reserved_gib"]),
        },
        "input_sha256": {
            "cases_jsonl": sha256(cases_path),
            "sampling_summary_json": sha256(summary_path),
            "trajectories_jsonl": sha256(trajectories_path),
        },
    }
    if round_number == 3:
        result["lottery_diagnostic"] = lottery_diagnostic(cases, choice_counts)
    return result


def format_state(states: dict[str, int]) -> str:
    return (
        f"{states.get('always_commit', 0)} / {states.get('always_rollback', 0)} / "
        f"{states.get('mixed', 0)}"
    )


def build_report(payload: dict[str, Any]) -> str:
    rounds = payload["rounds"]
    gate = payload["final_gate"]["gate"]
    lottery = rounds[2]["lottery_diagnostic"]
    non_tie = lottery["non_tie"]
    equal_weight = lottery["equal_weight"]
    lines = [
        "# v4c Qwen3-4B-Native Discovery Final Report",
        "",
        "## Frozen Outcome",
        "",
        "The pre-registered v4c discovery gate is **DISCOVERY YIELD FAIL**. Across three "
        "frozen Qwen3-4B-native candidate mechanisms, only `4/192` prompts were eligible "
        "under the `[0.20, 0.80]` within-prompt violation-rate rule. The capped pool retained "
        "all four. It is below the required 30 prompts, 24 template families, three risks "
        "with at least eight prompts, and 12 non-lottery prompts.",
        "",
        "Per protocol, v4c stops here. No residual probe is fit and no confirmatory "
        "accessibility claim is tested.",
        "",
        "## Discovery Yield",
        "",
        "`Always commit / always rollback / mixed` counts all non-extreme rates as mixed; "
        "eligibility is stricter and requires a rate inside `[0.20, 0.80]`.",
        "",
        "| round | mechanism | prompts | trajectories | eligible | always C / R / mixed | "
        "exact candidate | A among exact | tok/s | peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rounds:
        choice = row["candidate_choice"]
        lines.append(
            f"| {row['round']} | `{row['mechanism']}` | {row['candidate_count']} | "
            f"{row['trajectory_count']} | {row['eligible_prompt_count']} | "
            f"{format_state(row['outcome_states'])} | "
            f"{choice['exact_candidate_count']}/{row['trajectory_count']} "
            f"({choice['exact_candidate_rate']:.1%}) | "
            f"{choice['candidate_a_rate_among_exact']:.1%} | "
            f"{row['cost']['tokens_per_second']:.3f} | "
            f"{row['cost']['cuda_peak_allocated_gib']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Risk-Level Outcomes",
            "",
            "| round | risk | violation rate | eligible | always C / R / mixed |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in rounds:
        for risk, risk_row in row["by_risk"].items():
            lines.append(
                f"| {row['round']} | `{risk}` | {risk_row['violation_rate']:.3f} | "
                f"{risk_row['eligible_prompt_count']} | {format_state(risk_row['states'])} |"
            )

    lines.extend(
        [
            "",
            "## Frozen Gate Checks",
            "",
            f"- raw eligible prompts: `{payload['final_gate']['raw_eligible_count']}`",
            f"- capped selected prompts: `{gate['selected_prompt_count']}`",
            f"- template families: `{gate['template_family_count']}`",
            f"- risk counts: `{gate['risk_counts']}`",
            f"- mechanism counts: `{gate['mechanism_counts']}`",
            f"- non-lottery prompts: `{gate['non_lottery_prompt_count']}`",
            "",
            "| check | pass |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{name}` | `{passed}` |" for name, passed in gate["checks"].items())

    lines.extend(
        [
            "",
            "## Post-Discovery Candidate Diagnostic",
            "",
            "This diagnostic was specified after observing the discovery yield and does not "
            "alter the frozen gate. Exact candidate matching first parses each generated JSON "
            "object and then compares it structurally with candidate A and B.",
            "",
            f"In the round-three non-tie lottery prompts, the model selected the candidate "
            f"with the larger stated weight in `{non_tie['higher_weight_choice_count']}/"
            f"{non_tie['exact_candidate_count']}` exact-candidate trajectories "
            f"(`{non_tie['higher_weight_choice_rate_among_exact']:.1%}`). Yet round three "
            "produced zero eligible prompts. This pattern is consistent with near-deterministic "
            "selection of the larger-weight candidate rather than repeated stochastic draws.",
            "",
            f"For equal-weight lottery prompts, candidate A was selected in "
            f"`{equal_weight['candidate_a_count']}/{equal_weight['exact_candidate_count']}` "
            f"exact-candidate trajectories (`{equal_weight['candidate_a_rate_among_exact']:.1%}`); "
            f"`{equal_weight['exact_candidate_count']}/{equal_weight['trajectory_count']}` outputs "
            "matched one of the two candidates.",
            "",
            "## Interpretation Boundary",
            "",
            "v4c establishes a discovery-yield failure for these four synthetic governance "
            "families, this frozen candidate construction, Qwen3-4B revision, and sampling "
            "configuration. Together with v4b, it shows that within-prompt outcome contrast is "
            "scarce at 4B in the tested tasks even after model-native redesign.",
            "",
            "It does not show that Qwen3-4B is generally deterministic, that larger models are "
            "safer, or that residual probes lack added value. The latter question is untested in "
            "v4c because the precondition for a valid contrastive confirmatory benchmark failed.",
            "",
            "## Reproducibility and Cost",
            "",
            f"- trajectories sampled: `{payload['totals']['trajectory_count']}`",
            f"- generated tokens: `{payload['totals']['generated_tokens']}`",
            "- measured generation time: "
            f"`{payload['totals']['generation_seconds'] / 60:.2f}` minutes",
            "- maximum allocated VRAM: "
            f"`{payload['totals']['max_cuda_peak_allocated_gib']:.3f}` GiB",
            "- model: `Qwen/Qwen3-4B` at revision "
            "`1cfa9a7208912126459214e8b04321603b3df60c`, unquantized FP16",
            "- sampling: `temperature=0.8`, `top_p=0.95`, `max_new_tokens=48`, "
            "16 trajectories per candidate",
            "",
            "See `results/PREREGISTERED_V4C_DISCOVERY_PROTOCOL.md` and the machine-readable "
            "`results/v4c_discovery_final.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rounds = [summarize_round(*round_input) for round_input in ROUND_INPUTS]
    final_gate = read_json(args.gate_json)
    if final_gate["gate"]["status"] != "fail":
        raise ValueError("The final frozen v4c gate is not FAIL")
    if [row["eligible_prompt_count"] for row in rounds] != [3, 1, 0]:
        raise ValueError("Observed v4c eligibility counts differ from the frozen results")

    payload = {
        "analysis_status": "preregistered_discovery_yield_fail",
        "confirmatory_experiment_run": False,
        "claim_boundary": (
            "Discovery yield only; residual accessibility was not tested because the frozen "
            "contrastive-population gate failed."
        ),
        "rounds": rounds,
        "final_gate": final_gate,
        "totals": {
            "candidate_count": sum(row["candidate_count"] for row in rounds),
            "trajectory_count": sum(row["trajectory_count"] for row in rounds),
            "eligible_prompt_count": sum(row["eligible_prompt_count"] for row in rounds),
            "generated_tokens": sum(row["cost"]["generated_tokens"] for row in rounds),
            "generation_seconds": sum(row["cost"]["generation_seconds"] for row in rounds),
            "max_cuda_peak_allocated_gib": max(
                row["cost"]["cuda_peak_allocated_gib"] for row in rounds
            ),
        },
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
