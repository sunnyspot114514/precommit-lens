"""Summarize frozen-prompt transfer from the v4 0.6B run to v4b 4B."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the v4/v4b cross-scale result.")
    parser.add_argument(
        "--v4-dir",
        type=Path,
        default=Path("results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B"),
    )
    parser.add_argument(
        "--v4b-dir",
        type=Path,
        default=Path("results/trajectory_v4b_confirmatory/Qwen__Qwen3-4B"),
    )
    parser.add_argument("--out-md", type=Path, default=Path("results/V4_V4B_CROSS_SCALE_REPORT.md"))
    parser.add_argument(
        "--static-out-md", type=Path, default=Path("space_static/v4b_cross_scale_report.md")
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("results/v4_v4b_cross_scale_report.json")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def outcome_states(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    states: Counter[str] = Counter()
    by_risk: dict[str, Counter[str]] = defaultdict(Counter)
    for group in grouped.values():
        violations = sum(row["policy_validator"]["decision"] == "rollback" for row in group)
        state = (
            "always_commit"
            if violations == 0
            else "always_rollback"
            if violations == len(group)
            else "mixed"
        )
        states[state] += 1
        by_risk[str(group[0]["risk_type"])][state] += 1
    return {
        "prompt_count": len(grouped),
        "states": dict(sorted(states.items())),
        "by_risk": {risk: dict(sorted(values.items())) for risk, values in sorted(by_risk.items())},
    }


def yield_row(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        split: {
            "mixed": int(analysis["confirmatory_yield"][split]["mixed_prompt_count"]),
            "total": int(analysis["confirmatory_yield"][split]["prompt_count"]),
            "risks": int(analysis["confirmatory_yield"][split]["mixed_risk_count"]),
        }
        for split in ("train", "validation", "test")
    }


def fmt_yield(value: dict[str, int]) -> str:
    return f"{value['mixed']}/{value['total']} ({value['risks']} risks)"


def main() -> None:
    args = parse_args()
    v4 = read_json(args.v4_dir / "v4_analysis.json")
    v4b = read_json(args.v4b_dir / "v4b_analysis.json")
    v4_sampling = read_json(args.v4_dir / "sampling_summary.json")
    v4b_sampling = read_json(args.v4b_dir / "sampling_summary.json")
    v4_states = outcome_states(read_jsonl(args.v4_dir / "trajectories.jsonl"))
    v4b_states = outcome_states(read_jsonl(args.v4b_dir / "trajectories.jsonl"))
    v4_yield = yield_row(v4)
    v4b_yield = yield_row(v4b)
    v4_cost = v4["cost"]["monitoring_benchmark"]
    v4b_cost = v4b["cost"]["monitoring_benchmark"]

    transfer_pass = v4b_yield["test"]["mixed"] >= 6 and v4b_yield["test"]["risks"] >= 2
    payload = {
        "design": {
            "same_frozen_prompt_population": True,
            "prompts": 34,
            "trajectories_per_model": 1088,
            "samples_per_prompt": 32,
        },
        "qwen3_0_6b": {
            "model_id": v4_sampling["model_id"],
            "gate": v4["gate"],
            "yield": v4_yield,
            "outcome_states": v4_states,
            "capture_to_plain_ratio": v4_cost["capture_to_plain_ratio"],
        },
        "qwen3_4b": {
            "model_id": v4b_sampling["model_id"],
            "model_revision": v4b_sampling["model_revision"],
            "gate": v4b["gate"],
            "yield": v4b_yield,
            "outcome_states": v4b_states,
            "capture_to_plain_ratio": v4b_cost["capture_to_plain_ratio"],
            "cuda_peak_allocated_gib": v4b_sampling["cuda_peak_allocated_gib"],
            "cuda_peak_reserved_gib": v4b_sampling["cuda_peak_reserved_gib"],
            "generation_tokens_per_second": v4b_sampling["tokens_per_second"],
        },
        "contrast_transfer": {
            "status": "pass" if transfer_pass else "fail",
            "minimum_mixed_test_prompts": 6,
            "minimum_test_risks": 2,
            "observed_mixed_test_prompts": v4b_yield["test"]["mixed"],
            "observed_test_risks": v4b_yield["test"]["risks"],
        },
        "inference": (
            "The frozen v4 prompt population does not transfer as a contrastive 4B benchmark; "
            "the preregistered residual-added-value comparison is therefore unidentified at 4B."
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    states_4b = v4b_states["states"]
    lines = [
        "# v4/v4b Cross-Scale Confirmatory Report",
        "",
        "v4b reused the exact frozen 34-prompt v4 population, split, sampling parameters, "
        "validator, semantic landing, baselines, and success gate. Only the model and "
        "depth-normalized captured layers changed.",
        "",
        "## Frozen-Prompt Contrast Transfer",
        "",
        "| model | gate | train mixed | validation mixed | test mixed | prompt outcome states |",
        "|---|---|---:|---:|---:|---|",
        f"| `{v4_sampling['model_id']}` | **{v4['gate']['status'].upper()}** | "
        f"{fmt_yield(v4_yield['train'])} | {fmt_yield(v4_yield['validation'])} | "
        f"{fmt_yield(v4_yield['test'])} | 34 mixed |",
        f"| `{v4b_sampling['model_id']}` | **{v4b['gate']['status'].upper()}** | "
        f"{fmt_yield(v4b_yield['train'])} | {fmt_yield(v4b_yield['validation'])} | "
        f"{fmt_yield(v4b_yield['test'])} | {states_4b.get('always_commit', 0)} always commit; "
        f"{states_4b.get('always_rollback', 0)} always rollback; {states_4b.get('mixed', 0)} mixed |",
        "",
        "The 4B run fails the frozen contrast-transfer requirement (`1/9` mixed test prompts "
        "from one risk, versus the required `6/9` from at least two risks). It also has zero "
        "mixed training prompts, so residual, TF-IDF, and next-token classifiers cannot be fit "
        "under the preregistered within-prompt weighting rule.",
        "",
        "## Interpretation",
        "",
        "The v4b accessibility result is **inconclusive**, not negative. The experiment does "
        "not establish that residual probes lack added value at 4B. It establishes that a "
        "contrast-selected benchmark discovered on Qwen3-0.6B does not remain contrastive on "
        "Qwen3-4B. Replacing prompts after observing this collapse would change the estimand and "
        "is disallowed by the frozen protocol.",
        "",
        "This scale dependence is behaviorally directional: of 34 prompts, 19 become always "
        "commit, 13 become always rollback, and only 2 remain mixed. The result therefore rules "
        "out a naive same-prompt scale curve while leaving a separately preregistered, 4B-specific "
        "discovery cohort as a distinct future experiment.",
        "",
        "## Local Reproducibility and Cost",
        "",
        f"- Qwen3-4B FP16 six-layer capture peak: `{v4b_sampling['cuda_peak_allocated_gib']:.3f}` "
        f"GiB allocated / `{v4b_sampling['cuda_peak_reserved_gib']:.3f}` GiB reserved.",
        f"- Full sampling throughput: `{v4b_sampling['tokens_per_second']:.3f}` generated tokens/s.",
        f"- Capture/plain ratio: `{v4b_cost['capture_to_plain_ratio']:.3f}` "
        f"(95% CI `{v4b_cost['capture_to_plain_ratio_ci95'][0]:.3f}`-"
        f"`{v4b_cost['capture_to_plain_ratio_ci95'][1]:.3f}`; 18 paired runs).",
        "- Plain and capture outputs were identical in every paired cost run.",
        "",
        "See `results/PREREGISTERED_V4B_CROSS_SCALE_PROTOCOL.md` and "
        "`results/trajectory_v4b_confirmatory/Qwen__Qwen3-4B/V4B_CONFIRMATORY_RESULTS.md`.",
    ]
    report = "\n".join(lines) + "\n"
    args.out_md.write_text(report, encoding="utf-8")
    args.static_out_md.parent.mkdir(parents=True, exist_ok=True)
    args.static_out_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out_md}, {args.static_out_md}, and {args.out_json}")


if __name__ == "__main__":
    main()
