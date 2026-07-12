"""Create the final cross-stage v4d report and static-viewer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the completed v4d experiment.")
    parser.add_argument(
        "--stage1",
        type=Path,
        default=Path("results/trajectory_v4d_stage1/Qwen__Qwen3.5-4B/STAGE1_GATE.json"),
    )
    parser.add_argument(
        "--confirmatory-dir",
        type=Path,
        default=Path("results/trajectory_v4d_confirmatory/Qwen__Qwen3.5-4B"),
    )
    parser.add_argument(
        "--appendix", type=Path, default=Path("results/v4c_appendix_diagnostics.json")
    )
    parser.add_argument("--out-json", type=Path, default=Path("results/v4d_final_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/V4D_FINAL_REPORT.md"))
    parser.add_argument(
        "--static-out-md", type=Path, default=Path("space_static/v4d_final_report.md")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    stage1 = read_json(args.stage1)
    appendix = read_json(args.appendix)
    analysis = read_json(args.confirmatory_dir / "v4d_analysis.json")
    identity = read_json(args.confirmatory_dir / "v4d_prelanding_identity.json")
    cost = read_json(args.confirmatory_dir / "monitoring_cost_benchmark.json")
    judge = read_json(args.confirmatory_dir / "prefix_judge_summary.json")
    q4 = next(
        row for row in appendix["conditions"] if row["condition_id"] == "qwen35_4b_t0p8"
    )
    gate = stage1["gate"]
    if gate["status"] != "pass" or analysis["gate"]["status"] != "fail":
        raise ValueError("Unexpected v4d stage status")
    if identity["gate_changed"] or identity["new_checkpoint_sampled"]:
        raise ValueError("The v4d post-hoc diagnostic changed the frozen experiment")

    primary_rows = [
        row for row in analysis["checkpoint_results"] if row["checkpoint"] in [2, 4, 6, 8, 10, 12]
    ]
    payload = {
        "analysis_status": "completed_final_v4d",
        "stage1": {
            "status": gate["status"],
            "hf_fp16_eligible": gate["raw_eligible_count"],
            "hf_fp16_mixed": stage1["outcome_states"]["mixed"],
            "hf_fp16_exact_candidates": stage1["candidate_choice"]["exact_candidate_count"],
            "hf_fp16_ab_switch_prompts": stage1["candidate_choice"][
                "both_exact_candidates_prompt_count"
            ],
            "q4_eligible": q4["eligible_prompt_count"],
            "q4_mixed": q4["states"]["mixed"],
            "q4_exact_candidates": q4["candidate_choice"]["exact_candidate_count"],
            "q4_ab_switch_prompts": q4["candidate_choice"][
                "both_exact_candidates_prompt_count"
            ],
            "selected_prompts": gate["selected_prompt_count"],
            "selected_families": gate["selected_template_family_count"],
            "split_counts": gate["split_counts"],
        },
        "confirmatory": {
            "status": analysis["gate"]["status"],
            "trajectory_count": analysis["trajectory_count"],
            "prompt_count": analysis["prompt_count"],
            "yield": analysis["confirmatory_yield"],
            "winning_checkpoints": analysis["gate"]["winning_checkpoints"],
            "primary_curve": [
                {
                    "checkpoint": row["checkpoint"],
                    "evaluable_prompts": row["evaluable_test_prompts"],
                    "evaluable_risks": row["evaluable_test_risks"],
                    "residual_auc": row["metrics"]["residual_layer_21"]["auc"],
                    "tfidf_auc": row["metrics"]["visible_prefix_tfidf"]["auc"],
                    "next_token_auc": row["metrics"]["next_token_stats"]["auc"],
                    "judge_auc": row["metrics"]["prefix_model_judge"]["auc"],
                    "residual_delta": row["primary_comparison"]["delta"],
                    "residual_delta_ci95": row["primary_comparison"]["ci95"],
                }
                for row in primary_rows
            ],
        },
        "mechanism": {
            "cross_label_first_difference": identity["cross_label_first_difference"],
            "cross_label_first_difference_by_risk": identity[
                "cross_label_first_difference_by_risk"
            ],
            "prelanding_lead_tokens": identity["prelanding_lead_tokens"],
        },
        "cost": {
            "capture_to_plain_ratio": cost["capture_to_plain_ratio"],
            "capture_to_plain_ratio_ci95": cost["capture_to_plain_ratio_ci95"],
            "paired_runs": cost["paired_runs"],
            "judge_milliseconds_per_unique_prefix": judge[
                "milliseconds_per_unique_prefix"
            ],
            "judge_unique_prefixes": judge["unique_rendered_prefixes"],
        },
        "stopping": {
            "experimental_development_complete": True,
            "derive_v4e": False,
            "scale_curve_deferred": True,
        },
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v4d Final Qwen3.5-4B Report",
        "",
        "v4d is complete. Stage 1 passed the frozen benchmark-feasibility gate, but the "
        "confirmatory residual-accessibility gate **FAILS** with no winning checkpoint.",
        "",
        "## Stage 1: Deployment Transfer",
        "",
        "| deployment | eligible | mixed | exact candidate outputs | A/B-switch prompts |",
        "|---|---:|---:|---:|---:|",
        f"| Ollama Q4_K_M | {q4['eligible_prompt_count']}/64 | {q4['states']['mixed']}/64 | "
        f"{q4['candidate_choice']['exact_candidate_count']}/1024 | "
        f"{q4['candidate_choice']['both_exact_candidates_prompt_count']}/64 |",
        f"| Transformers FP16 | {gate['raw_eligible_count']}/64 | "
        f"{stage1['outcome_states']['mixed']}/64 | "
        f"{stage1['candidate_choice']['exact_candidate_count']}/1024 | "
        f"{stage1['candidate_choice']['both_exact_candidates_prompt_count']}/64 |",
        "",
        "The FP16 run passes every pre-registered trigger and freezes 33 prompts from 22 "
        "template families into a 17/8/8 train/validation/test split. Benchmark viability "
        "therefore survives the backend and precision change. This does not identify which "
        "model or deployment component produces the stochasticity.",
        "",
        "## Confirmatory Result",
        "",
        "All 17 train, 8 validation, and 8 test prompts remain mixed under 32 fresh seeds, "
        "so the result is conclusive rather than a yield failure.",
        "",
        "| checkpoint | prompts | risks | residual L21 | TF-IDF | next-token | judge | delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_rows:
        metrics = row["metrics"]
        lines.append(
            f"| {row['checkpoint']} | {row['evaluable_test_prompts']} | "
            f"{row['evaluable_test_risks']} | {metrics['residual_layer_21']['auc']:.3f} | "
            f"{metrics['visible_prefix_tfidf']['auc']:.3f} | "
            f"{metrics['next_token_stats']['auc']:.3f} | "
            f"{metrics['prefix_model_judge']['auc']:.3f} | "
            f"{row['primary_comparison']['delta']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Every primary method is exactly at chance. No checkpoint approaches the frozen "
            "`+0.03` residual margin, and no secondary captured layer changes that result.",
            "",
            "## Why the Curve Is Exactly 0.5",
            "",
            "The post-hoc identity diagnostic changes no gate and samples no new checkpoint. "
            "At checkpoints 0-16, every evaluable prompt has one visible prefix and exactly "
            "zero within-prompt residual span at all six captured layers. Across test pairs, "
            "1,014 early-spoiler/fake-commit pairs first diverge exactly at policy landing. "
            "The 714 hidden-fields pairs diverge two tokens before landing, but that window "
            "falls between frozen checkpoints 16 and 24; at checkpoint 24 only two prompts "
            "from one risk remain evaluable.",
            "",
            "Thus v4d finds abundant outcome stochasticity without an operationally usable "
            "signal at the frozen early checkpoints. It does not rule out a narrow two-token "
            "hidden-fields signal that this protocol was not powered to evaluate.",
            "",
            "## Cost and Stop",
            "",
            f"Six-layer capture costs `{cost['capture_to_plain_ratio']:.3f}x` plain generation "
            f"(95% paired CI `{cost['capture_to_plain_ratio_ci95'][0]:.3f}`-"
            f"`{cost['capture_to_plain_ratio_ci95'][1]:.3f}`, {cost['paired_runs']} pairs). "
            f"The same-model judge costs `{judge['milliseconds_per_unique_prefix']:.1f}` "
            "ms per unique prefix and is also at chance.",
            "",
            "Per the frozen stopping boundary, v4d ends experimental development. No v4e or "
            "pre-submission scale curve is derived from this result.",
        ]
    )
    report = "\n".join(lines) + "\n"
    args.out_md.write_text(report, encoding="utf-8")
    args.static_out_md.write_text(report, encoding="utf-8")
    print(f"Wrote {args.out_json}, {args.out_md}, and {args.static_out_md}")


if __name__ == "__main__":
    main()
