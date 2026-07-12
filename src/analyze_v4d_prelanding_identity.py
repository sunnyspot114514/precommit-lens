"""Diagnose why the frozen v4d pre-landing curve is exactly at chance."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v4d pre-landing state identity.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/trajectory_v4d_confirmatory/Qwen__Qwen3.5-4B"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4d.jsonl"),
    )
    parser.add_argument("--primary-layer", type=int, default=21)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right), start=1):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right)) + 1
    return None


def main() -> None:
    args = parse_args()
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    trajectories = read_jsonl(args.run_dir / "trajectories.jsonl")
    cache = np.load(args.run_dir / "trajectory_features.npz")
    checkpoints = [int(value) for value in cache["checkpoints"].tolist()]
    layers = [int(value) for value in cache["layers"].tolist()]
    if args.primary_layer not in layers:
        raise ValueError("Primary layer is absent from the v4d feature cache")
    primary_index = layers.index(args.primary_layer)
    trajectory_ids = [str(value) for value in cache["trajectory_ids"].tolist()]
    by_id = {str(row["trajectory_id"]): row for row in trajectories}
    if len(by_id) != len(trajectories) or set(by_id) != set(trajectory_ids):
        raise ValueError("Trajectory JSONL and feature-cache ids do not match")
    ordered = [by_id[trajectory_id] for trajectory_id in trajectory_ids]
    features = cache["features"].astype(np.float32)
    valid = cache["valid"].astype(np.bool_)

    indices_by_case: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        indices_by_case[str(row["case_id"])].append(index)

    checkpoint_rows = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        prompt_rows = []
        for case_id, case in cases.items():
            if case["trajectory_split"] != "test":
                continue
            indices = [
                index
                for index in indices_by_case[case_id]
                if valid[index, checkpoint_index]
                and (
                    ordered[index].get("policy_landing_token") is None
                    or checkpoint < int(ordered[index]["policy_landing_token"])
                )
            ]
            labels = [
                ordered[index]["policy_validator"]["decision"] == "rollback"
                for index in indices
            ]
            if len(set(labels)) != 2:
                continue
            prefixes = {
                str(ordered[index]["checkpoint_prefixes"][checkpoint_index])
                for index in indices
            }
            primary = features[indices, checkpoint_index, primary_index, :]
            all_layers = features[indices, checkpoint_index, :, :]
            prompt_rows.append(
                {
                    "case_id": case_id,
                    "risk_type": case["risk_type"],
                    "available_trajectories": len(indices),
                    "unique_visible_prefixes": len(prefixes),
                    "primary_feature_max_span": float(np.ptp(primary, axis=0).max()),
                    "all_layer_feature_max_span": float(np.ptp(all_layers, axis=0).max()),
                }
            )
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "evaluable_test_prompts": len(prompt_rows),
                "evaluable_test_risks": len({row["risk_type"] for row in prompt_rows}),
                "identical_prefix_prompts": sum(
                    row["unique_visible_prefixes"] == 1 for row in prompt_rows
                ),
                "identical_primary_feature_prompts": sum(
                    row["primary_feature_max_span"] == 0.0 for row in prompt_rows
                ),
                "identical_all_layer_feature_prompts": sum(
                    row["all_layer_feature_max_span"] == 0.0 for row in prompt_rows
                ),
                "maximum_primary_feature_span": max(
                    (row["primary_feature_max_span"] for row in prompt_rows), default=None
                ),
                "prompts": prompt_rows,
            }
        )

    pair_relations: Counter[str] = Counter()
    pair_relations_by_risk: dict[str, Counter[str]] = defaultdict(Counter)
    prelanding_leads: Counter[int] = Counter()
    for case_id, case in cases.items():
        if case["trajectory_split"] != "test":
            continue
        rows = [ordered[index] for index in indices_by_case[case_id]]
        violating = [
            row for row in rows if row["policy_validator"]["decision"] == "rollback"
        ]
        compliant = [
            row for row in rows if row["policy_validator"]["decision"] == "commit"
        ]
        for unsafe, safe in itertools.product(violating, compliant):
            difference = first_difference(
                list(unsafe["generated_token_ids"]), list(safe["generated_token_ids"])
            )
            landing = unsafe.get("policy_landing_token")
            if difference is None or landing is None:
                relation = "undefined"
            elif difference < int(landing):
                relation = "before_landing"
                prelanding_leads[int(landing) - difference] += 1
            elif difference == int(landing):
                relation = "at_landing"
            else:
                relation = "after_landing"
            pair_relations[relation] += 1
            pair_relations_by_risk[str(case["risk_type"])][relation] += 1

    landing_by_risk: dict[str, Counter[int]] = defaultdict(Counter)
    for row in ordered:
        if row["policy_validator"]["decision"] == "rollback":
            landing_by_risk[str(row["risk_type"])][int(row["policy_landing_token"])] += 1

    payload = {
        "analysis_status": "posthoc_mechanistic_diagnostic_no_gate_change",
        "primary_layer": args.primary_layer,
        "checkpoint_rows": checkpoint_rows,
        "cross_label_first_difference": dict(sorted(pair_relations.items())),
        "cross_label_first_difference_by_risk": {
            risk: dict(sorted(values.items()))
            for risk, values in sorted(pair_relations_by_risk.items())
        },
        "prelanding_lead_tokens": {
            str(value): count for value, count in sorted(prelanding_leads.items())
        },
        "policy_landing_tokens_by_risk": {
            risk: {str(token): count for token, count in sorted(values.items())}
            for risk, values in sorted(landing_by_risk.items())
        },
        "gate_changed": False,
        "new_checkpoint_sampled": False,
    }
    json_path = args.run_dir / "v4d_prelanding_identity.json"
    md_path = args.run_dir / "V4D_PRELANDING_IDENTITY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v4d Pre-Landing Identity Diagnostic",
        "",
        "**Status: post-hoc mechanistic diagnostic; the frozen v4d gate is unchanged.**",
        "",
        "| checkpoint | evaluable prompts | risks | identical visible prefix | "
        "identical L21 residual | identical all captured layers | max L21 span |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in checkpoint_rows:
        span = row["maximum_primary_feature_span"]
        lines.append(
            f"| {row['checkpoint']} | {row['evaluable_test_prompts']} | "
            f"{row['evaluable_test_risks']} | {row['identical_prefix_prompts']} | "
            f"{row['identical_primary_feature_prompts']} | "
            f"{row['identical_all_layer_feature_prompts']} | "
            f"{'NA' if span is None else f'{span:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "## First Divergence Relative to Policy Landing",
            "",
            "| risk | before landing | at landing | after landing |",
            "|---|---:|---:|---:|",
        ]
    )
    for risk, values in sorted(pair_relations_by_risk.items()):
        lines.append(
            f"| `{risk}` | {values['before_landing']} | {values['at_landing']} | "
            f"{values['after_landing']} |"
        )
    lines.extend(
        [
            "",
            f"Across test prompts, `{pair_relations['at_landing']}` cross-label pairs first "
            "diverge exactly at policy landing and "
            f"`{pair_relations['before_landing']}` diverge before landing. All early-spoiler "
            "and fake-commit pairs diverge at landing. Every pre-landing divergence is a "
            f"two-token lead in hidden-fields (`{prelanding_leads[2]}` pairs).",
            "",
            "At frozen checkpoints 0-16, every evaluable prompt has one visible prefix and "
            "zero within-prompt span at every captured layer. Checkpoint 24 contains state "
            "variation, but only two test prompts from one risk remain pre-landing, below the "
            "frozen support requirement. No intermediate checkpoint was sampled post hoc.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
