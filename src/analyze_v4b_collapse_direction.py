"""Post-hoc directional analysis of prompt collapse from v4 to v4b."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20260712


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze v4b collapse direction post hoc.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4.jsonl"),
    )
    parser.add_argument(
        "--v4-trajectories",
        type=Path,
        default=Path(
            "results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B/trajectories.jsonl"
        ),
    )
    parser.add_argument(
        "--v4b-trajectories",
        type=Path,
        default=Path(
            "results/trajectory_v4b_confirmatory/Qwen__Qwen3-4B/trajectories.jsonl"
        ),
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("results/v4b_collapse_direction_posthoc.json")
    )
    parser.add_argument(
        "--out-md", type=Path, default=Path("results/V4B_COLLAPSE_DIRECTION_POSTHOC.md")
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def violation_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row["policy_validator"]["decision"] == "rollback")
    return {case_id: float(np.mean(values)) for case_id, values in grouped.items()}


def collapse_state(rate: float) -> str:
    if rate == 0.0:
        return "always_commit"
    if rate == 1.0:
        return "always_rollback"
    return "mixed"


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = (positive[:, None] > negative[None, :]).astype(np.float64)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def bootstrap_auc(
    positive: np.ndarray, negative: np.ndarray, *, samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        pos = positive[rng.integers(0, len(positive), len(positive))]
        neg = negative[rng.integers(0, len(negative), len(negative))]
        values[index] = pairwise_auc(pos, neg)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def chi_square(table: np.ndarray) -> float:
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True)
    expected = expected / table.sum()
    return float(np.sum((table - expected) ** 2 / expected))


def permutation_association(
    risks: np.ndarray,
    labels: np.ndarray,
    risk_names: list[str],
) -> dict[str, float]:
    observed_table = np.asarray(
        [[np.sum((risks == risk) & (labels == value)) for value in (0, 1)] for risk in risk_names]
    )
    observed = chi_square(observed_table)
    risk_sizes = observed_table.sum(axis=1).astype(int)
    rollback_total = int(labels.sum())
    denominator = math.comb(len(labels), rollback_total)
    exact_p = 0.0
    for first in range(risk_sizes[0] + 1):
        for second in range(risk_sizes[1] + 1):
            third = rollback_total - first - second
            if third < 0 or third > risk_sizes[2]:
                continue
            rollbacks = np.asarray([first, second, third])
            table = np.column_stack([risk_sizes - rollbacks, rollbacks])
            if chi_square(table) >= observed - 1e-12:
                exact_p += math.prod(
                    math.comb(int(size), int(count))
                    for size, count in zip(risk_sizes, rollbacks, strict=True)
                ) / denominator
    return {
        "chi_square": observed,
        "cramers_v": math.sqrt(observed / len(labels)),
        "exact_permutation_p": exact_p,
    }


def main() -> None:
    args = parse_args()
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    v4_rows = read_jsonl(args.v4_trajectories)
    v4b_rows = read_jsonl(args.v4b_trajectories)
    v4_rates = violation_rates(v4_rows)
    v4b_rates = violation_rates(v4b_rows)
    if set(v4_rates) != set(cases) or set(v4b_rates) != set(cases):
        raise ValueError("Cases and trajectory prompt sets differ")

    prompt_rows = []
    for case_id in sorted(cases):
        case = cases[case_id]
        prompt_rows.append(
            {
                "case_id": case_id,
                "risk_type": str(case["risk_type"]),
                "trajectory_split": str(case["trajectory_split"]),
                "qwen3_0_6b_violation_rate": v4_rates[case_id],
                "qwen3_4b_violation_rate": v4b_rates[case_id],
                "qwen3_4b_state": collapse_state(v4b_rates[case_id]),
            }
        )

    states = ("always_commit", "always_rollback", "mixed")
    risks = sorted({row["risk_type"] for row in prompt_rows})
    by_risk = {}
    for risk in risks:
        selected = [row for row in prompt_rows if row["risk_type"] == risk]
        by_risk[risk] = {
            "prompt_count": len(selected),
            "qwen3_0_6b_violation_rate": float(
                np.mean([row["qwen3_0_6b_violation_rate"] for row in selected])
            ),
            "qwen3_4b_violation_rate": float(
                np.mean([row["qwen3_4b_violation_rate"] for row in selected])
            ),
            "states": {state: sum(row["qwen3_4b_state"] == state for row in selected) for state in states},
        }

    collapsed = [row for row in prompt_rows if row["qwen3_4b_state"] != "mixed"]
    rollback_rates = np.asarray(
        [
            row["qwen3_0_6b_violation_rate"]
            for row in collapsed
            if row["qwen3_4b_state"] == "always_rollback"
        ]
    )
    commit_rates = np.asarray(
        [
            row["qwen3_0_6b_violation_rate"]
            for row in collapsed
            if row["qwen3_4b_state"] == "always_commit"
        ]
    )
    rate_auc = pairwise_auc(rollback_rates, commit_rates)
    rate_auc_ci = bootstrap_auc(
        rollback_rates, commit_rates, samples=args.bootstrap_samples, seed=SEED
    )
    risk_array = np.asarray([row["risk_type"] for row in collapsed])
    direction_array = np.asarray(
        [int(row["qwen3_4b_state"] == "always_rollback") for row in collapsed]
    )
    association = permutation_association(
        risk_array,
        direction_array,
        risks,
    )

    state_summaries = {}
    for state in states:
        values = [
            row["qwen3_0_6b_violation_rate"]
            for row in prompt_rows
            if row["qwen3_4b_state"] == state
        ]
        state_summaries[state] = {
            "count": len(values),
            "qwen3_0_6b_rate_mean": float(np.mean(values)),
            "qwen3_0_6b_rate_median": float(np.median(values)),
            "qwen3_0_6b_rate_range": [float(min(values)), float(max(values))],
        }

    payload = {
        "analysis_status": "post_hoc_diagnostic",
        "claim_boundary": (
            "This analysis describes direction within the frozen 0.6B-selected prompt corpus. "
            "It is not a preregistered scale-law test."
        ),
        "prompt_count": len(prompt_rows),
        "collapsed_prompt_count": len(collapsed),
        "mixed_prompt_count": len(prompt_rows) - len(collapsed),
        "by_risk": by_risk,
        "state_summaries": state_summaries,
        "qwen3_0_6b_rate_predicts_4b_direction": {
            "auc": rate_auc,
            "bootstrap_ci95": rate_auc_ci,
            "bootstrap_samples": args.bootstrap_samples,
        },
        "risk_family_direction_association": {
            **association,
            "collapsed_prompts_only": True,
        },
        "prompts": prompt_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v4b Collapse-Direction Post-Hoc Diagnostic",
        "",
        "**Status: post-hoc diagnostic, not pre-registered confirmatory evidence.**",
        "",
        "This analysis asks what determines the direction of the `32/34` prompt collapses in "
        "the frozen Qwen3-0.6B-selected corpus. It does not alter the v4b `INCONCLUSIVE` gate.",
        "",
        "## Risk-Family Direction",
        "",
        "| risk | prompts | 0.6B violation rate | 4B violation rate | always commit | always rollback | mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for risk in risks:
        row = by_risk[risk]
        lines.append(
            f"| `{risk}` | {row['prompt_count']} | {row['qwen3_0_6b_violation_rate']:.3f} | "
            f"{row['qwen3_4b_violation_rate']:.3f} | {row['states']['always_commit']} | "
            f"{row['states']['always_rollback']} | {row['states']['mixed']} |"
        )
    lines.extend(
        [
            "",
            f"Among collapsed prompts, risk family and 4B direction have Cramer's V "
            f"`{association['cramers_v']:.3f}`; the fixed-margin permutation diagnostic is "
            f"exact `p={association['exact_permutation_p']:.8f}`. "
            "Because this test was chosen after inspecting collapse counts, the p-value is descriptive.",
            "",
            "## Does the 0.6B Violation Rate Predict Direction?",
            "",
            f"The 0.6B prompt violation rate predicts 4B always-rollback versus always-commit "
            f"with AUC `{rate_auc:.3f}` (stratified bootstrap 95% CI "
            f"`[{rate_auc_ci[0]:.3f}, {rate_auc_ci[1]:.3f}]`). The corresponding 0.6B mean rates "
            f"are `{state_summaries['always_rollback']['qwen3_0_6b_rate_mean']:.3f}` for prompts "
            f"that become always rollback and `{state_summaries['always_commit']['qwen3_0_6b_rate_mean']:.3f}` "
            "for prompts that become always commit. This is weak directional prediction.",
            "",
            "## Interpretation Boundary",
            "",
            "Within this selected corpus, Qwen3-4B reduces sampling variability but shifts policy "
            "outcomes in risk-specific, bidirectional ways: early-spoiler prompts mostly stabilize "
            "toward commit, while hidden-field and schema-bypass prompts mostly stabilize toward "
            "rollback. This supports a benchmark-validity warning, not a general claim that scale "
            "causes either safety or violation determinism.",
            "",
            "## Per-Prompt Direction Table",
            "",
            "| case | risk | split | 0.6B rate | 4B rate | 4B state |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in prompt_rows:
        lines.append(
            f"| `{row['case_id']}` | `{row['risk_type']}` | {row['trajectory_split']} | "
            f"{row['qwen3_0_6b_violation_rate']:.3f} | {row['qwen3_4b_violation_rate']:.3f} | "
            f"`{row['qwen3_4b_state']}` |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
