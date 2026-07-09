"""Summarize leakage-controlled evaluation by risk and condition."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_probe_auc import average_precision, fpr_at_tpr, roc_auc  # noqa: E402


METHODS = [
    "keyword_target_present",
    "keyword_hit_count",
    "logit_lens",
    "dense_jlens",
    "jvp_lens",
    "linear_probe",
]

LABELS = ["expected_rollback", "generated_rollback"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize leakage-controlled AUC results.")
    parser.add_argument("--case-scores", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def label_value(row: dict[str, Any], label: str) -> int:
    if label == "expected_rollback":
        return int(row.get("expected_validator_decision") == "rollback")
    if label == "semantic_risk":
        return int(bool(row.get("semantic_risk_label")))
    if label == "generated_rollback":
        return int(row.get("generated_validator", {}).get("decision") == "rollback")
    raise KeyError(label)


def score_value(row: dict[str, Any], method: str) -> float:
    value = row.get("scores", {}).get(method, {}).get("score")
    return float(value) if value is not None else float("nan")


def summarize_subset(rows: list[dict[str, Any]], method: str, label: str) -> dict[str, Any]:
    scores = [score_value(row, method) for row in rows]
    labels = [label_value(row, label) for row in rows]
    return {
        "n": len(rows),
        "positive": int(sum(labels)),
        "auc": roc_auc(scores, labels),
        "auprc": average_precision(scores, labels),
        "fpr_at_90_tpr": fpr_at_tpr(scores, labels, 0.9),
    }


def finite_mean(values: list[float]) -> float | None:
    filtered = [v for v in values if math.isfinite(v)]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["risk_type"], row["condition"])].append(row)
    out = []
    for (risk, condition), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "risk_type": risk,
            "condition": condition,
            "n": len(group),
            "generated_rollback_rate": sum(label_value(row, "generated_rollback") for row in group) / len(group),
            "target_present_rate": sum(int(bool(row.get("target_present"))) for row in group) / len(group),
        }
        for method in METHODS:
            item[f"{method}_mean"] = finite_mean([score_value(row, method) for row in group])
        out.append(item)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Leakage-Controlled Falsification Summary",
        "",
        "## Per-Risk AUC",
        "",
        "| risk | method | label | n | pos | AUC | AUPRC | FPR@90%TPR |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_risk"]:
        lines.append(
            f"| {row['risk_type']} | `{row['method']}` | {row['label']} | "
            f"{row['n']} | {row['positive']} | {fmt(row['auc'])} | "
            f"{fmt(row['auprc'])} | {fmt(row['fpr_at_90_tpr'])} |"
        )

    lines.extend(
        [
            "",
            "## Condition Means",
            "",
            "| risk | condition | n | gen rollback | target present | keyword hits | logit | dense | JVP | linear |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["conditions"]:
        lines.append(
            f"| {row['risk_type']} | `{row['condition']}` | {row['n']} | "
            f"{fmt(row['generated_rollback_rate'])} | {fmt(row['target_present_rate'])} | "
            f"{fmt(row['keyword_hit_count_mean'])} | {fmt(row['logit_lens_mean'])} | "
            f"{fmt(row['dense_jlens_mean'])} | {fmt(row['jvp_lens_mean'])} | "
            f"{fmt(row['linear_probe_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Checks",
            "",
        ]
    )
    for note in payload["interpretation_checks"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_interpretation_checks(per_risk: list[dict[str, Any]]) -> list[str]:
    checks = []
    by_method_label: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in per_risk:
        by_method_label[(row["risk_type"], row["method"], row["label"])] = row
    risks = sorted({row["risk_type"] for row in per_risk})
    for risk in risks:
        dense = by_method_label.get((risk, "dense_jlens", "expected_rollback"), {}).get("auc")
        keyword = by_method_label.get((risk, "keyword_hit_count", "expected_rollback"), {}).get("auc")
        linear = by_method_label.get((risk, "linear_probe", "expected_rollback"), {}).get("auc")
        if dense is not None and keyword is not None and dense <= keyword:
            checks.append(
                f"{risk}: dense J-lens does not beat the keyword-hit baseline on expected rollback "
                f"(dense {dense:.3f}, keyword {keyword:.3f})."
            )
        if linear is not None and dense is not None and linear >= 0.9 and linear > dense:
            checks.append(
                f"{risk}: linear probe dominates dense J-lens on expected rollback "
                f"(linear {linear:.3f}, dense {dense:.3f})."
            )
    if not checks:
        checks.append("No automatic falsification check fired.")
    return checks


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.case_scores)

    per_risk = []
    risks = sorted({row["risk_type"] for row in rows})
    for risk in risks:
        subset = [row for row in rows if row["risk_type"] == risk]
        for label in LABELS:
            for method in METHODS:
                if method not in subset[0].get("scores", {}):
                    continue
                per_risk.append(
                    {
                        "risk_type": risk,
                        "method": method,
                        "label": label,
                        **summarize_subset(subset, method, label),
                    }
                )

    payload = {
        "source": str(args.case_scores),
        "per_risk": per_risk,
        "conditions": condition_summary(rows),
    }
    payload["interpretation_checks"] = build_interpretation_checks(per_risk)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    write_md(args.out_md, payload)
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
