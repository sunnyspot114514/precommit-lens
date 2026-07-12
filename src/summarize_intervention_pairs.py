"""Add paired uncertainty estimates to an existing intervention sweep."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired intervention outcomes.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rollback(row: dict[str, Any], mode: str) -> int:
    return int(row[f"{mode}_validator"]["decision"] == "rollback")


def paired_difference(
    left: list[int],
    right: list[int],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        return {"difference": None, "ci95": [None, None], "mcnemar_p": None}
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(differences[rng.integers(0, len(differences), size=len(differences))]))
        for _ in range(samples)
    ]
    b = sum(int(l == 1 and r == 0) for l, r in zip(left, right))
    c = sum(int(l == 0 and r == 1) for l, r in zip(left, right))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, idx) for idx in range(min(b, c) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "difference": float(np.mean(differences)),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "mcnemar_p": float(p_value),
        "discordant_left_only": b,
        "discordant_right_only": c,
    }


def summarize(rows: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["risk_type"]), str(row["condition"]))].append(row)
    out = []
    for group_idx, ((risk, condition), group) in enumerate(sorted(grouped.items())):
        baseline = [rollback(row, "baseline") for row in group]
        suppress = [rollback(row, "suppress") for row in group]
        sham = [rollback(row, "sham") for row in group]
        out.append(
            {
                "risk_type": risk,
                "condition": condition,
                "n": len(group),
                "baseline_rate": float(np.mean(baseline)),
                "suppress_rate": float(np.mean(suppress)),
                "sham_rate": float(np.mean(sham)),
                "suppress_vs_baseline": paired_difference(suppress, baseline, samples, 41 + group_idx),
                "sham_vs_baseline": paired_difference(sham, baseline, samples, 71 + group_idx),
                "suppress_vs_sham": paired_difference(suppress, sham, samples, 101 + group_idx),
            }
        )
    return out


def fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}"


def ci_text(value: list[float | None]) -> str:
    return "-" if value[0] is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Paired Intervention Statistics",
        "",
        "All differences preserve the per-prompt baseline/suppress/sham pairing. "
        "The p-value is the exact two-sided McNemar test on discordant outcomes.",
        "",
        "| risk | condition | n | baseline | suppress | sham | suppress-sham | 95% CI | McNemar p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary"]:
        comparison = row["suppress_vs_sham"]
        lines.append(
            f"| {row['risk_type']} | `{row['condition']}` | {row['n']} | "
            f"{row['baseline_rate']:.3f} | {row['suppress_rate']:.3f} | {row['sham_rate']:.3f} | "
            f"{fmt(comparison['difference'])} | {ci_text(comparison['ci95'])} | "
            f"{fmt(comparison['mcnemar_p'])} |"
        )
    lines.extend(
        [
            "",
            "A suppression effect requires a negative suppress-minus-sham difference whose paired interval excludes zero.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = {
        "source": str(args.rows),
        "bootstrap_samples": args.bootstrap_samples,
        "summary": summarize(read_jsonl(args.rows), args.bootstrap_samples),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    write_markdown(args.out_md, payload)
    print(f"Saved paired intervention statistics to {args.out_md}")


if __name__ == "__main__":
    main()
