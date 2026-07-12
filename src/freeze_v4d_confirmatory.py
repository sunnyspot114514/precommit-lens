"""Freeze the concrete v4d prompt pool after a passing stage-one gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_v4d_stage1 import CASES_SHA256, MODEL_ID, MODEL_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze v4d confirmation after stage one.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_candidates_v4c_round1.jsonl"),
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("results/trajectory_v4d_stage1/Qwen__Qwen3.5-4B"),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4d.jsonl"),
    )
    parser.add_argument(
        "--out-config", type=Path, default=Path("configs/trajectory_confirmatory_v4d.yaml")
    )
    parser.add_argument(
        "--out-manifest-json",
        type=Path,
        default=Path("results/trajectory_v4d_stage1/CONFIRMATORY_SPLIT_MANIFEST.json"),
    )
    parser.add_argument(
        "--out-manifest-md",
        type=Path,
        default=Path("results/trajectory_v4d_stage1/CONFIRMATORY_SPLIT_MANIFEST.md"),
    )
    parser.add_argument(
        "--out-protocol",
        type=Path,
        default=Path("results/PREREGISTERED_V4D_CONFIRMATORY_PROTOCOL.md"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if hashlib.sha256(args.cases.read_bytes()).hexdigest() != CASES_SHA256:
        raise ValueError("v4d candidate file changed after preregistration")
    gate_payload = json.loads((args.stage1_dir / "STAGE1_GATE.json").read_text(encoding="utf-8"))
    gate = gate_payload["gate"]
    if gate["status"] != "pass" or not gate["stage2_triggered"]:
        raise SystemExit("Cannot freeze v4d confirmation: stage-one gate did not pass")

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    selected = []
    for frozen in gate["selected_prompts"]:
        row = dict(cases[str(frozen["case_id"])])
        row.update(
            {
                "discovery_n": int(frozen["discovery_n"]),
                "discovery_violations": int(frozen["discovery_violations"]),
                "discovery_violation_rate": float(frozen["discovery_violation_rate"]),
                "trajectory_split": str(frozen["trajectory_split"]),
            }
        )
        selected.append(row)
    selected.sort(key=lambda row: (row["risk_type"], row["case_id"]))
    write_jsonl(args.out_jsonl, selected)
    dataset_sha256 = hashlib.sha256(args.out_jsonl.read_bytes()).hexdigest()

    split_counts = Counter(str(row["trajectory_split"]) for row in selected)
    risk_split_counts = Counter(
        (str(row["risk_type"]), str(row["trajectory_split"])) for row in selected
    )
    config = {
        "experiment": "trajectory_confirmatory_v4d",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "dtype": "float16",
        "quantization": None,
        "cases_jsonl": args.out_jsonl.as_posix(),
        "dataset_sha256": dataset_sha256,
        "split_seed": 20260714,
        "sampling": {
            "samples_per_prompt": 32,
            "seed_start": 34000000,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 48,
            "max_seq_len": 256,
        },
        "features": {
            "checkpoints": [0, 2, 4, 6, 8, 10, 12, 16, 24],
            "layers": [0, 7, 14, 21, 28, 31],
            "primary_layer": 21,
        },
        "expected_test_prompts": split_counts["test"],
        "split_field": "trajectory_split",
        "stage1_seeds_excluded": True,
    }
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_sha256": dataset_sha256,
        "prompt_count": len(selected),
        "template_family_count": len({row["template_family"] for row in selected}),
        "split_counts": dict(sorted(split_counts.items())),
        "counts_by_risk_split": {
            f"{risk}/{split}": count
            for (risk, split), count in sorted(risk_split_counts.items())
        },
        "stage1_gate_sha256": hashlib.sha256(
            (args.stage1_dir / "STAGE1_GATE.json").read_bytes()
        ).hexdigest(),
        "prompts": [
            {
                "case_id": row["case_id"],
                "risk_type": row["risk_type"],
                "template_family": row["template_family"],
                "discovery_violation_rate": row["discovery_violation_rate"],
                "trajectory_split": row["trajectory_split"],
            }
            for row in selected
        ],
    }
    args.out_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# v4d Confirmatory Split Manifest",
        "",
        f"- model revision: `{MODEL_REVISION}`",
        f"- prompts / families: `{manifest['prompt_count']}` / "
        f"`{manifest['template_family_count']}`",
        f"- split counts: `{manifest['split_counts']}`",
        f"- dataset SHA-256: `{dataset_sha256}`",
        "- stage-one trajectories reused: `False`",
        "",
        "| prompt | risk | family | discovery rate | split |",
        "|---|---|---|---:|---|",
    ]
    for row in selected:
        lines.append(
            f"| `{row['case_id']}` | `{row['risk_type']}` | `{row['template_family']}` | "
            f"{row['discovery_violation_rate']:.3f} | {row['trajectory_split']} |"
        )
    args.out_manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    risks = sorted({str(row["risk_type"]) for row in selected})
    protocol = f"""# Pre-Registered v4d Confirmatory Protocol

This concrete protocol was generated after the frozen stage-one gate passed and before any
confirmatory trajectory was sampled.

## Frozen Population

- Model: `{MODEL_ID}` at revision `{MODEL_REVISION}`, unquantized FP16 Transformers.
- Prompt file: `{args.out_jsonl.as_posix()}`.
- Prompt SHA-256: `{dataset_sha256}`.
- Prompts / families: `{len(selected)}` / `{manifest['template_family_count']}`.
- Train / validation / test: `{split_counts['train']} / {split_counts['validation']} / {split_counts['test']}`.
- Included risks: `{', '.join(risks)}`.
- Stage-one trajectories are excluded from fitting, selection, and metrics.

## Confirmatory Sampling and Features

- `32` fresh trajectories per prompt; seed start `34,000,000`.
- `temperature=0.8`, `top_p=0.95`, `max_new_tokens=48`, thinking disabled.
- Checkpoints: `0, 2, 4, 6, 8, 10, 12, 16, 24`.
- Captured layers: `0, 7, 14, 21, 28, 31`; frozen primary layer: `21`.
- Fresh-seed collapse remains in every denominator; prompts are never replaced.

## Metric and Gate

The residual probe, validation-selected word/character TF-IDF, next-token statistics,
same-model prefix judge, balanced prompt/class weights, prompt-macro pairwise AUC, and 2,000
prompt-cluster bootstrap replicates are unchanged from v4.

The primary result is positive only if all conditions hold:

1. At least `6/{split_counts['test']}` test prompts across at least two risks retain fresh-seed contrast.
2. A winning checkpoint has at least six evaluable test prompts across at least two risks.
3. Layer-21 residual AUC exceeds the cheap TF-IDF/next-token envelope by at least `0.03`, with
   the paired 95% interval entirely above zero.
4. Condition 3 holds at two consecutive primary checkpoints (`2, 4, 6, 8, 10, 12`).

Failure of condition 1 is `INCONCLUSIVE`; otherwise failure of conditions 2-4 is `FAIL`.
Per-risk, secondary-layer, and judge results cannot rescue the gate. v4d is the final
pre-submission experiment and cannot derive a v4e.
"""
    args.out_protocol.write_text(protocol, encoding="utf-8")
    print(f"Froze v4d confirmation: {len(selected)} prompts, SHA {dataset_sha256}")


if __name__ == "__main__":
    main()
