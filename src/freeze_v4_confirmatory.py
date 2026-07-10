"""Freeze the v4 eligible prompt pool and grouped confirmatory split."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


SPLITS = ("train", "validation", "test")
SPLIT_SEED = 20260710
ROUNDS = (
    (
        "round1",
        Path("data/prompt_sets/trajectory_candidates_v4.jsonl"),
        Path("results/trajectory_v4_discovery/Qwen__Qwen3-0.6B/sampling_summary.json"),
    ),
    (
        "round2",
        Path("data/prompt_sets/trajectory_candidates_v4_round2.jsonl"),
        Path("results/trajectory_v4_discovery_round2/Qwen__Qwen3-0.6B/sampling_summary.json"),
    ),
    (
        "round3",
        Path("data/prompt_sets/trajectory_candidates_v4_round3.jsonl"),
        Path("results/trajectory_v4_discovery_round3/Qwen__Qwen3-0.6B/sampling_summary.json"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze v4 confirmatory prompts and splits.")
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4.jsonl"),
    )
    parser.add_argument(
        "--out-config",
        type=Path,
        default=Path("configs/trajectory_confirmatory_v4.yaml"),
    )
    parser.add_argument(
        "--out-manifest-json",
        type=Path,
        default=Path("results/trajectory_v4_discovery/CONFIRMATORY_SPLIT_MANIFEST.json"),
    )
    parser.add_argument(
        "--out-manifest-md",
        type=Path,
        default=Path("results/trajectory_v4_discovery/CONFIRMATORY_SPLIT_MANIFEST.md"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_hash(value: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}:{value}".encode()).hexdigest()


def target_counts(total: int) -> dict[str, int]:
    validation = max(2, round(total * 0.25))
    test = max(2, round(total * 0.25))
    return {"train": total - validation - test, "validation": validation, "test": test}


def assign_risk_families(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["template_family"])].append(row)
    families = sorted(by_family, key=stable_hash)
    targets = target_counts(len(rows))
    best: tuple[float, str, tuple[str, ...]] | None = None
    for assignment in itertools.product(SPLITS, repeat=len(families)):
        counts = Counter()
        family_counts = Counter(assignment)
        for family, split in zip(families, assignment, strict=True):
            counts[split] += len(by_family[family])
        if any(counts[split] < 2 for split in SPLITS):
            continue
        prompt_loss = sum((counts[split] - targets[split]) ** 2 for split in SPLITS)
        family_target = len(families) / len(SPLITS)
        family_loss = sum((family_counts[split] - family_target) ** 2 for split in SPLITS)
        tie = stable_hash("|".join(f"{family}:{split}" for family, split in zip(families, assignment)))
        candidate = (prompt_loss + 0.01 * family_loss, tie, assignment)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError(f"No valid grouped split for {len(rows)} prompts across {len(families)} families")
    return dict(zip(families, best[2], strict=True))


def collect_eligible() -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for round_name, prompt_path, summary_path in ROUNDS:
        prompts = {str(row["case_id"]): row for row in read_jsonl(prompt_path)}
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for outcome in summary["prompt_summary"]:
            if not outcome["eligible"]:
                continue
            row = dict(prompts[str(outcome["case_id"])])
            row.update(
                {
                    "discovery_round": round_name,
                    "discovery_n": int(outcome["n"]),
                    "discovery_violations": int(outcome["violations"]),
                    "discovery_violation_rate": float(outcome["violation_rate"]),
                }
            )
            selected.append(row)
    if len({row["case_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate case_id in the combined eligible pool")
    return sorted(selected, key=lambda row: (row["risk_type"], row["case_id"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = collect_eligible()
    by_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_risk[str(row["risk_type"])].append(row)
    for risk, risk_rows in sorted(by_risk.items()):
        family_split = assign_risk_families(risk_rows)
        for row in risk_rows:
            row["trajectory_split"] = family_split[str(row["template_family"])]

    by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_family[str(row["template_family"])].add(str(row["trajectory_split"]))
    crossing = {family: splits for family, splits in by_family.items() if len(splits) != 1}
    if crossing:
        raise ValueError(f"Template families crossed splits: {crossing}")

    write_jsonl(args.out_jsonl, rows)
    dataset_sha256 = hashlib.sha256(args.out_jsonl.read_bytes()).hexdigest()
    config = {
        "experiment": "trajectory_confirmatory_v4",
        "model_id": "Qwen/Qwen3-0.6B",
        "cases_jsonl": args.out_jsonl.as_posix(),
        "dataset_sha256": dataset_sha256,
        "split_seed": SPLIT_SEED,
        "sampling": {
            "samples_per_prompt": 32,
            "seed_start": 5000000,
            "temperature": 0.8,
            "top_p": 0.95,
            "max_new_tokens": 48,
        },
        "features": {
            "checkpoints": [0, 2, 4, 6, 8, 10, 12, 16, 24],
            "layers": [0, 6, 12, 18, 24, 27],
            "primary_layer": 18,
        },
        "split_field": "trajectory_split",
        "discovery_seeds_excluded": True,
    }
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    counts = Counter((str(row["risk_type"]), str(row["trajectory_split"])) for row in rows)
    source_counts = Counter((str(row["discovery_round"]), str(row["trajectory_split"])) for row in rows)
    manifest = {
        "dataset_sha256": dataset_sha256,
        "prompt_count": len(rows),
        "template_family_count": len(by_family),
        "split_seed": SPLIT_SEED,
        "counts_by_risk_split": {
            f"{risk}/{split}": count for (risk, split), count in sorted(counts.items())
        },
        "counts_by_round_split": {
            f"{round_name}/{split}": count
            for (round_name, split), count in sorted(source_counts.items())
        },
        "prompts": [
            {
                "case_id": row["case_id"],
                "risk_type": row["risk_type"],
                "template_family": row["template_family"],
                "discovery_round": row["discovery_round"],
                "discovery_violation_rate": row["discovery_violation_rate"],
                "trajectory_split": row["trajectory_split"],
            }
            for row in rows
        ],
    }
    args.out_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# v4 Confirmatory Split Manifest",
        "",
        f"- prompts: `{len(rows)}`",
        f"- template families: `{len(by_family)}`",
        f"- split seed: `{SPLIT_SEED}`",
        f"- dataset SHA-256: `{dataset_sha256}`",
        "- discovery trajectories reused: `False`",
        "",
        "## Counts by Risk and Split",
        "",
        "| risk | train | validation | test | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for risk in sorted(by_risk):
        values = [counts[(risk, split)] for split in SPLITS]
        lines.append(f"| {risk} | {values[0]} | {values[1]} | {values[2]} | {sum(values)} |")
    lines.extend(
        [
            "",
            "## Frozen Prompts",
            "",
            "| prompt | risk | family | discovery | rate | split |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` | {row['risk_type']} | `{row['template_family']}` | "
            f"{row['discovery_round']} | {row['discovery_violation_rate']:.3f} | "
            f"{row['trajectory_split']} |"
        )
    args.out_manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Froze {len(rows)} prompts across {len(by_family)} template families")
    print(f"Dataset SHA-256: {dataset_sha256}")


if __name__ == "__main__":
    main()
