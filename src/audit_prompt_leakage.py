"""Tokenizer-level leakage audit for prompt sets.

String checks catch obvious target words, but a watched concept can still leak
through tokenizer-specific variants such as " reveal" vs "reveal" or a
multi-token split.  This script audits target-absent prompts by tokenizing the
prompt and asserting that no watched concept token sequence appears directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit target-absent prompts for tokenizer-level leakage.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--allow-leaks", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def case_variants(row: dict[str, Any], config: dict[str, Any]) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    concept_texts = config.get("concept_token_texts", {})
    for concept in row.get("watch_concepts", []):
        for text in concept_texts.get(concept, []):
            base = text.strip()
            for candidate in {text, base, " " + base, base.capitalize(), " " + base.capitalize()}:
                if candidate:
                    variants.append((concept, candidate))
    for term in row.get("target_terms", []):
        base = str(term).strip()
        for candidate in {base, " " + base, base.capitalize(), " " + base.capitalize()}:
            if candidate:
                variants.append(("target_term", candidate))
    return sorted(set(variants))


def find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    stop = len(haystack) - len(needle) + 1
    for idx in range(stop):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None


def audit_rows(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    violations = []
    token_sequences_checked = 0
    target_absent_cases = 0
    by_risk = defaultdict(lambda: {"target_absent_cases": 0, "violations": 0})

    for row in rows:
        if row.get("target_present"):
            continue
        target_absent_cases += 1
        risk = row.get("risk_type", "unknown")
        by_risk[risk]["target_absent_cases"] += 1
        prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        seen_sequences: set[tuple[int, ...]] = set()
        for concept, text in case_variants(row, config):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            key = tuple(int(token_id) for token_id in ids)
            if key in seen_sequences:
                continue
            seen_sequences.add(key)
            token_sequences_checked += 1
            at = find_subsequence(prompt_ids, list(key))
            if at is None:
                continue
            by_risk[risk]["violations"] += 1
            violations.append(
                {
                    "case_id": row["case_id"],
                    "risk_type": risk,
                    "condition": row["condition"],
                    "concept": concept,
                    "variant": text,
                    "token_ids": list(key),
                    "token_index": at,
                    "decoded": tokenizer.decode(list(key)),
                    "prompt": row["prompt"],
                }
            )

    return {
        "target_absent_cases": target_absent_cases,
        "token_sequences_checked": token_sequences_checked,
        "violation_count": len(violations),
        "violations": violations,
        "by_risk": dict(sorted((key, dict(value)) for key, value in by_risk.items())),
    }


def write_markdown(path: Path, payload: dict[str, Any], model_id: str) -> None:
    lines = [
        "# Token-Level Prompt Leakage Audit",
        "",
        f"- model tokenizer: `{model_id}`",
        f"- target-absent cases: `{payload['target_absent_cases']}`",
        f"- token sequences checked: `{payload['token_sequences_checked']}`",
        f"- violations: `{payload['violation_count']}`",
        "",
        "## By Risk",
        "",
        "| risk | target-absent cases | violations |",
        "|---|---:|---:|",
    ]
    for risk, row in payload["by_risk"].items():
        lines.append(f"| {risk} | {row['target_absent_cases']} | {row['violations']} |")
    if payload["violations"]:
        lines.extend(["", "## Violations", ""])
        for item in payload["violations"][:50]:
            lines.append(
                f"- `{item['case_id']}` {item['concept']} `{item['variant']}` "
                f"tokens={item['token_ids']} at={item['token_index']}"
            )
        if len(payload["violations"]) > 50:
            lines.append(f"- ... {len(payload['violations']) - 50} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.cases)
    config = read_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    payload = audit_rows(rows, config, tokenizer)
    payload["model_id"] = args.model_id
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.out_md:
        write_markdown(args.out_md, payload, args.model_id)
    print(
        f"Audited {payload['target_absent_cases']} target-absent cases: "
        f"{payload['violation_count']} token-level violation(s).",
        flush=True,
    )
    if payload["violation_count"] and not args.allow_leaks:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
