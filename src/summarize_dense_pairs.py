"""Summarize matched attack/control deltas from dense J-lens results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dense J-lens matched pairs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--lens-kind", choices=["jlens", "direct"], default="jlens")
    return parser.parse_args()


def rows_by_concept(case: dict[str, Any], lens_kind: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for layer in case.get("layers", []):
        for row in layer.get(lens_kind, {}).get("watched", []):
            item = dict(row)
            item["layer"] = layer["layer"]
            grouped[item["concept"]].append(item)
    return grouped


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return min(rows, key=lambda row: (row["rank"], -row["score"]))


def summarize_pair(control: dict[str, Any], attack: dict[str, Any], lens_kind: str) -> dict[str, Any]:
    control_rows = rows_by_concept(control, lens_kind)
    attack_rows = rows_by_concept(attack, lens_kind)
    concepts = sorted(set(control_rows) | set(attack_rows))
    concept_deltas = []
    for concept in concepts:
        c_best = best_row(control_rows.get(concept, []))
        a_best = best_row(attack_rows.get(concept, []))
        if c_best is None or a_best is None:
            continue
        concept_deltas.append(
            {
                "concept": concept,
                "control": c_best,
                "attack": a_best,
                "rank_delta_control_minus_attack": c_best["rank"] - a_best["rank"],
                "score_delta_attack_minus_control": a_best["score"] - c_best["score"],
            }
        )
    concept_deltas.sort(
        key=lambda item: (
            -item["rank_delta_control_minus_attack"],
            -item["score_delta_attack_minus_control"],
        )
    )
    best_delta = concept_deltas[0] if concept_deltas else None
    return {
        "pair": attack["pair"],
        "control_id": control["id"],
        "attack_id": attack["id"],
        "control_validator": control["validator"],
        "attack_validator": attack["validator"],
        "best_delta": best_delta,
        "concept_deltas": concept_deltas,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dense J-Lens Paired Delta Summary",
        "",
        f"- source: `{payload['source']}`",
        f"- lens kind: `{payload['lens_kind']}`",
        "",
        "Positive rank delta means the watched concept ranked higher in the attack than in its matched control.",
        "",
        "| pair | control validator | attack validator | best concept | control rank | attack rank | rank delta | score delta |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for pair in payload["pairs"]:
        best = pair.get("best_delta")
        if not best:
            continue
        c = best["control"]
        a = best["attack"]
        lines.append(
            "| {pair} | {cv} | {av} | {concept} | {cr} | {ar} | {rd} | {sd:.3f} |".format(
                pair=pair["pair"],
                cv=pair["control_validator"]["decision"],
                av=pair["attack_validator"]["decision"],
                concept=f"L{a['layer']} `{best['concept']}:{a['decoded']}`",
                cr=c["rank"],
                ar=a["rank"],
                rd=best["rank_delta_control_minus_attack"],
                sd=best["score_delta_attack_minus_control"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A useful pre-commit signal should ideally have a positive rank delta and a rollback attack outcome.",
            "- Negative or small deltas mean the raw watched token is also active in the benign control.",
            "- This paired view is stricter than reporting the best raw rank per case.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in data["cases"]:
        group = case.get("group")
        pair = case.get("pair")
        if group in {"control", "attack"} and pair:
            by_pair[pair][group] = case

    pairs = []
    for pair, grouped in sorted(by_pair.items()):
        if "control" not in grouped or "attack" not in grouped:
            continue
        pairs.append(summarize_pair(grouped["control"], grouped["attack"], args.lens_kind))

    payload = {
        "source": str(args.input),
        "lens_kind": args.lens_kind,
        "pairs": pairs,
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    write_markdown(args.out_md, payload)


if __name__ == "__main__":
    main()
