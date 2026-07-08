from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    lines = [
        "# Jacobian-Vector Cross-Model Interpretation",
        "",
        "This is a finite-difference Jacobian-vector lens, not a fitted full Jacobian lens.",
        "",
        "## Compact Table",
        "",
        "| Model | Case | J best | J layer | J rank | Direct best | Direct rank | Validator | Hits |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for payload in payloads:
        for row in payload["results"]:
            jv = row["jacobian_vector_summary"]
            direct = row["direct_logit_lens_summary"]
            validation = row["validation"]
            hits = ", ".join(validation["hits"]) if validation["hits"] else "-"
            lines.append(
                f"| `{payload['model_id']}` | `{row['case_id']}` | {jv['best_concept']} | "
                f"{jv['best_layer']} | {jv['best_rank']} | {direct['best_concept']} | "
                f"{direct['best_rank']} | {validation['action']} | {hits} |"
            )

    lines.extend(
        [
            "",
            "## Takeaways",
            "",
            "- Partial positive reproduction: multiple open models expose silent or safety-relevant concepts through the lens before or despite final-output filtering.",
            "- Strongest cross-model pattern: `fake_commit` surfaces `commit` in all three models, while the visible response is refused or validator-blocked in two of three models.",
            "- Silent-instruction pattern appears in smaller/older models: Phi-3 surfaces `nine` for silent math; Qwen2.5-0.5B surfaces `fruit` for silent citrus; Phi-4 direct lens surfaces `orange` but Jacobian-vector signal is weaker.",
            "- Safety-monitoring pattern appears in spoiler/schema cases, but several are final-output leaks rather than purely hidden activations.",
            "- This does not reproduce Anthropic's strongest causal claims: reportability, flexible broadcast, ablation, and concept swapping still require interventions and a better-fitted lens.",
            "",
            "## Useful Next Experiments",
            "",
            "- Add matched prompt pairs where the target concept is not lexically present in the prompt.",
            "- Run per-position readouts instead of only the final prompt position.",
            "- Fit a small residual-to-final linear lens on safe/unsafe traces, then compare it with this finite-difference JVP lens.",
            "- Add causal intervention: damp or replace the transported concept direction and measure generation changes.",
        ]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

