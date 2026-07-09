"""Compare dense J-lens readout stability between two fitted lenses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_dense_jlens_qwen import (  # noqa: E402
    apply_chat_template,
    build_watched_tokens,
    choose_dtype,
    get_layers,
    load_lens_npz,
    safe_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two fitted dense J-lens artifacts.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lens-a", type=Path, required=True)
    parser.add_argument("--lens-b", type=Path, required=True)
    parser.add_argument("--label-a", default="4fit")
    parser.add_argument("--label-b", default="32fit")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--condition", default="concept_present_target_absent")
    parser.add_argument("--limit-cases", type=int, default=200)
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
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


def select_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("stride:"):
        stride = max(1, int(spec.split(":", 1)[1]))
        layers = list(range(0, n_layers, stride))
        if (n_layers - 1) not in layers:
            layers.append(n_layers - 1)
        return layers
    layers = sorted({int(item.strip()) for item in spec.split(",") if item.strip()})
    bad = [layer for layer in layers if layer < 0 or layer >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) outside 0..{n_layers - 1}: {bad}")
    return layers


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return None
    return float(np.dot(x, y) / denom)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    return pearson(rank_values(np.asarray(x)), rank_values(np.asarray(y)))


@torch.no_grad()
def lens_readout_scores(
    *,
    model: torch.nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    lens: dict[int, torch.Tensor],
    layers: list[int],
    watched: list[Any],
    source_pos: int,
    device: torch.device,
) -> dict[str, float]:
    lm_head = model.get_output_embeddings()
    scores: dict[str, float] = {}
    for layer in layers:
        if layer not in lens:
            continue
        hidden = hidden_states[layer + 1][0, source_pos, :].detach()
        J = lens[layer].to(device=device, dtype=torch.float32)
        transported = torch.matmul(J, hidden.float())
        logits = lm_head(transported.to(dtype=hidden.dtype)[None, :])[0].float()
        for item in watched:
            key = f"L{layer}:{item.concept}:{item.text}:{item.token_id}"
            scores[key] = float(logits[item.token_id].item())
    return scores


def summarize(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"mean": None, "median": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dense J-Lens Stability Comparison",
        "",
        f"- model: `{payload['model_id']}`",
        f"- lens A: `{payload['label_a']}`",
        f"- lens B: `{payload['label_b']}`",
        f"- cases: `{payload['case_count']}`",
        f"- condition filter: `{payload['condition']}`",
        f"- layers: `{payload['layers']}`",
        "",
        "## Aggregate",
        "",
        "| metric | mean | median | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, stats in payload["aggregate"].items():
        lines.append(
            f"| {metric} | {fmt(stats['mean'])} | {fmt(stats['median'])} | "
            f"{fmt(stats['min'])} | {fmt(stats['max'])} |"
        )

    lines.extend(
        [
            "",
            "## By Risk",
            "",
            "| risk | cases | spearman mean | spearman median | top-1 agreement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for risk, row in payload["by_risk"].items():
        lines.append(
            f"| {risk} | {row['cases']} | {fmt(row['spearman']['mean'])} | "
            f"{fmt(row['spearman']['median'])} | {fmt(row['top1_agreement'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- High rank correlation means the fitted dense lens is stable to fitting-prompt choice.",
            "- Low top-1 agreement means individual headline tokens are unstable even if broad ranking is correlated.",
            "- This ablation should be run after fitting the 32-prompt lens artifact.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    rows = read_jsonl(args.cases)
    if args.condition:
        rows = [row for row in rows if row.get("condition") == args.condition]
    if args.limit_cases:
        rows = rows[: args.limit_cases]

    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=not args.allow_download, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    layers = select_layers(args.layers, len(get_layers(model)))
    lens_a, metadata_a = load_lens_npz(args.lens_a, layers=layers)
    lens_b, metadata_b = load_lens_npz(args.lens_b, layers=layers)
    layers = sorted(set(lens_a) & set(lens_b))

    case_rows = []
    for idx, row in enumerate(rows, start=1):
        batch = apply_chat_template(
            tokenizer,
            row["prompt"],
            row.get("system") or cfg.get("system_prompt"),
            max_seq_len=args.max_seq_len,
            use_chat_template=args.chat_template,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        source_pos = int(batch["input_ids"].shape[1] - 1)
        with torch.inference_mode():
            outputs = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
        watched = build_watched_tokens(tokenizer, cfg["concept_token_texts"], row.get("watch_concepts", []))
        scores_a = lens_readout_scores(
            model=model,
            hidden_states=outputs.hidden_states,
            lens=lens_a,
            layers=layers,
            watched=watched,
            source_pos=source_pos,
            device=device,
        )
        scores_b = lens_readout_scores(
            model=model,
            hidden_states=outputs.hidden_states,
            lens=lens_b,
            layers=layers,
            watched=watched,
            source_pos=source_pos,
            device=device,
        )
        keys = sorted(set(scores_a) & set(scores_b))
        vector_a = [scores_a[key] for key in keys]
        vector_b = [scores_b[key] for key in keys]
        top_a = max(keys, key=lambda key: scores_a[key]) if keys else ""
        top_b = max(keys, key=lambda key: scores_b[key]) if keys else ""
        case_rows.append(
            {
                "case_id": row["case_id"],
                "risk_type": row["risk_type"],
                "condition": row["condition"],
                "keys": len(keys),
                "spearman": spearman(vector_a, vector_b),
                "pearson": pearson(np.asarray(vector_a), np.asarray(vector_b)),
                "top_a": top_a,
                "top_b": top_b,
                "top1_agreement": top_a == top_b,
            }
        )
        if idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] compared {row['case_id']}", flush=True)

    by_risk: dict[str, Any] = {}
    for risk in sorted({row["risk_type"] for row in case_rows}):
        group = [row for row in case_rows if row["risk_type"] == risk]
        by_risk[risk] = {
            "cases": len(group),
            "spearman": summarize([row["spearman"] for row in group]),
            "pearson": summarize([row["pearson"] for row in group]),
            "top1_agreement": sum(1 for row in group if row["top1_agreement"]) / max(1, len(group)),
        }

    payload = {
        "model_id": args.model_id,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "lens_a": str(args.lens_a),
        "lens_b": str(args.lens_b),
        "metadata_a": metadata_a,
        "metadata_b": metadata_b,
        "case_count": len(case_rows),
        "condition": args.condition,
        "layers": layers,
        "aggregate": {
            "spearman": summarize([row["spearman"] for row in case_rows]),
            "pearson": summarize([row["pearson"] for row in case_rows]),
        },
        "top1_agreement": sum(1 for row in case_rows if row["top1_agreement"]) / max(1, len(case_rows)),
        "by_risk": by_risk,
        "cases": case_rows,
    }
    out_dir = args.out_dir / safe_name(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lens_stability.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    write_markdown(out_dir / "LENS_STABILITY.md", payload)
    print(f"Saved stability comparison to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
