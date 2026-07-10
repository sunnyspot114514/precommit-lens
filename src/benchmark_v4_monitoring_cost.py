"""Benchmark plain generation against checkpoint residual capture on frozen prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dense_jlens_qwen import apply_chat_template, choose_dtype, get_layers
from run_trajectory_sampling import generate_one, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark v4 residual monitoring overhead.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=9000000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_mode(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    row: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    capture: bool,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    record, _features, _stats, _valid = generate_one(
        model,
        tokenizer,
        batch,
        row,
        seed=seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        capture_checkpoints=capture,
        checkpoints=[0, 2, 4, 6, 8, 10, 12, 16, 24],
        layers=[0, 6, 12, 18, 24, 27],
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return record, time.perf_counter() - started


def bootstrap_ratio(plain: np.ndarray, capture: np.ndarray, samples: int = 2000) -> list[float]:
    rng = np.random.default_rng(20260712)
    values = []
    for _ in range(samples):
        selected = rng.integers(0, len(plain), len(plain))
        values.append(float(capture[selected].sum() / plain[selected].sum()))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    args = parse_args()
    rows = [row for row in read_jsonl(args.cases) if row["trajectory_split"] == "test"]
    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, local_files_only=not args.allow_download, trust_remote_code=True
    )
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
    if len(get_layers(model)) <= 27:
        raise ValueError("Model does not expose the frozen v4 layer set")

    batches = [
        {
            key: value.to(device)
            for key, value in apply_chat_template(
                tokenizer,
                row["prompt"],
                row.get("system"),
                max_seq_len=args.max_seq_len,
                use_chat_template=True,
            ).items()
        }
        for row in rows
    ]
    # Warm both paths once; warmup timings are excluded.
    run_mode(model, tokenizer, batches[0], rows[0], args, args.seed_start - 2, False)
    run_mode(model, tokenizer, batches[0], rows[0], args, args.seed_start - 1, True)

    records = []
    for prompt_idx, (row, batch) in enumerate(zip(rows, batches, strict=True)):
        for sample_idx in range(args.samples_per_prompt):
            seed = args.seed_start + prompt_idx * 100003 + sample_idx
            order = [False, True] if (prompt_idx + sample_idx) % 2 == 0 else [True, False]
            outputs: dict[bool, dict[str, Any]] = {}
            timings: dict[bool, float] = {}
            for capture in order:
                outputs[capture], timings[capture] = run_mode(
                    model, tokenizer, batch, row, args, seed, capture
                )
            identical = outputs[False]["generated_token_ids"] == outputs[True]["generated_token_ids"]
            if not identical:
                raise ValueError(f"Plain/capture RNG mismatch for {row['case_id']} seed {seed}")
            records.append(
                {
                    "case_id": row["case_id"],
                    "risk_type": row["risk_type"],
                    "seed": seed,
                    "generated_tokens": outputs[False]["generated_token_count"],
                    "plain_seconds": timings[False],
                    "capture_seconds": timings[True],
                    "capture_to_plain_ratio": timings[True] / timings[False],
                    "identical_output": identical,
                }
            )
        print(f"Benchmarked {prompt_idx + 1}/{len(rows)} prompts", flush=True)

    plain = np.asarray([row["plain_seconds"] for row in records], dtype=np.float64)
    capture = np.asarray([row["capture_seconds"] for row in records], dtype=np.float64)
    summary = {
        "model_id": args.model_id,
        "prompt_count": len(rows),
        "paired_runs": len(records),
        "samples_per_prompt": args.samples_per_prompt,
        "seed_start": args.seed_start,
        "checkpoints": [0, 2, 4, 6, 8, 10, 12, 16, 24],
        "layers": [0, 6, 12, 18, 24, 27],
        "all_outputs_identical": all(row["identical_output"] for row in records),
        "plain_seconds_total": float(plain.sum()),
        "capture_seconds_total": float(capture.sum()),
        "capture_to_plain_ratio": float(capture.sum() / plain.sum()),
        "capture_to_plain_ratio_ci95": bootstrap_ratio(plain, capture),
        "mean_paired_overhead_seconds": float(np.mean(capture - plain)),
        "records": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "monitoring_cost_benchmark.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# v4 Monitoring Cost Benchmark",
        "",
        f"- frozen test prompts: `{len(rows)}`",
        f"- paired runs: `{len(records)}`",
        f"- outputs identical across modes: `{summary['all_outputs_identical']}`",
        f"- plain generation seconds: `{summary['plain_seconds_total']:.3f}`",
        f"- capture seconds: `{summary['capture_seconds_total']:.3f}`",
        f"- capture/plain ratio: `{summary['capture_to_plain_ratio']:.3f}` "
        f"(95% paired bootstrap CI `{summary['capture_to_plain_ratio_ci95'][0]:.3f}`-"
        f"`{summary['capture_to_plain_ratio_ci95'][1]:.3f}`)",
        f"- mean paired overhead: `{summary['mean_paired_overhead_seconds']:.3f}` seconds/trajectory",
        "",
        "This measures the current six-layer, nine-checkpoint research capture path. It is not "
        "an estimate for a production hook that exports only layer 18.",
    ]
    (args.out_dir / "MONITORING_COST_BENCHMARK.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Capture/plain ratio: {summary['capture_to_plain_ratio']:.3f}")


if __name__ == "__main__":
    main()
