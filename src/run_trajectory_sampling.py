"""Sample divergent trajectories and optionally cache residual checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_dense_jlens_qwen import apply_chat_template, choose_dtype, get_layers, safe_name  # noqa: E402
from runtime_validators import validate_runtime_output  # noqa: E402


STAT_NAMES = ["entropy", "max_probability", "top5_mass", "top10_mass", "target_token_mass"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample policy-divergent model trajectories.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--conditions", default="concept_present_target_absent,target_present_concept_absent")
    parser.add_argument("--risks", default="")
    parser.add_argument("--per-risk-condition", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--screen-min-rate", type=float, default=0.2)
    parser.add_argument("--screen-max-rate", type=float, default=0.8)
    parser.add_argument("--capture-checkpoints", action="store_true")
    parser.add_argument("--checkpoints", default="0,2,4,6,8,12,16,24")
    parser.add_argument("--layers", default="0,6,12,18,24,27")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def stable_order(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_cases(
    rows: list[dict[str, Any]],
    conditions: set[str],
    risks: set[str],
    per_risk_condition: int,
) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if (not conditions or row.get("condition") in conditions)
        and (not risks or row.get("risk_type") in risks)
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        grouped[(str(row["risk_type"]), str(row["condition"]))].append(row)
    selected = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda row: stable_order(str(row["case_id"])))
        selected.extend(group[:per_risk_condition] if per_risk_condition else group)
    return sorted(selected, key=lambda row: (row["risk_type"], row["condition"], row["case_id"]))


def trajectory_seed(seed_start: int, prompt_index: int, sample_index: int) -> int:
    return int(seed_start + prompt_index * 100003 + sample_index)


def generated_ids_without_padding(ids: torch.Tensor, eos_token_id: int | None) -> list[int]:
    values = [int(value) for value in ids.detach().cpu().tolist()]
    if eos_token_id is not None and eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return values


def first_policy_landing(
    tokenizer: Any,
    generated_ids: list[int],
    row: dict[str, Any],
) -> tuple[int | None, str]:
    spec = row.get("runtime_validation") or {}
    markers = [str(value).strip() for value in spec.get("policy_markers", []) if str(value).strip()]
    if not markers:
        risk_type = str(spec.get("type", ""))
        if risk_type == "early_spoiler" and spec.get("forbidden_fact"):
            parts = str(spec["forbidden_fact"]).split(maxsplit=2)
            markers = [parts[2] if len(parts) == 3 else str(spec["forbidden_fact"])]
        elif risk_type == "fake_commit":
            markers = [str(value) for value in spec.get("unsafe_statuses", [])]
            markers.extend(str(value) for value in spec.get("unsafe_text_markers", []))
        elif risk_type in {"hidden_fields", "schema_bypass"}:
            markers = [str(value) for value in spec.get("forbidden_keys", [])]

    full_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    low = full_text.lower()
    matches = [(low.find(marker.lower()), marker) for marker in markers if marker.lower() in low]
    if matches:
        marker_start, marker = min(matches, key=lambda item: item[0])
        for token_count in range(1, len(generated_ids) + 1):
            prefix = tokenizer.decode(
                generated_ids[:token_count],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(prefix) > marker_start:
                return token_count, f"policy_marker:{marker}"

    # Fallback covers unexpected violating outputs that do not copy a declared marker.
    for token_count in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:token_count], skip_special_tokens=True)
        decision = validate_runtime_output(prefix, row)["policy"]["decision"]
        if decision == "rollback":
            return token_count, "incremental_validator_fallback"
    return None, "no_policy_landing"


def target_token_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    texts = list(row.get("target_terms", []))
    spec = row.get("runtime_validation") or {}
    texts.extend(str(value) for value in spec.get("unsafe_statuses", []))
    texts.extend(str(value) for value in spec.get("forbidden_keys", []))
    texts.extend(str(value) for value in spec.get("unsafe_text_markers", []))
    if spec.get("forbidden_fact"):
        texts.extend(str(spec["forbidden_fact"]).split())
    token_ids: set[int] = set()
    for text in texts:
        for variant in {str(text), " " + str(text).strip()}:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(int(ids[0]))
    return sorted(token_ids)


def distribution_stats(logits: torch.Tensor, watched_ids: list[int]) -> list[float]:
    probs = torch.softmax(logits.detach().float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = float(-(probs * log_probs).sum().item())
    top_values = torch.topk(probs, k=10).values
    target_mass = float(probs[watched_ids].sum().item()) if watched_ids else 0.0
    return [
        entropy,
        float(top_values[0].item()),
        float(top_values[:5].sum().item()),
        float(top_values.sum().item()),
        target_mass,
    ]


def generate_one(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    row: dict[str, Any],
    *,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    capture_checkpoints: bool,
    checkpoints: list[int],
    layers: list[int],
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    kwargs = {
        **batch,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    if capture_checkpoints:
        kwargs.update(return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    elapsed = time.perf_counter() - started
    prompt_len = int(batch["input_ids"].shape[1])
    sequences = output.sequences if capture_checkpoints else output
    generated_ids = generated_ids_without_padding(
        sequences[0, prompt_len:], getattr(tokenizer, "eos_token_id", None)
    )
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    validators = validate_runtime_output(text, row)
    landing, landing_basis = first_policy_landing(tokenizer, generated_ids, row)
    record = {
        "output": text,
        "generated_token_ids": generated_ids,
        "generated_token_count": len(generated_ids),
        "generation_seconds": elapsed,
        "policy_validator": validators["policy"],
        "lexical_validator": validators["lexical"],
        "structural_validator": validators["structural"],
        "policy_landing_token": landing,
        "policy_landing_basis": landing_basis,
    }
    if not capture_checkpoints:
        return record, None, None, None

    d_model = int(model.config.hidden_size)
    features = np.full((len(checkpoints), len(layers), d_model), np.nan, dtype=np.float16)
    stats = np.full((len(checkpoints), len(STAT_NAMES)), np.nan, dtype=np.float32)
    valid = np.zeros(len(checkpoints), dtype=np.bool_)
    prefixes = [""] * len(checkpoints)
    checkpoint_index = {step: idx for idx, step in enumerate(checkpoints)}
    watched_ids = target_token_ids(tokenizer, row)
    available_steps = min(len(output.hidden_states), len(output.scores))
    for step in checkpoints:
        if step >= available_steps or step > len(generated_ids):
            continue
        idx = checkpoint_index[step]
        hidden_tuple = output.hidden_states[step]
        for layer_idx, layer in enumerate(layers):
            hidden = hidden_tuple[layer + 1][0, -1, :].detach().float().cpu().numpy()
            features[idx, layer_idx] = hidden.astype(np.float16)
        stats[idx] = np.asarray(distribution_stats(output.scores[step][0], watched_ids), dtype=np.float32)
        prefixes[idx] = tokenizer.decode(generated_ids[:step], skip_special_tokens=True)
        valid[idx] = True
    record["checkpoint_prefixes"] = prefixes
    return record, features, stats, valid


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Trajectory Sampling Summary",
        "",
        f"- model: `{payload['model_id']}`",
        f"- prompts: `{payload['prompt_count']}`",
        f"- trajectories: `{payload['trajectory_count']}`",
        f"- samples per prompt: `{payload['samples_per_prompt']}`",
        f"- temperature / top-p: `{payload['temperature']}` / `{payload['top_p']}`",
        f"- seed start: `{payload['seed_start']}`",
        f"- capture checkpoints: `{payload['capture_checkpoints']}`",
        f"- CUDA peak allocated / reserved GiB: "
        f"`{payload['cuda_peak_allocated_gib']}` / `{payload['cuda_peak_reserved_gib']}`",
        f"- generation tokens/sec: `{payload['tokens_per_second']:.3f}`",
        f"- eligible prompts: `{payload['eligible_prompt_count']}`",
        "",
        "## Prompt Outcomes",
        "",
        "| prompt | risk | condition | n | violations | rate | eligible |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["prompt_summary"]:
        lines.append(
            f"| `{row['case_id']}` | {row['risk_type']} | `{row['condition']}` | "
            f"{row['n']} | {row['violations']} | {row['violation_rate']:.3f} | "
            f"{row['eligible']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    risks = {value.strip() for value in args.risks.split(",") if value.strip()}
    checkpoints = parse_ints(args.checkpoints)
    layers = parse_ints(args.layers)
    rows = select_cases(read_jsonl(args.cases), conditions, risks, args.per_risk_condition)
    if not rows:
        raise SystemExit("No cases matched the requested filters")

    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=not args.allow_download,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    n_layers = len(get_layers(model))
    if any(layer < 0 or layer >= n_layers for layer in layers):
        raise ValueError(f"Requested layers {layers} outside 0..{n_layers - 1}")

    out_dir = args.out_dir / safe_name(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    trajectory_rows = []
    feature_rows = []
    stat_rows = []
    valid_rows = []
    total_tokens = 0
    total_seconds = 0.0
    for prompt_index, row in enumerate(rows):
        prompt_rows.append(row)
        batch = apply_chat_template(
            tokenizer,
            row["prompt"],
            row.get("system"),
            max_seq_len=args.max_seq_len,
            use_chat_template=True,
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        for sample_index in range(args.samples_per_prompt):
            seed = trajectory_seed(args.seed_start, prompt_index, sample_index)
            record, features, stats, valid = generate_one(
                model,
                tokenizer,
                batch,
                row,
                seed=seed,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                capture_checkpoints=args.capture_checkpoints,
                checkpoints=checkpoints,
                layers=layers,
            )
            trajectory_id = f"{row['case_id']}__seed{seed}"
            trajectory_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "case_id": row["case_id"],
                    "risk_type": row["risk_type"],
                    "condition": row["condition"],
                    "template_family": row.get("template_family"),
                    "seed": seed,
                    **record,
                }
            )
            total_tokens += int(record["generated_token_count"])
            total_seconds += float(record["generation_seconds"])
            if args.capture_checkpoints:
                feature_rows.append(features)
                stat_rows.append(stats)
                valid_rows.append(valid)
        completed = (prompt_index + 1) * args.samples_per_prompt
        print(f"[{prompt_index + 1}/{len(rows)} prompts] {completed} trajectories", flush=True)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        by_case[str(row["case_id"])].append(row)
    prompt_summary = []
    eligible_ids = set()
    for prompt in prompt_rows:
        group = by_case[str(prompt["case_id"])]
        violations = sum(row["policy_validator"]["decision"] == "rollback" for row in group)
        rate = violations / len(group)
        eligible = args.screen_min_rate <= rate <= args.screen_max_rate
        if eligible:
            eligible_ids.add(str(prompt["case_id"]))
        prompt_summary.append(
            {
                "case_id": prompt["case_id"],
                "risk_type": prompt["risk_type"],
                "condition": prompt["condition"],
                "n": len(group),
                "violations": violations,
                "violation_rate": rate,
                "eligible": eligible,
            }
        )

    payload = {
        "model_id": args.model_id,
        "model_revision": args.revision,
        "prompt_count": len(prompt_rows),
        "trajectory_count": len(trajectory_rows),
        "samples_per_prompt": args.samples_per_prompt,
        "seed_start": args.seed_start,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "capture_checkpoints": args.capture_checkpoints,
        "checkpoints": checkpoints if args.capture_checkpoints else [],
        "layers": layers if args.capture_checkpoints else [],
        "stat_names": STAT_NAMES if args.capture_checkpoints else [],
        "total_generated_tokens": total_tokens,
        "generation_seconds": total_seconds,
        "tokens_per_second": total_tokens / max(total_seconds, 1e-9),
        "screen_min_rate": args.screen_min_rate,
        "screen_max_rate": args.screen_max_rate,
        "eligible_prompt_count": len(eligible_ids),
        "cuda_peak_allocated_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
        "cuda_peak_reserved_gib": (
            round(torch.cuda.max_memory_reserved(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
        "prompt_summary": prompt_summary,
    }
    write_jsonl(out_dir / "prompts.jsonl", prompt_rows)
    write_jsonl(out_dir / "trajectories.jsonl", trajectory_rows)
    write_jsonl(out_dir / "eligible_prompts.jsonl", [row for row in prompt_rows if row["case_id"] in eligible_ids])
    (out_dir / "sampling_summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    write_summary(out_dir / "SAMPLING_SUMMARY.md", payload)
    if args.capture_checkpoints:
        np.savez(
            out_dir / "trajectory_features.npz",
            features=np.stack(feature_rows),
            stats=np.stack(stat_rows),
            valid=np.stack(valid_rows),
            trajectory_ids=np.asarray([row["trajectory_id"] for row in trajectory_rows]),
            checkpoints=np.asarray(checkpoints, dtype=np.int32),
            layers=np.asarray(layers, dtype=np.int32),
            stat_names=np.asarray(STAT_NAMES),
        )
    print(f"Saved trajectory run to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
