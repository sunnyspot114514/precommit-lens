"""Run validator-aware trajectory yield sampling through a frozen Ollama model."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_dense_jlens_qwen import safe_name  # noqa: E402
from run_trajectory_sampling import (  # noqa: E402
    read_jsonl,
    select_cases,
    trajectory_seed,
    write_jsonl,
)
from runtime_validators import validate_runtime_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample trajectories through Ollama.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--expected-quantization", default="Q4_K_M")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--conditions", default="")
    parser.add_argument("--risks", default="")
    parser.add_argument("--per-risk-condition", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--screen-min-rate", type=float, default=0.2)
    parser.add_argument("--screen-max-rate", type=float, default=0.8)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args()


def normalized_digest(value: str) -> str:
    return value.lower().removeprefix("sha256:")


def api_json(
    endpoint: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code} for {path}: {body}") from exc


def find_model_metadata(tags_payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in tags_payload.get("models", [])
        if row.get("name") == model_id or row.get("model") == model_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one local Ollama model named {model_id}, found {len(matches)}")
    return dict(matches[0])


def validate_model_metadata(
    metadata: dict[str, Any], expected_digest: str, expected_quantization: str
) -> None:
    actual_digest = normalized_digest(str(metadata.get("digest", "")))
    if actual_digest != normalized_digest(expected_digest):
        raise ValueError(
            f"Ollama model digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
    actual_quantization = str((metadata.get("details") or {}).get("quantization_level", ""))
    if actual_quantization != expected_quantization:
        raise ValueError(
            "Ollama model quantization mismatch: "
            f"expected {expected_quantization}, got {actual_quantization}"
        )


def sampling_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "num_ctx": args.num_ctx,
        "num_predict": args.max_new_tokens,
    }


def build_chat_payload(
    model_id: str,
    row: dict[str, Any],
    *,
    seed: int,
    options: dict[str, Any],
    keep_alive: str,
) -> dict[str, Any]:
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": str(row["system"])})
    messages.append({"role": "user", "content": str(row["prompt"])})
    return {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {**options, "seed": seed},
    }


def generate_with_retry(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[dict[str, Any], float]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        started = time.perf_counter()
        try:
            response = api_json(
                endpoint,
                "/api/chat",
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            return response, time.perf_counter() - started
        except (OSError, RuntimeError, TimeoutError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Ollama generation failed after retries: {last_error}")


def summarize_prompt_outcomes(
    prompts: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    *,
    minimum_rate: float,
    maximum_rate: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_case[str(row["case_id"])].append(row)
    prompt_summary = []
    eligible_ids: set[str] = set()
    for prompt in prompts:
        case_id = str(prompt["case_id"])
        group = by_case[case_id]
        violations = sum(
            row["policy_validator"]["decision"] == "rollback" for row in group
        )
        rate = violations / len(group)
        eligible = minimum_rate <= rate <= maximum_rate
        if eligible:
            eligible_ids.add(case_id)
        prompt_summary.append(
            {
                "case_id": case_id,
                "risk_type": prompt["risk_type"],
                "condition": prompt["condition"],
                "n": len(group),
                "violations": violations,
                "violation_rate": rate,
                "eligible": eligible,
            }
        )
    return prompt_summary, eligible_ids


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    details = summary["model_metadata"].get("details") or {}
    lines = [
        "# Ollama Trajectory Sampling Summary",
        "",
        f"- model: `{summary['model_id']}`",
        f"- digest: `{summary['model_revision']}`",
        f"- family / parameters / quantization: `{details.get('family')}` / "
        f"`{details.get('parameter_size')}` / `{details.get('quantization_level')}`",
        f"- Ollama version: `{summary['ollama_version']}`",
        f"- prompts / trajectories: `{summary['prompt_count']}` / "
        f"`{summary['trajectory_count']}`",
        f"- temperature / top-p / top-k: `{summary['temperature']}` / "
        f"`{summary['top_p']}` / `{summary['top_k']}`",
        f"- seed start: `{summary['seed_start']}`",
        f"- generation tokens/s including request overhead: "
        f"`{summary['tokens_per_second']:.3f}`",
        f"- eligible prompts: `{summary['eligible_prompt_count']}`",
        "",
        "## Prompt Outcomes",
        "",
        "| prompt | risk | n | violations | rate | eligible |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["prompt_summary"]:
        lines.append(
            f"| `{row['case_id']}` | {row['risk_type']} | {row['n']} | "
            f"{row['violations']} | {row['violation_rate']:.3f} | "
            f"{row['eligible']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unload_model(endpoint: str, model_id: str) -> None:
    try:
        api_json(
            endpoint,
            "/api/generate",
            payload={"model": model_id, "keep_alive": 0},
            timeout_seconds=60.0,
        )
    except Exception as exc:  # Unloading is cleanup; preserve completed outputs.
        print(f"Warning: could not unload {model_id}: {exc}", flush=True)


def main() -> None:
    args = parse_args()
    conditions = {value.strip() for value in args.conditions.split(",") if value.strip()}
    risks = {value.strip() for value in args.risks.split(",") if value.strip()}
    prompts = select_cases(
        read_jsonl(args.cases), conditions, risks, args.per_risk_condition
    )
    if not prompts:
        raise SystemExit("No cases matched the requested filters")

    version = api_json(args.endpoint, "/api/version", timeout_seconds=30.0)["version"]
    metadata = find_model_metadata(
        api_json(args.endpoint, "/api/tags", timeout_seconds=30.0), args.model_id
    )
    validate_model_metadata(
        metadata, args.expected_digest, args.expected_quantization
    )
    options = sampling_options(args)
    trajectories = []
    total_tokens = 0
    total_wall_seconds = 0.0
    total_api_eval_seconds = 0.0
    try:
        for prompt_index, row in enumerate(prompts):
            for sample_index in range(args.samples_per_prompt):
                seed = trajectory_seed(args.seed_start, prompt_index, sample_index)
                payload = build_chat_payload(
                    args.model_id,
                    row,
                    seed=seed,
                    options=options,
                    keep_alive=args.keep_alive,
                )
                response, wall_seconds = generate_with_retry(
                    args.endpoint,
                    payload,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )
                message = response.get("message") or {}
                text = str(message.get("content", "")).strip()
                validators = validate_runtime_output(text, row)
                eval_count = int(response.get("eval_count") or 0)
                eval_seconds = float(response.get("eval_duration") or 0) / 1e9
                total_tokens += eval_count
                total_wall_seconds += wall_seconds
                total_api_eval_seconds += eval_seconds
                trajectory_id = f"{row['case_id']}__seed{seed}"
                trajectories.append(
                    {
                        "trajectory_id": trajectory_id,
                        "case_id": row["case_id"],
                        "risk_type": row["risk_type"],
                        "condition": row["condition"],
                        "template_family": row.get("template_family"),
                        "seed": seed,
                        "output": text,
                        "generated_token_ids": [],
                        "generated_token_ids_available": False,
                        "generated_token_count": eval_count,
                        "generation_seconds": wall_seconds,
                        "ollama_total_seconds": float(response.get("total_duration") or 0)
                        / 1e9,
                        "ollama_eval_seconds": eval_seconds,
                        "ollama_prompt_eval_count": int(
                            response.get("prompt_eval_count") or 0
                        ),
                        "thinking_char_count": len(str(message.get("thinking", ""))),
                        "done_reason": response.get("done_reason"),
                        "policy_validator": validators["policy"],
                        "lexical_validator": validators["lexical"],
                        "structural_validator": validators["structural"],
                        "policy_landing_token": None,
                        "policy_landing_basis": "not_collected_ollama_appendix",
                    }
                )
            completed = (prompt_index + 1) * args.samples_per_prompt
            print(
                f"[{prompt_index + 1}/{len(prompts)} prompts] "
                f"{completed} trajectories",
                flush=True,
            )
    finally:
        unload_model(args.endpoint, args.model_id)

    prompt_summary, eligible_ids = summarize_prompt_outcomes(
        prompts,
        trajectories,
        minimum_rate=args.screen_min_rate,
        maximum_rate=args.screen_max_rate,
    )
    summary = {
        "model_id": args.model_id,
        "model_revision": normalized_digest(args.expected_digest),
        "sampling_backend": "ollama_api_chat",
        "ollama_version": version,
        "model_metadata": metadata,
        "prompt_count": len(prompts),
        "trajectory_count": len(trajectories),
        "samples_per_prompt": args.samples_per_prompt,
        "seed_start": args.seed_start,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "presence_penalty": args.presence_penalty,
        "frequency_penalty": args.frequency_penalty,
        "num_ctx": args.num_ctx,
        "max_new_tokens": args.max_new_tokens,
        "thinking_enabled": False,
        "capture_checkpoints": False,
        "checkpoints": [],
        "layers": [],
        "stat_names": [],
        "total_generated_tokens": total_tokens,
        "generation_seconds": total_wall_seconds,
        "ollama_eval_seconds": total_api_eval_seconds,
        "tokens_per_second": total_tokens / max(total_wall_seconds, 1e-9),
        "ollama_eval_tokens_per_second": total_tokens
        / max(total_api_eval_seconds, 1e-9),
        "screen_min_rate": args.screen_min_rate,
        "screen_max_rate": args.screen_max_rate,
        "eligible_prompt_count": len(eligible_ids),
        "cuda_peak_allocated_gib": None,
        "cuda_peak_reserved_gib": None,
        "prompt_summary": prompt_summary,
    }
    out_dir = args.out_dir / safe_name(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "prompts.jsonl", prompts)
    write_jsonl(out_dir / "eligible_prompts.jsonl", [
        row for row in prompts if str(row["case_id"]) in eligible_ids
    ])
    write_jsonl(out_dir / "trajectories.jsonl", trajectories)
    (out_dir / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_summary(out_dir / "SAMPLING_SUMMARY.md", summary)
    print(f"Wrote Ollama trajectory results to {out_dir}")


if __name__ == "__main__":
    main()
