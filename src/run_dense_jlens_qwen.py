"""Dense Jacobian-lens probe for small Hugging Face causal language models.

This script fits full local Jacobian matrices for selected layers:

    J_l = d(final_hidden[last]) / d(layer_hidden_l[last])

The fitted matrix is dense for each selected layer, but the default run is a
smoke test over a small layer subset. The goal is to validate the
runtime-governance path before scaling to all layers or larger models.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".vendor-qwen"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class WatchedToken:
    concept: str
    text: str
    token_id: int
    decoded: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a dense J-lens probe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--load-lens",
        type=Path,
        default=None,
        help="Load a previously fitted .npz lens and only run evaluation.",
    )
    parser.add_argument("--save-lens-name", default="dense_lens_smoke.npz")
    parser.add_argument(
        "--layers",
        default="0,7,14,21,27",
        help="Layer spec: comma list, 'all', or 'stride:N'.",
    )
    parser.add_argument("--limit-fit-prompts", type=int, default=2)
    parser.add_argument("--limit-cases", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--jacobian-chunk", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name).strip("_")


def clean_text(value: str) -> str:
    return value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def choose_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32" or device.type == "cpu":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = [
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return list(obj)
    raise AttributeError("Could not locate decoder layers on model.")


def get_core_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "layers"):
            return language_model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model
    raise AttributeError("Could not locate core transformer model.")


def get_hidden_size(config: Any) -> int:
    hidden_size = int(
        getattr(config, "hidden_size", 0)
        or getattr(getattr(config, "text_config", None), "hidden_size", 0)
    )
    if hidden_size <= 0:
        raise ValueError("Could not determine the text hidden size from model config.")
    return hidden_size


def final_norm(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    core = get_core_model(model)
    if hasattr(core, "norm"):
        return core.norm(hidden)
    return hidden


def select_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("stride:"):
        stride = int(spec.split(":", 1)[1])
        layers = list(range(0, n_layers, stride))
        if (n_layers - 1) not in layers:
            layers.append(n_layers - 1)
        return layers
    layers = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) outside 0..{n_layers - 1}: {bad}")
    return layers


def apply_chat_template(
    tokenizer,
    prompt: str,
    system_prompt: str | None,
    *,
    max_seq_len: int,
    use_chat_template: bool,
) -> dict[str, torch.Tensor]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "truncation": True,
            "max_length": max_seq_len,
        }
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask") if hasattr(encoded, "get") else None
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        input_ids = encoded
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )


def decode_generation(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    generated = full_ids[0, prompt_len:]
    return clean_text(tokenizer.decode(generated, skip_special_tokens=True).strip())


def extract_hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden_in_layer_output(output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def dense_jacobian_for_layer(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    layer_idx: int,
    *,
    source_pos: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (J, layer_hidden_at_source_pos), both CPU float32 tensors."""
    layers = get_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = extract_hidden_from_layer_output(output)
        leaf = hidden.detach().requires_grad_(True)
        captured["hidden"] = leaf
        return replace_hidden_in_layer_output(output, leaf)

    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        outputs = model(
            **batch,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        handle.remove()

    if "hidden" not in captured:
        raise RuntimeError(f"Layer hook did not capture layer {layer_idx}.")

    layer_hidden = captured["hidden"]
    final_hidden = outputs.hidden_states[-1]
    final_vec = final_hidden[0, source_pos, :]
    d_model = final_vec.numel()

    rows: list[torch.Tensor] = []
    eye = torch.eye(d_model, device=final_vec.device, dtype=final_vec.dtype)
    for start in range(0, d_model, chunk_size):
        stop = min(start + chunk_size, d_model)
        grad_outputs = eye[start:stop]
        grads = torch.autograd.grad(
            final_vec,
            layer_hidden,
            grad_outputs=grad_outputs,
            retain_graph=stop < d_model,
            is_grads_batched=True,
            allow_unused=False,
        )[0]
        rows.append(grads[:, 0, source_pos, :].detach().float().cpu())

    J = torch.cat(rows, dim=0).contiguous()
    h = layer_hidden[0, source_pos, :].detach().float().cpu()
    return J, h


def fit_dense_lens(
    model: torch.nn.Module,
    tokenizer,
    fit_prompts: list[str],
    system_prompt: str | None,
    layers: list[int],
    *,
    max_seq_len: int,
    chunk_size: int,
    device: torch.device,
    use_chat_template: bool,
) -> dict[int, torch.Tensor]:
    sums: dict[int, torch.Tensor] = {}
    for prompt_idx, prompt in enumerate(fit_prompts, start=1):
        batch = apply_chat_template(
            tokenizer,
            prompt,
            system_prompt,
            max_seq_len=max_seq_len,
            use_chat_template=use_chat_template,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        source_pos = int(batch["input_ids"].shape[1] - 1)
        for layer_idx in layers:
            t0 = time.perf_counter()
            model.zero_grad(set_to_none=True)
            J, _ = dense_jacobian_for_layer(
                model,
                batch,
                layer_idx,
                source_pos=source_pos,
                chunk_size=chunk_size,
            )
            sums[layer_idx] = J if layer_idx not in sums else sums[layer_idx] + J
            print(
                f"fit prompt {prompt_idx}/{len(fit_prompts)} layer {layer_idx}: "
                f"J={tuple(J.shape)} {time.perf_counter() - t0:.1f}s",
                flush=True,
            )
            del J
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return {layer_idx: value / len(fit_prompts) for layer_idx, value in sums.items()}


def build_watched_tokens(tokenizer, concepts: dict[str, list[str]], selected: list[str]) -> list[WatchedToken]:
    watched: list[WatchedToken] = []
    seen: set[tuple[str, int]] = set()
    for concept in selected:
        for text in concepts.get(concept, []):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                continue
            key = (concept, int(ids[0]))
            if key in seen:
                continue
            seen.add(key)
            watched.append(
                WatchedToken(
                    concept=concept,
                    text=text,
                    token_id=int(ids[0]),
                    decoded=clean_text(tokenizer.decode([int(ids[0])])),
                )
            )
    return watched


def token_ranks(logits: torch.Tensor, watched: list[WatchedToken]) -> list[dict[str, Any]]:
    scores = logits.detach().float()
    out = []
    for item in watched:
        score = scores[item.token_id]
        rank = int(torch.sum(scores > score).item()) + 1
        out.append(
            {
                "concept": item.concept,
                "text": item.text,
                "token_id": item.token_id,
                "decoded": item.decoded,
                "score": float(score.item()),
                "rank": rank,
            }
        )
    return sorted(out, key=lambda x: x["rank"])


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
    values, ids = torch.topk(logits.detach().float(), k=min(k, logits.numel()))
    return [
        {"token_id": int(tok), "token": clean_text(tokenizer.decode([int(tok)])), "score": float(val)}
        for val, tok in zip(values.cpu(), ids.cpu())
    ]


def validate_output(text: str, patterns: list[str]) -> dict[str, Any]:
    low = text.lower()
    hits = [p for p in patterns if p.lower() in low]
    return {"decision": "rollback" if hits else "commit", "hits": hits}


def prompt_term_hits(prompt: str, terms: list[str]) -> list[str]:
    low = prompt.lower()
    return [term for term in terms if term.lower() in low]


@torch.no_grad()
def evaluate_case(
    model: torch.nn.Module,
    tokenizer,
    lens: dict[int, torch.Tensor],
    case: dict[str, Any],
    concepts: dict[str, list[str]],
    system_prompt: str | None,
    *,
    max_seq_len: int,
    max_new_tokens: int,
    device: torch.device,
    top_k: int,
    use_chat_template: bool,
) -> dict[str, Any]:
    batch = apply_chat_template(
        tokenizer,
        case["prompt"],
        system_prompt,
        max_seq_len=max_seq_len,
        use_chat_template=use_chat_template,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    source_pos = int(batch["input_ids"].shape[1] - 1)

    outputs = model(
        **batch,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    generated_ids = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    text = decode_generation(tokenizer, generated_ids, batch["input_ids"].shape[1])
    validator = validate_output(text, case.get("forbidden_output_patterns", []))

    watched = build_watched_tokens(tokenizer, concepts, case.get("watch_concepts", []))
    lm_head = model.get_output_embeddings()
    layer_results = []
    for layer_idx, J_cpu in lens.items():
        hidden = outputs.hidden_states[layer_idx + 1][0, source_pos, :].detach()
        direct_logits = lm_head(final_norm(model, hidden[None, None, :]))[0, 0]

        J = J_cpu.to(device=device, dtype=torch.float32)
        transported = torch.matmul(J, hidden.float())
        j_logits = lm_head(transported.to(dtype=hidden.dtype)[None, :])[0]

        layer_results.append(
            {
                "layer": layer_idx,
                "direct": {
                    "watched": token_ranks(direct_logits, watched),
                    "top": top_tokens(tokenizer, direct_logits, top_k),
                },
                "jlens": {
                    "watched": token_ranks(j_logits, watched),
                    "top": top_tokens(tokenizer, j_logits, top_k),
                },
            }
        )

    best_j = None
    for layer in layer_results:
        watched_rows = layer["jlens"]["watched"]
        if not watched_rows:
            continue
        candidate = dict(watched_rows[0])
        candidate["layer"] = layer["layer"]
        if best_j is None or candidate["rank"] < best_j["rank"]:
            best_j = candidate

    return {
        "id": case["id"],
        "group": case.get("group"),
        "pair": case.get("pair"),
        "prompt_term_hits": prompt_term_hits(case["prompt"], case.get("target_absent_terms", [])),
        "output": text,
        "validator": validator,
        "best_jlens": best_j,
        "layers": layer_results,
    }


def save_lens_npz(path: Path, lens: dict[int, torch.Tensor], metadata: dict[str, Any]) -> None:
    arrays = {f"J_{layer}": J.numpy().astype(np.float16) for layer, J in lens.items()}
    arrays["metadata_json"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez(path, **arrays)


def load_lens_npz(path: Path, layers: list[int] | None = None) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    metadata: dict[str, Any] = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"].item()))
    lens: dict[int, torch.Tensor] = {}
    wanted = set(layers) if layers is not None else None
    for key in data.files:
        if not key.startswith("J_"):
            continue
        layer = int(key.split("_", 1)[1])
        if wanted is not None and layer not in wanted:
            continue
        lens[layer] = torch.from_numpy(data[key].astype(np.float32))
    if not lens:
        raise ValueError(f"No matching J_* matrices found in {path}")
    return dict(sorted(lens.items())), metadata


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Dense J-Lens Probe",
        "",
        f"- model: `{payload['model_id']}`",
        f"- layers: `{payload['layers']}`",
        f"- fit prompts: `{payload['fit_prompt_count']}`",
        f"- dtype: `{payload['dtype']}`",
        f"- device: `{payload['device']}`",
        "",
        "## Case Summary",
        "",
        "| case | group | validator | prompt term hits | best J-lens watched token | output preview |",
        "|---|---|---|---|---|---|",
    ]
    for case in payload["cases"]:
        best = case.get("best_jlens")
        best_text = ""
        if best:
            best_text = (
                f"L{best['layer']} `{best['concept']}:{best['decoded']}` "
                f"rank {best['rank']}"
            )
        preview = re.sub(r"\s+", " ", case["output"]).strip()[:110]
        lines.append(
            "| {id} | {group} | {decision} {hits} | {term_hits} | {best} | {preview} |".format(
                id=case["id"],
                group=case.get("group", ""),
                decision=case["validator"]["decision"],
                hits=",".join(case["validator"]["hits"]),
                term_hits=",".join(case["prompt_term_hits"]),
                best=best_text,
                preview=preview.replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- This run fits actual dense local Jacobian matrices for the selected layers.",
            "- The default smoke run is not yet a full-depth corpus-averaged lens.",
            "- Prompt term hits flag cases where a watched word appears directly in the prompt.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = args.out_dir / safe_name(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Using transformers from: {sys.modules['transformers'].__file__}", flush=True)
    print(f"Loading config for {args.model_id}", flush=True)
    config = AutoConfig.from_pretrained(args.model_id, local_files_only=not args.allow_download, trust_remote_code=True)
    print(f"Config: {type(config).__name__}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=not args.allow_download,
        dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    n_layers = len(get_layers(model))
    layers = select_layers(args.layers, n_layers)
    d_model = get_hidden_size(config)
    print(f"Loaded model: n_layers={n_layers} d_model={d_model} layers={layers}", flush=True)

    t0 = time.perf_counter()
    loaded_lens_metadata: dict[str, Any] = {}
    if args.load_lens is not None:
        lens, loaded_lens_metadata = load_lens_npz(args.load_lens, layers=layers)
        layers = sorted(lens)
        fit_seconds = 0.0
        print(f"Loaded lens from {args.load_lens}: layers={layers}", flush=True)
    else:
        fit_prompts = list(cfg["fit_prompts"])
        if args.limit_fit_prompts:
            fit_prompts = fit_prompts[: args.limit_fit_prompts]
        lens = fit_dense_lens(
            model,
            tokenizer,
            fit_prompts,
            cfg.get("system_prompt"),
            layers,
            max_seq_len=args.max_seq_len,
            chunk_size=args.jacobian_chunk,
            device=device,
            use_chat_template=args.chat_template,
        )
        fit_seconds = time.perf_counter() - t0

    cases = list(cfg["cases"])
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    results = [
        evaluate_case(
            model,
            tokenizer,
            lens,
            case,
            cfg["concepts"],
            cfg.get("system_prompt"),
            max_seq_len=args.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            device=device,
            top_k=args.top_k,
            use_chat_template=args.chat_template,
        )
        for case in cases
    ]

    metadata = {
        "model_id": args.model_id,
        "layers": layers,
        "fit_prompt_count": int(loaded_lens_metadata.get("fit_prompt_count", args.limit_fit_prompts or 0)),
        "d_model": d_model,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "max_seq_len": args.max_seq_len,
        "jacobian_chunk": args.jacobian_chunk,
        "fit_seconds": fit_seconds,
        "loaded_lens": str(args.load_lens) if args.load_lens else None,
    }
    if args.load_lens is None:
        save_lens_npz(out_dir / args.save_lens_name, lens, metadata)
    payload = {**metadata, "cases": results}
    (out_dir / "dense_jlens_results.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    write_summary(out_dir / "dense_jlens_summary.md", payload)
    print(f"Saved results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
