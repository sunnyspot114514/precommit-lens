"""Minimal pre-commit intervention using a fitted dense J-lens.

This script applies an approximate token-direction suppression at a selected
layer during generation. It is intended as a sanity check, not a final causal
claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / ".vendor-qwen"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_dense_jlens_qwen import (
    apply_chat_template,
    choose_dtype,
    clean_text,
    decode_generation,
    extract_hidden_from_layer_output,
    get_layers,
    load_lens_npz,
    replace_hidden_in_layer_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small pre-commit intervention.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--concept-text", default=" reveal")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--mode", choices=["suppress", "steer"], default="suppress")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def find_case(config: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in config["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(f"Case not found: {case_id}")


def generate_text(model, tokenizer, batch: dict[str, torch.Tensor], max_new_tokens: int) -> str:
    ids = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    return decode_generation(tokenizer, ids, batch["input_ids"].shape[1])


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    case = find_case(cfg, args.case_id)
    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    batch = apply_chat_template(
        tokenizer,
        case["prompt"],
        cfg.get("system_prompt"),
        max_seq_len=args.max_seq_len,
        use_chat_template=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    baseline = generate_text(model, tokenizer, batch, args.max_new_tokens)

    lens, _ = load_lens_npz(args.lens, layers=[args.layer])
    J = lens[args.layer].to(device=device, dtype=torch.float32)
    token_ids = tokenizer.encode(args.concept_text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"concept-text must be one token, got {token_ids}")
    token_id = int(token_ids[0])
    unembed = model.get_output_embeddings().weight[token_id].detach().float().to(device)
    direction = torch.matmul(J.T, unembed)
    direction = direction / direction.norm().clamp_min(1e-6)
    if args.mode == "suppress":
        direction = -direction

    layers = get_layers(model)

    def hook(_module, _inputs, output):
        hidden = extract_hidden_from_layer_output(output)
        patched = hidden.clone()
        scale = patched[:, -1, :].norm(dim=-1, keepdim=True).clamp_min(1e-6) / (patched.shape[-1] ** 0.5)
        patched[:, -1, :] = patched[:, -1, :] + args.alpha * direction.to(patched.dtype) * scale
        return replace_hidden_in_layer_output(output, patched)

    handle = layers[args.layer].register_forward_hook(hook)
    try:
        intervened = generate_text(model, tokenizer, batch, args.max_new_tokens)
    finally:
        handle.remove()

    payload = {
        "case_id": args.case_id,
        "layer": args.layer,
        "concept_text": args.concept_text,
        "concept_token_id": token_id,
        "concept_token": clean_text(tokenizer.decode([token_id])),
        "mode": args.mode,
        "alpha": args.alpha,
        "baseline": baseline,
        "intervened": intervened,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
