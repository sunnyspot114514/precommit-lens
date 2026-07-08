from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = ["microsoft/Phi-3-mini-4k-instruct"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-difference Jacobian-vector lens probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rank-threshold", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--include-last-layer", action="store_true", default=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def slug_model(model_id: str) -> str:
    return model_id.replace("/", "__").replace(":", "_")


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    for name in ("model", "transformer", "gpt_neox"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError("Could not locate base decoder model")


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    base = get_base_model(model)
    if hasattr(base, "layers"):
        return getattr(base, "layers")
    if hasattr(base, "h"):
        return getattr(base, "h")
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    raise AttributeError("Could not locate decoder layers")


def concept_token_ids(tokenizer: Any, concepts: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for concept in concepts:
        variants = {
            concept,
            " " + concept,
            concept.capitalize(),
            " " + concept.capitalize(),
        }
        ids: set[int] = set()
        for variant in variants:
            for token_id in tokenizer.encode(variant, add_special_tokens=False):
                if tokenizer.decode([int(token_id)]).strip():
                    ids.add(int(token_id))
        out[concept] = sorted(ids)
    return out


def format_chat(tokenizer: Any, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system.strip()}\nUser: {user.strip()}\nAssistant:"


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)]).replace("\n", "\\n")


def score_concepts(logits: torch.Tensor, concept_ids: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
    scores: dict[str, dict[str, float | int]] = {}
    for concept, ids in concept_ids.items():
        if not ids:
            continue
        idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
        concept_logit = logits.index_select(0, idx).max()
        rank = int((logits > concept_logit).sum().item()) + 1
        scores[concept] = {"rank": rank, "logit": float(concept_logit.item())}
    return scores


def best_concept(scores: dict[str, dict[str, float | int]]) -> tuple[str, int, float]:
    if not scores:
        return "", 10**9, float("-inf")
    concept, payload = min(scores.items(), key=lambda item: int(item[1]["rank"]))
    return concept, int(payload["rank"]), float(payload["logit"])


@dataclass
class LensRow:
    layer: int
    best_concept: str
    best_rank: int
    best_logit: float
    concept_scores: dict[str, dict[str, float | int]]
    top_tokens: list[str]


def selected_layers(num_layers: int, stride: int, include_last: bool) -> list[int]:
    layers = list(range(0, num_layers, max(1, stride)))
    if include_last and (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))


def run_base_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Any:
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )


def direct_logit_lens_rows(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
    concept_ids: dict[str, list[int]],
    top_k: int,
) -> list[LensRow]:
    lm_head = model.get_output_embeddings()
    rows: list[LensRow] = []
    for layer_idx in layer_ids:
        # hidden_states[0] is embedding; hidden_states[layer_idx + 1] is after decoder layer.
        h = hidden_states[layer_idx + 1][:, -1, :]
        with torch.inference_mode():
            logits = lm_head(h)[0].float()
        scores = score_concepts(logits, concept_ids)
        concept, rank, logit = best_concept(scores)
        top = torch.topk(logits, k=top_k).indices.tolist()
        rows.append(
            LensRow(
                layer=layer_idx,
                best_concept=concept,
                best_rank=rank,
                best_logit=logit,
                concept_scores=scores,
                top_tokens=[decode_token(tokenizer, tid) for tid in top],
            )
        )
    return rows


def jacobian_vector_rows(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_hidden_states: tuple[torch.Tensor, ...],
    layer_ids: list[int],
    concept_ids: dict[str, list[int]],
    top_k: int,
    eps: float,
) -> list[LensRow]:
    base_model = get_base_model(model)
    layers = get_decoder_layers(model)
    lm_head = model.get_output_embeddings()
    base_final = baseline_hidden_states[-1][:, -1, :].detach().float()
    rows: list[LensRow] = []

    for layer_idx in layer_ids:
        layer_hidden = baseline_hidden_states[layer_idx + 1][:, -1, :].detach()
        rms = layer_hidden.float().pow(2).mean().sqrt().clamp_min(1e-6)
        direction = (layer_hidden / rms).to(dtype=layer_hidden.dtype)
        delta = (eps * direction).detach()

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, -1, :] = hidden[:, -1, :] + delta
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + delta
            return hidden

        handle = layers[layer_idx].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                )
            perturbed_final = out.last_hidden_state[:, -1, :].detach().float()
        finally:
            handle.remove()

        transported = (perturbed_final - base_final) / eps
        with torch.inference_mode():
            logits = lm_head(transported.to(dtype=baseline_hidden_states[-1].dtype))[0].float()
        scores = score_concepts(logits, concept_ids)
        concept, rank, logit = best_concept(scores)
        top = torch.topk(logits, k=top_k).indices.tolist()
        rows.append(
            LensRow(
                layer=layer_idx,
                best_concept=concept,
                best_rank=rank,
                best_logit=logit,
                concept_scores=scores,
                top_tokens=[decode_token(tokenizer, tid) for tid in top],
            )
        )
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize()
    return rows


def validate_output(text: str, banned: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    hits = [term for term in banned if term.lower() in lowered]
    return {
        "accepted": not hits,
        "action": "commit" if not hits else "rollback",
        "hits": hits,
    }


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> str:
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def summarize_rows(rows: list[LensRow], threshold: int) -> dict[str, Any]:
    best = min(rows, key=lambda row: row.best_rank)
    concepts = sorted({concept for row in rows for concept in row.concept_scores})
    min_rank_by_concept = {
        concept: min(int(row.concept_scores[concept]["rank"]) for row in rows if concept in row.concept_scores)
        for concept in concepts
    }
    hit_rows = [row for row in rows if row.best_rank <= threshold]
    return {
        "best_layer": best.layer,
        "best_concept": best.best_concept,
        "best_rank": best.best_rank,
        "hit_layers": [row.layer for row in hit_rows],
        "hit": bool(hit_rows),
        "min_rank_by_concept": min_rank_by_concept,
    }


def load_model(model_id: str, device: str, local_files_only: bool) -> tuple[Any, torch.nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        "low_cpu_mem_usage": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return tokenizer, model


def run_model(args: argparse.Namespace, cfg: dict[str, Any], model_id: str) -> dict[str, Any]:
    started = time.time()
    local_only = not args.allow_download
    tokenizer, model = load_model(model_id, args.device, local_only)
    layers = get_decoder_layers(model)
    layer_ids = selected_layers(len(layers), args.layer_stride, args.include_last_layer)
    concept_ids_all = concept_token_ids(tokenizer, cfg["concepts"])
    cases = cfg["cases"][: args.limit_cases or None]
    results: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        prompt = format_chat(tokenizer, case["system"], case["user"])
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        watch = case.get("watch_concepts") or cfg["concepts"]
        concept_ids = {concept: concept_ids_all[concept] for concept in watch}

        base_outputs = run_base_forward(model, encoded["input_ids"], encoded["attention_mask"])
        direct_rows = direct_logit_lens_rows(
            model=model,
            tokenizer=tokenizer,
            hidden_states=base_outputs.hidden_states,
            layer_ids=layer_ids,
            concept_ids=concept_ids,
            top_k=args.top_k,
        )
        jv_rows = jacobian_vector_rows(
            model=model,
            tokenizer=tokenizer,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            baseline_hidden_states=base_outputs.hidden_states,
            layer_ids=layer_ids,
            concept_ids=concept_ids,
            top_k=args.top_k,
            eps=args.eps,
        )
        generated = generate_text(
            model,
            tokenizer,
            encoded["input_ids"],
            encoded["attention_mask"],
            args.max_new_tokens,
        )
        validation = validate_output(generated, cfg["validator_banned"])
        jv_summary = summarize_rows(jv_rows, args.rank_threshold)
        direct_summary = summarize_rows(direct_rows, args.rank_threshold)

        results.append(
            {
                "case_id": case["id"],
                "group": case["group"],
                "watch_concepts": watch,
                "prompt_chars": len(prompt),
                "generated": generated,
                "validation": validation,
                "jacobian_vector_summary": jv_summary,
                "direct_logit_lens_summary": direct_summary,
                "jacobian_vector_layers": [row.__dict__ for row in jv_rows],
                "direct_logit_lens_layers": [row.__dict__ for row in direct_rows],
            }
        )
        print(
            f"[{model_id} {idx}/{len(cases)}] {case['id']}: "
            f"J={jv_summary['best_concept']}@L{jv_summary['best_layer']} r{jv_summary['best_rank']} "
            f"D={direct_summary['best_concept']}@L{direct_summary['best_layer']} r{direct_summary['best_rank']} "
            f"validator={validation['action']}"
        )

    payload = {
        "model_id": model_id,
        "device": args.device,
        "num_decoder_layers": len(layers),
        "sampled_layers": layer_ids,
        "eps": args.eps,
        "rank_threshold": args.rank_threshold,
        "elapsed_s": round(time.time() - started, 2),
        "results": results,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def write_model_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Jacobian-Vector Probe: {payload['model_id']}",
        "",
        f"- Device: `{payload['device']}`",
        f"- Decoder layers: `{payload['num_decoder_layers']}`",
        f"- Sampled layers: `{payload['sampled_layers']}`",
        f"- Finite-difference eps: `{payload['eps']}`",
        f"- Elapsed: `{payload['elapsed_s']}s`",
        "",
        "| Case | Group | J best | J layer | J rank | Direct best | Direct layer | Direct rank | Validator | Hits |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in payload["results"]:
        jv = row["jacobian_vector_summary"]
        direct = row["direct_logit_lens_summary"]
        validation = row["validation"]
        hits = ", ".join(validation["hits"]) if validation["hits"] else "-"
        lines.append(
            f"| `{row['case_id']}` | {row['group']} | {jv['best_concept']} | "
            f"{jv['best_layer']} | {jv['best_rank']} | {direct['best_concept']} | "
            f"{direct['best_layer']} | {direct['best_rank']} | {validation['action']} | {hits} |"
        )
    lines.extend(["", "## Generated Outputs", ""])
    for row in payload["results"]:
        lines.extend(["### " + row["case_id"], "", "```text", row["generated"] or "(empty)", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cross_model_summary(path: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [
        "# Cross-Model Jacobian-Vector Probe Summary",
        "",
        "This is a finite-difference Jacobian-vector lens, not a fitted full Jacobian lens.",
        "",
        "| Model | Case | J best | J layer | J rank | Direct best | Direct rank | Validator |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for payload in payloads:
        for row in payload["results"]:
            jv = row["jacobian_vector_summary"]
            direct = row["direct_logit_lens_summary"]
            lines.append(
                f"| `{payload['model_id']}` | `{row['case_id']}` | {jv['best_concept']} | "
                f"{jv['best_layer']} | {jv['best_rank']} | {direct['best_concept']} | "
                f"{direct['best_rank']} | {row['validation']['action']} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = read_yaml(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    payloads: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_dir = args.out_dir / slug_model(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = run_model(args, cfg, model_id)
        payloads.append(payload)
        (model_dir / "jacobian_vector_results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_model_summary(model_dir / "jacobian_vector_summary.md", payload)
    write_cross_model_summary(args.out_dir / "jacobian_vector_cross_model_summary.md", payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

