from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forbidden concept logit-lens probe")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--rank-threshold", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def get_final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    candidates = [
        ("model", "norm"),
        ("transformer", "ln_f"),
        ("gpt_neox", "final_layer_norm"),
        ("model", "decoder", "final_layer_norm"),
    ]
    for path in candidates:
        obj: Any = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            return obj
    return None


def concept_token_ids(tokenizer: Any, concepts: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for concept in concepts:
        variants = {concept, " " + concept, concept.capitalize(), " " + concept.capitalize()}
        ids: set[int] = set()
        for variant in variants:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            for token_id in encoded:
                # Some SentencePiece tokenizers encode a leading space as its
                # own high-prior token. Counting it would create false concept
                # hits across unrelated words.
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


def clean_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)])
    return text.replace("\n", "\\n")


@dataclass
class LayerReadout:
    layer: int
    top_tokens: list[str]
    best_concept: str
    best_rank: int
    best_logit: float
    concept_ranks: dict[str, int]


def layer_readouts(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    norm: torch.nn.Module | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    concept_ids: dict[str, list[int]],
    top_k: int,
) -> list[LayerReadout]:
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    lm_head = model.get_output_embeddings()
    readouts: list[LayerReadout] = []
    for layer_idx, hidden in enumerate(outputs.hidden_states):
        h = hidden[:, -1, :]
        if norm is not None:
            h = norm(h)
        logits = lm_head(h)[0].float()
        top = torch.topk(logits, k=top_k)
        top_tokens = [clean_token(tokenizer, int(tid)) for tid in top.indices.tolist()]

        best_concept = ""
        best_rank = 10**9
        best_logit = float("-inf")
        concept_ranks: dict[str, int] = {}
        for concept, ids in concept_ids.items():
            if not ids:
                continue
            ids_tensor = torch.tensor(ids, device=logits.device, dtype=torch.long)
            concept_logit = logits.index_select(0, ids_tensor).max()
            rank = int((logits > concept_logit).sum().item()) + 1
            concept_ranks[concept] = rank
            if rank < best_rank:
                best_rank = rank
                best_concept = concept
                best_logit = float(concept_logit.item())

        readouts.append(
            LayerReadout(
                layer=layer_idx,
                top_tokens=top_tokens,
                best_concept=best_concept,
                best_rank=best_rank,
                best_logit=best_logit,
                concept_ranks=concept_ranks,
            )
        )

    # Force a small GPU synchronization point before generation.
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize()
    return readouts


def generate_text(
    *,
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


def validate_output(text: str, banned: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    hits = [term for term in banned if term.lower() in lowered]
    return {
        "accepted": not hits,
        "action": "commit" if not hits else "rollback",
        "hits": hits,
    }


def summarize_layers(readouts: list[LayerReadout], threshold: int) -> dict[str, Any]:
    best = min(readouts, key=lambda row: row.best_rank)
    early_cutoff = max(1, int(len(readouts) * 0.75))
    early = [row for row in readouts[:early_cutoff] if row.best_rank <= threshold]
    concepts = sorted({concept for row in readouts for concept in row.concept_ranks})
    min_rank_by_concept = {
        concept: min(row.concept_ranks[concept] for row in readouts if concept in row.concept_ranks)
        for concept in concepts
    }
    return {
        "num_layers_including_embedding": len(readouts),
        "best_layer": best.layer,
        "best_concept": best.best_concept,
        "best_rank": best.best_rank,
        "min_rank_by_concept": min_rank_by_concept,
        "early_hit_layers": [row.layer for row in early],
        "early_hit": bool(early),
    }


def main() -> int:
    args = parse_args()
    cfg = read_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    local_only = not args.allow_download
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=local_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=local_only,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(args.device)
    model.eval()
    norm = get_final_norm(model)

    concept_ids = concept_token_ids(tokenizer, cfg["concepts"])
    cases = cfg["cases"][: args.limit or None]
    results: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        prompt = format_chat(tokenizer, case["system"], case["user"])
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        watch_concepts = case.get("watch_concepts") or cfg["concepts"]
        watch_ids = {concept: concept_ids[concept] for concept in watch_concepts}

        readouts = layer_readouts(
            model=model,
            tokenizer=tokenizer,
            norm=norm,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            concept_ids=watch_ids,
            top_k=args.top_k,
        )
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=args.max_new_tokens,
        )
        validation = validate_output(generated, cfg["validator_banned"])
        layer_summary = summarize_layers(readouts, args.rank_threshold)

        results.append(
            {
                "case_id": case["id"],
                "group": case["group"],
                "watch_concepts": watch_concepts,
                "prompt_chars": len(prompt),
                "generated": generated,
                "validation": validation,
                "layer_summary": layer_summary,
                "layers": [row.__dict__ for row in readouts],
            }
        )
        print(
            f"[{idx}/{len(cases)}] {case['id']}: "
            f"best={layer_summary['best_concept']}@L{layer_summary['best_layer']} "
            f"rank={layer_summary['best_rank']} "
            f"validator={validation['action']}"
        )

    payload = {
        "model_id": args.model_id,
        "device": args.device,
        "rank_threshold": args.rank_threshold,
        "elapsed_s": round(time.time() - started, 2),
        "concept_token_ids": concept_ids,
        "results": results,
    }
    (args.out_dir / "probe_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(args.out_dir / "probe_summary.md", payload)
    return 0


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Probe Summary",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Device: `{payload['device']}`",
        f"- Rank threshold: `{payload['rank_threshold']}`",
        f"- Elapsed: `{payload['elapsed_s']}s`",
        "",
        "| Case | Group | Best concept | Best layer | Best rank | Early hit | Validator | Hits |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in payload["results"]:
        summary = row["layer_summary"]
        validation = row["validation"]
        hits = ", ".join(validation["hits"]) if validation["hits"] else "-"
        lines.append(
            f"| `{row['case_id']}` | {row['group']} | {summary['best_concept']} | "
            f"{summary['best_layer']} | {summary['best_rank']} | "
            f"{summary['early_hit']} | {validation['action']} | {hits} |"
        )

    lines.extend(["", "## Generated Outputs", ""])
    for row in payload["results"]:
        lines.extend(
            [
                f"### {row['case_id']}",
                "",
                "```text",
                row["generated"] or "(empty)",
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
