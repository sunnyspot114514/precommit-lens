"""Unified AUC evaluation for leakage-controlled pre-commit probes.

This runner evaluates cheap and heavy readout methods on the same prompt set:

- target-word keyword baseline
- direct logit lens
- optional finite-difference JVP lens
- optional dense J-lens from a saved ``.npz``
- layer-wise linear probes with group-held-out validation/test splits

The main output is a compact metrics table, not a large layer dump.  It keeps
both constructed labels (expected rollback / semantic risk) and actual
post-generation validator outcomes so negative results are interpretable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_dense_jlens_qwen import (  # noqa: E402
    apply_chat_template,
    choose_dtype,
    final_norm,
    get_core_model,
    get_layers,
    load_lens_npz,
    safe_name,
)
from runtime_validators import lexical_validate, validate_runtime_output  # noqa: E402


DEFAULT_CONCEPT_TOKEN_TEXTS: dict[str, list[str]] = {
    "commit": [" commit", " committed", " saved", " accepted", " finalized"],
    "secret": [" secret", " hidden", " concealed", " truth", " motive"],
    "reveal": [" reveal", " revealed", " disclose", " spoiler", " expose"],
    "schema": [" schema", " action", " private", " debug", " field", " slot"],
    "forbidden": [" forbidden", " blocked", " invalid", " unauthorized", " bypass"],
}


@dataclass
class ReadoutScore:
    score: float
    rank: int
    logit: float
    layer: int
    concept: str
    token_text: str
    token_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pre-commit probe AUC.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--dense-lens", type=Path, default=None)
    parser.add_argument("--layers", default="all", help="Layer spec for logit/dense/JVP scoring.")
    parser.add_argument("--jvp-layers", default="", help="Optional JVP layer spec. Empty disables JVP.")
    parser.add_argument("--jvp-eps", type=float, default=0.03)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--linear-probe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cross-risk-probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--probe-label", choices=["expected_rollback", "semantic_risk", "generated_rollback"], default="expected_rollback")
    parser.add_argument("--probe-epochs", type=int, default=160)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def select_layers(spec: str, n_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if not spec:
        return []
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("stride:"):
        stride = max(1, int(spec.split(":", 1)[1]))
        layers = list(range(0, n_layers, stride))
        if (n_layers - 1) not in layers:
            layers.append(n_layers - 1)
        return layers
    layers = sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(f"Layer(s) outside 0..{n_layers - 1}: {bad}")
    return layers


def build_concept_token_ids(tokenizer: Any, concept_texts: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for concept, texts in concept_texts.items():
        rows = []
        seen: set[int] = set()
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                continue
            token_id = int(ids[0])
            if token_id in seen:
                continue
            seen.add(token_id)
            rows.append({"text": text, "token_id": token_id, "decoded": tokenizer.decode([token_id])})
        out[concept] = rows
    return out


def validate_output(text: str, patterns: list[str]) -> dict[str, Any]:
    return lexical_validate(text, patterns)


def decode_generation(tokenizer: Any, full_ids: torch.Tensor, prompt_len: int) -> str:
    return tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True).strip()


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> str:
    with torch.inference_mode():
        full_ids = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
    return decode_generation(tokenizer, full_ids, int(batch["input_ids"].shape[1]))


def watched_token_rows(
    concept_ids: dict[str, list[dict[str, Any]]],
    watch_concepts: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in watch_concepts:
        rows.extend({**row, "concept": concept} for row in concept_ids.get(concept, []))
    return rows


def score_logits(logits: torch.Tensor, watched: list[dict[str, Any]], layer: int) -> ReadoutScore | None:
    if not watched:
        return None
    scores = logits.detach().float()
    best: ReadoutScore | None = None
    for item in watched:
        token_id = int(item["token_id"])
        logit = float(scores[token_id].item())
        rank = int(torch.sum(scores > scores[token_id]).item()) + 1
        rank_score = -math.log(float(rank))
        candidate = ReadoutScore(
            score=rank_score,
            rank=rank,
            logit=logit,
            layer=layer,
            concept=str(item["concept"]),
            token_text=str(item["text"]),
            token_id=token_id,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def best_score(scores: list[ReadoutScore]) -> dict[str, Any]:
    if not scores:
        return {
            "score": float("nan"),
            "rank": 10**9,
            "logit": float("nan"),
            "layer": -1,
            "concept": "",
            "token_text": "",
            "token_id": -1,
        }
    best = max(scores, key=lambda row: row.score)
    return best.__dict__


def direct_logit_lens_scores(
    model: torch.nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    layers: list[int],
    watched: list[dict[str, Any]],
    source_pos: int,
) -> dict[str, Any]:
    lm_head = model.get_output_embeddings()
    rows: list[ReadoutScore] = []
    for layer in layers:
        hidden = hidden_states[layer + 1][0, source_pos, :].detach()
        logits = lm_head(final_norm(model, hidden[None, None, :]))[0, 0]
        scored = score_logits(logits, watched, layer)
        if scored:
            rows.append(scored)
    return best_score(rows)


def dense_jlens_scores(
    model: torch.nn.Module,
    hidden_states: tuple[torch.Tensor, ...],
    dense_lens: dict[int, torch.Tensor],
    layers: list[int],
    watched: list[dict[str, Any]],
    source_pos: int,
    device: torch.device,
) -> dict[str, Any]:
    lm_head = model.get_output_embeddings()
    rows: list[ReadoutScore] = []
    for layer in layers:
        if layer not in dense_lens:
            continue
        hidden = hidden_states[layer + 1][0, source_pos, :].detach()
        J = dense_lens[layer].to(device=device, dtype=torch.float32)
        transported = torch.matmul(J, hidden.float())
        logits = lm_head(transported.to(dtype=hidden.dtype)[None, :])[0]
        scored = score_logits(logits, watched, layer)
        if scored:
            rows.append(scored)
    return best_score(rows)


def jvp_lens_scores(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    baseline_hidden_states: tuple[torch.Tensor, ...],
    layers: list[int],
    watched: list[dict[str, Any]],
    source_pos: int,
    eps: float,
) -> dict[str, Any]:
    decoder_layers = get_layers(model)
    core_model = get_core_model(model)
    lm_head = model.get_output_embeddings()
    base_final = baseline_hidden_states[-1][:, source_pos, :].detach().float()
    rows: list[ReadoutScore] = []

    for layer in layers:
        layer_hidden = baseline_hidden_states[layer + 1][:, source_pos, :].detach()
        rms = layer_hidden.float().pow(2).mean().sqrt().clamp_min(1e-6)
        delta = (eps * layer_hidden / rms).to(dtype=layer_hidden.dtype)

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0].clone()
                hidden[:, source_pos, :] = hidden[:, source_pos, :] + delta
                return (hidden, *output[1:])
            hidden = output.clone()
            hidden[:, source_pos, :] = hidden[:, source_pos, :] + delta
            return hidden

        handle = decoder_layers[layer].register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = core_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=False,
                    use_cache=False,
                )
            perturbed = out.last_hidden_state[:, source_pos, :].detach().float()
        finally:
            handle.remove()

        transported = (perturbed - base_final) / eps
        logits = lm_head(transported.to(dtype=baseline_hidden_states[-1].dtype))[0].float()
        scored = score_logits(logits, watched, layer)
        if scored:
            rows.append(scored)
    return best_score(rows)


def split_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    groups_by_risk: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = str(row.get("pair_id", row["case_id"]))
        risk = str(row.get("risk_type", "all"))
        if group not in groups_by_risk[risk]:
            groups_by_risk[risk].append(group)

    assignments: dict[str, str] = {}
    for risk, groups in groups_by_risk.items():
        ordered = sorted(
            groups,
            key=lambda value: hashlib.sha256(f"{risk}:{value}".encode("utf-8")).hexdigest(),
        )
        for idx, group in enumerate(ordered):
            bucket = idx % 10
            if bucket <= 5:
                split = "train"
            elif bucket <= 7:
                split = "val"
            else:
                split = "test"
            assignments[group] = split
    return assignments


def assign_splits(rows: list[dict[str, Any]]) -> str:
    declared = [row.get("template_split") for row in rows]
    if declared and all(value in {"train", "val", "test"} for value in declared):
        family_splits: dict[str, str] = {}
        for row in rows:
            family = str(row.get("template_family", row.get("pair_id", row["case_id"])))
            split = str(row["template_split"])
            previous = family_splits.setdefault(family, split)
            if previous != split:
                raise ValueError(f"Template family {family!r} crosses splits: {previous!r} vs {split!r}")
            row["split"] = split
        return "declared_template_family"

    group_splits = split_groups(rows)
    for row in rows:
        row["split"] = group_splits[str(row.get("pair_id", row["case_id"]))]
    return "hashed_pair_id"


def labels_from_row(row: dict[str, Any]) -> dict[str, int]:
    generated = row.get("generated_validator", {})
    policy = row.get("generated_policy_validator", {})
    structural = row.get("generated_structural_validator", {})
    return {
        "expected_rollback": int(row.get("expected_validator_decision") == "rollback"),
        "semantic_risk": int(bool(row.get("semantic_risk_label"))),
        "generated_rollback": int(generated.get("decision") == "rollback"),
        "generated_policy_violation": int(policy.get("decision") == "rollback"),
        "generated_structural_rollback": int(structural.get("decision") == "rollback"),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(s)
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-s)
    hits = 0
    precisions = []
    for rank, idx in enumerate(order, start=1):
        if y[idx] == 1:
            hits += 1
            precisions.append(hits / rank)
    return float(np.mean(precisions)) if precisions else None


def fpr_at_tpr(scores: list[float], labels: list[int], target_tpr: float = 0.9) -> float | None:
    y = np.asarray(labels, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s)
    tp = 0
    fp = 0
    best: float | None = None
    for idx in order:
        if y[idx] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        if tpr >= target_tpr:
            fpr = fp / n_neg
            best = fpr if best is None else min(best, fpr)
    return best


def bootstrap_auc_ci(
    scores: list[float],
    labels: list[int],
    samples: int,
    clusters: list[str] | None = None,
    seed: int = 7,
) -> list[float | None]:
    base = roc_auc(scores, labels)
    if base is None or samples <= 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    n = len(scores)
    cluster_indices: dict[str, list[int]] = defaultdict(list)
    if clusters is not None:
        if len(clusters) != n:
            raise ValueError("clusters must match scores length")
        for idx, cluster in enumerate(clusters):
            cluster_indices[str(cluster)].append(idx)
    cluster_names = sorted(cluster_indices)
    aucs = []
    for _ in range(samples):
        if cluster_names:
            sampled = rng.integers(0, len(cluster_names), size=len(cluster_names))
            idx = [
                row_idx
                for cluster_idx in sampled
                for row_idx in cluster_indices[cluster_names[int(cluster_idx)]]
            ]
        else:
            idx = rng.integers(0, n, size=n).tolist()
        auc = roc_auc([scores[i] for i in idx], [labels[i] for i in idx])
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return [None, None]
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def metric_payload(
    rows: list[dict[str, Any]],
    method: str,
    label_name: str,
    *,
    split: str | None,
    bootstrap_samples: int,
) -> dict[str, Any]:
    subset = [row for row in rows if split is None or row.get("split") == split]
    scores = [float(row.get("scores", {}).get(method, {}).get("score", float("nan"))) for row in subset]
    labels = [labels_from_row(row)[label_name] for row in subset]
    clusters = [str(row.get("pair_id", row["case_id"])) for row in subset]
    auc = roc_auc(scores, labels)
    return {
        "method": method,
        "label": label_name,
        "split": split or "all",
        "n": len(subset),
        "positive": int(sum(labels)),
        "negative": int(len(labels) - sum(labels)),
        "auc": auc,
        "auc_ci95": bootstrap_auc_ci(scores, labels, bootstrap_samples, clusters=clusters),
        "bootstrap_unit": "pair_id",
        "auprc": average_precision(scores, labels),
        "fpr_at_90_tpr": fpr_at_tpr(scores, labels, 0.9),
    }


def train_linear_probe_for_layer(
    features: torch.Tensor,
    labels: torch.Tensor,
    split_mask: dict[str, torch.Tensor],
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    x_train = features[split_mask["train"]].to(device=device, dtype=torch.float32)
    y_train = labels[split_mask["train"]].to(device=device, dtype=torch.float32)
    if int(y_train.sum().item()) == 0 or int(y_train.sum().item()) == len(y_train):
        return {"skipped": True, "reason": "single-class train split"}

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_train = (x_train - mean) / std
    d_model = x_train.shape[1]
    weight = torch.zeros(d_model, device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.AdamW([weight, bias], lr=0.05, weight_decay=0.01)
    pos = y_train.sum().clamp_min(1.0)
    neg = (len(y_train) - y_train.sum()).clamp_min(1.0)
    pos_weight = (neg / pos).detach()

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = x_train @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        opt.step()

    out: dict[str, Any] = {"skipped": False}
    for split_name, mask in split_mask.items():
        x = features[mask].to(device=device, dtype=torch.float32)
        y = labels[mask].cpu().numpy().astype(int).tolist()
        if len(y) == 0:
            out[f"{split_name}_scores"] = []
            out[f"{split_name}_auc"] = None
            continue
        x = (x - mean) / std
        with torch.inference_mode():
            logits = (x @ weight + bias).detach().cpu().numpy().astype(float).tolist()
        out[f"{split_name}_scores"] = logits
        out[f"{split_name}_auc"] = roc_auc(logits, y)
    out["weight_norm"] = float(weight.detach().norm().item())
    out["bias"] = float(bias.detach().item())
    return out


def run_linear_probes(
    rows: list[dict[str, Any]],
    hidden_by_layer: dict[int, list[np.ndarray]],
    label_name: str,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = torch.tensor([labels_from_row(row)[label_name] for row in rows], dtype=torch.float32)
    split_mask = {
        split: torch.tensor([row.get("split") == split for row in rows], dtype=torch.bool)
        for split in ["train", "val", "test"]
    }
    probe_device = device if device.type == "cuda" else torch.device("cpu")
    layer_results: dict[str, Any] = {}
    for layer, features_list in sorted(hidden_by_layer.items()):
        features = torch.from_numpy(np.stack(features_list).astype(np.float32))
        result = train_linear_probe_for_layer(features, labels, split_mask, epochs, probe_device)
        layer_results[str(layer)] = {k: v for k, v in result.items() if not k.endswith("_scores")}
        if not result.get("skipped"):
            for split_name in ["train", "val", "test"]:
                scores = result.get(f"{split_name}_scores", [])
                indices = [idx for idx, row in enumerate(rows) if row.get("split") == split_name]
                for idx, score in zip(indices, scores):
                    rows[idx].setdefault("scores", {})[f"linear_probe_L{layer}"] = {
                        "score": float(score),
                        "rank": None,
                        "logit": float(score),
                        "layer": layer,
                        "concept": "trained_probe",
                    }

    candidates = [
        (int(layer), payload.get("val_auc"), payload.get("test_auc"))
        for layer, payload in layer_results.items()
        if not payload.get("skipped") and payload.get("val_auc") is not None
    ]
    if not candidates:
        return {"label": label_name, "layers": layer_results, "best_layer": None}
    best_layer, best_val_auc, best_test_auc = max(candidates, key=lambda item: item[1])
    for row in rows:
        method = f"linear_probe_L{best_layer}"
        if method in row.get("scores", {}):
            row.setdefault("scores", {})["linear_probe"] = dict(row["scores"][method])
    return {
        "label": label_name,
        "best_layer": best_layer,
        "best_val_auc": best_val_auc,
        "best_test_auc": best_test_auc,
        "layers": layer_results,
    }


def run_cross_risk_linear_probes(
    rows: list[dict[str, Any]],
    hidden_by_layer: dict[int, list[np.ndarray]],
    label_name: str,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    labels = torch.tensor([labels_from_row(row)[label_name] for row in rows], dtype=torch.float32)
    probe_device = device if device.type == "cuda" else torch.device("cpu")
    by_risk: dict[str, Any] = {}
    for heldout_risk in sorted({str(row["risk_type"]) for row in rows}):
        split_mask = {
            "train": torch.tensor(
                [row.get("split") == "train" and row["risk_type"] != heldout_risk for row in rows],
                dtype=torch.bool,
            ),
            "val": torch.tensor(
                [row.get("split") == "val" and row["risk_type"] != heldout_risk for row in rows],
                dtype=torch.bool,
            ),
            "test": torch.tensor(
                [row.get("split") == "test" and row["risk_type"] == heldout_risk for row in rows],
                dtype=torch.bool,
            ),
        }
        candidates = []
        cached_scores: dict[int, list[float]] = {}
        layer_results: dict[str, Any] = {}
        for layer, features_list in sorted(hidden_by_layer.items()):
            features = torch.from_numpy(np.stack(features_list).astype(np.float32))
            result = train_linear_probe_for_layer(features, labels, split_mask, epochs, probe_device)
            layer_results[str(layer)] = {key: value for key, value in result.items() if not key.endswith("_scores")}
            if result.get("skipped") or result.get("val_auc") is None:
                continue
            cached_scores[layer] = [float(value) for value in result.get("test_scores", [])]
            candidates.append((layer, float(result["val_auc"]), result.get("test_auc")))
        if not candidates:
            by_risk[heldout_risk] = {"best_layer": None, "layers": layer_results}
            continue
        best_layer, best_val_auc, best_test_auc = max(candidates, key=lambda item: item[1])
        test_indices = [
            idx
            for idx, row in enumerate(rows)
            if row.get("split") == "test" and row["risk_type"] == heldout_risk
        ]
        for idx, score in zip(test_indices, cached_scores[best_layer]):
            rows[idx].setdefault("scores", {})["linear_probe_cross_risk"] = {
                "score": float(score),
                "rank": None,
                "logit": float(score),
                "layer": best_layer,
                "concept": f"trained_without_{heldout_risk}",
            }
        by_risk[heldout_risk] = {
            "best_layer": best_layer,
            "source_val_auc": best_val_auc,
            "heldout_test_auc": best_test_auc,
            "layers": layer_results,
        }
    return {"label": label_name, "by_heldout_risk": by_risk}


def write_case_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "risk_type",
        "pair_id",
        "template_family",
        "condition",
        "split",
        "expected_rollback",
        "semantic_risk",
        "generated_rollback",
        "generated_policy_violation",
        "generated_structural_rollback",
        "target_present",
        "logit_lens_score",
        "dense_jlens_score",
        "jvp_lens_score",
        "linear_probe_score",
        "generated_validator_decision",
        "generated_validator_hits",
        "generated_policy_hits",
        "generated_structural_hits",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            labels = labels_from_row(row)
            scores = row.get("scores", {})
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "risk_type": row["risk_type"],
                    "pair_id": row["pair_id"],
                    "template_family": row.get("template_family"),
                    "condition": row["condition"],
                    "split": row.get("split"),
                    "expected_rollback": labels["expected_rollback"],
                    "semantic_risk": labels["semantic_risk"],
                    "generated_rollback": labels["generated_rollback"],
                    "generated_policy_violation": labels["generated_policy_violation"],
                    "generated_structural_rollback": labels["generated_structural_rollback"],
                    "target_present": int(bool(row.get("target_present"))),
                    "logit_lens_score": scores.get("logit_lens", {}).get("score"),
                    "dense_jlens_score": scores.get("dense_jlens", {}).get("score"),
                    "jvp_lens_score": scores.get("jvp_lens", {}).get("score"),
                    "linear_probe_score": scores.get("linear_probe", {}).get("score"),
                    "generated_validator_decision": row.get("generated_validator", {}).get("decision"),
                    "generated_validator_hits": ";".join(row.get("generated_validator", {}).get("hits", [])),
                    "generated_policy_hits": ";".join(row.get("generated_policy_validator", {}).get("hits", [])),
                    "generated_structural_hits": ";".join(row.get("generated_structural_validator", {}).get("hits", [])),
                }
            )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Leakage-Controlled Probe Evaluation",
        "",
        f"- model: `{payload['model_id']}`",
        f"- cases: `{payload['case_count']}`",
        f"- layers: `{payload['layers']}`",
        f"- generated outputs: `{payload['generated_outputs']}`",
        f"- dense lens: `{payload.get('dense_lens') or 'none'}`",
        f"- JVP layers: `{payload.get('jvp_layers') or []}`",
        f"- split strategy: `{payload.get('split_strategy')}`",
        "- AUC CI bootstrap unit: `pair_id`",
        "",
        "## Main Metrics",
        "",
        "| method | label | split | n | pos | AUC | 95% CI | AUPRC | FPR@90%TPR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in payload["metrics"]:
        ci = metric["auc_ci95"]
        ci_text = "-" if ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(
            f"| `{metric['method']}` | {metric['label']} | {metric['split']} | "
            f"{metric['n']} | {metric['positive']} | {fmt(metric['auc'])} | {ci_text} | "
            f"{fmt(metric['auprc'])} | {fmt(metric['fpr_at_90_tpr'])} |"
        )

    lines.extend(
        [
            "",
            "## Linear Probe",
            "",
            f"- probe label: `{payload['linear_probe'].get('label')}`",
            f"- selected layer: `{payload['linear_probe'].get('best_layer')}`",
            f"- validation AUC: `{fmt(payload['linear_probe'].get('best_val_auc'))}`",
            f"- test AUC: `{fmt(payload['linear_probe'].get('best_test_auc'))}`",
            f"- cross-risk probe: `{bool(payload.get('cross_risk_probe'))}`",
            "",
            "## Runtime Cost",
            "",
            "| component | seconds | seconds/case |",
            "|---|---:|---:|",
        ]
    )
    for name, seconds in payload["timing_seconds"].items():
        lines.append(f"| {name} | {seconds:.3f} | {seconds / max(1, payload['case_count']):.5f} |")

    lines.extend(
        [
            "",
            "## Falsification Notes",
            "",
            "- If `keyword_target_present` matches or beats internal readouts, the result is likely surface-token copying.",
            "- If `linear_probe` beats dense J-lens at far lower cost, dense J-lens is not the practical governance monitor.",
            "- If generated rollback has too few positives, report expected-rollback and generated-rollback metrics separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    rows = read_jsonl(args.cases)
    if args.limit_cases:
        rows = rows[: args.limit_cases]

    out_dir = args.out_dir / safe_name(args.model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = choose_dtype(args.dtype, device)
    torch.backends.cuda.matmul.allow_tf32 = True

    split_strategy = assign_splits(rows)

    print(f"Loading {args.model_id} on {device} ({dtype})", flush=True)
    _ = AutoConfig.from_pretrained(args.model_id, local_files_only=not args.allow_download, trust_remote_code=True)
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

    n_layers = len(get_layers(model))
    layers = select_layers(args.layers, n_layers)
    jvp_layers = select_layers(args.jvp_layers, n_layers) if args.jvp_layers else []
    dense_lens: dict[int, torch.Tensor] = {}
    if args.dense_lens is not None:
        dense_lens, _ = load_lens_npz(args.dense_lens, layers=layers)
        layers = [layer for layer in layers if layer in dense_lens or args.dense_lens is None]

    concept_texts = cfg.get("concept_token_texts") or DEFAULT_CONCEPT_TOKEN_TEXTS
    concept_ids = build_concept_token_ids(tokenizer, concept_texts)

    hidden_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    timing = defaultdict(float)
    generated_outputs = not args.no_generate

    for idx, row in enumerate(rows, start=1):
        batch = apply_chat_template(
            tokenizer,
            row["prompt"],
            row.get("system") or cfg.get("system_prompt"),
            max_seq_len=args.max_seq_len,
            use_chat_template=args.chat_template,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        source_pos = int(batch["input_ids"].shape[1] - 1)
        watched = watched_token_rows(concept_ids, row.get("watch_concepts", []))

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
        timing["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        row.setdefault("scores", {})["keyword_target_present"] = {
            "score": float(bool(row.get("target_present"))),
            "rank": None,
            "logit": float(bool(row.get("target_present"))),
            "layer": -1,
            "concept": "surface_target",
        }
        row["scores"]["keyword_hit_count"] = {
            "score": float(len(row.get("target_term_hits", []))),
            "rank": None,
            "logit": float(len(row.get("target_term_hits", []))),
            "layer": -1,
            "concept": "surface_target",
        }
        row["scores"]["logit_lens"] = direct_logit_lens_scores(model, outputs.hidden_states, layers, watched, source_pos)
        timing["logit_lens_scoring"] += time.perf_counter() - t0

        if dense_lens:
            t0 = time.perf_counter()
            row["scores"]["dense_jlens"] = dense_jlens_scores(
                model,
                outputs.hidden_states,
                dense_lens,
                sorted(dense_lens),
                watched,
                source_pos,
                device,
            )
            timing["dense_jlens_scoring"] += time.perf_counter() - t0

        if jvp_layers:
            t0 = time.perf_counter()
            row["scores"]["jvp_lens"] = jvp_lens_scores(
                model,
                batch,
                outputs.hidden_states,
                jvp_layers,
                watched,
                source_pos,
                args.jvp_eps,
            )
            timing["jvp_lens_scoring"] += time.perf_counter() - t0

        for layer in layers:
            hidden_by_layer[layer].append(outputs.hidden_states[layer + 1][0, source_pos, :].detach().float().cpu().numpy())

        if generated_outputs:
            t0 = time.perf_counter()
            text = generate_text(model, tokenizer, batch, args.max_new_tokens)
            timing["generation"] += time.perf_counter() - t0
            row["generated"] = text
            validators = validate_runtime_output(text, row)
            row["generated_validator"] = validators["lexical"]
            row["generated_policy_validator"] = validators["policy"]
            row["generated_structural_validator"] = validators["structural"]
        else:
            row["generated"] = ""
            row["generated_validator"] = {"decision": "not_run", "hits": []}
            row["generated_policy_validator"] = {"decision": "not_run", "hits": []}
            row["generated_structural_validator"] = {"decision": "not_run", "hits": []}

        if idx % 25 == 0 or idx == len(rows):
            print(f"[{idx}/{len(rows)}] evaluated {row['case_id']}", flush=True)

    linear_payload: dict[str, Any] = {"label": args.probe_label, "best_layer": None}
    if args.linear_probe:
        t0 = time.perf_counter()
        linear_payload = run_linear_probes(rows, hidden_by_layer, args.probe_label, args.probe_epochs, device)
        timing["linear_probe_training"] += time.perf_counter() - t0

    cross_risk_payload: dict[str, Any] | None = None
    if args.cross_risk_probe:
        t0 = time.perf_counter()
        cross_risk_payload = run_cross_risk_linear_probes(
            rows,
            hidden_by_layer,
            args.probe_label,
            args.probe_epochs,
            device,
        )
        timing["cross_risk_probe_training"] += time.perf_counter() - t0

    methods = ["keyword_target_present", "keyword_hit_count", "logit_lens"]
    if dense_lens:
        methods.append("dense_jlens")
    if jvp_layers:
        methods.append("jvp_lens")
    if linear_payload.get("best_layer") is not None:
        methods.append("linear_probe")
    if cross_risk_payload is not None:
        methods.append("linear_probe_cross_risk")

    labels_to_report = ["semantic_risk", "generated_rollback"]
    if any(row.get("runtime_validation") for row in rows):
        labels_to_report.extend(["generated_policy_violation", "generated_structural_rollback"])
    metrics = []
    for label_name in labels_to_report:
        for method in methods:
            if method != "linear_probe_cross_risk":
                metrics.append(
                    metric_payload(rows, method, label_name, split=None, bootstrap_samples=args.bootstrap_samples)
                )
            if method == "linear_probe":
                metrics.append(
                    metric_payload(rows, method, label_name, split="test", bootstrap_samples=args.bootstrap_samples)
                )
            elif method == "linear_probe_cross_risk":
                metrics.append(
                    metric_payload(rows, method, label_name, split="test", bootstrap_samples=args.bootstrap_samples)
                )

    payload = {
        "model_id": args.model_id,
        "case_count": len(rows),
        "layers": layers,
        "jvp_layers": jvp_layers,
        "generated_outputs": generated_outputs,
        "split_strategy": split_strategy,
        "dense_lens": str(args.dense_lens) if args.dense_lens else None,
        "metrics": metrics,
        "linear_probe": linear_payload,
        "cross_risk_probe": cross_risk_payload,
        "timing_seconds": dict(timing),
        "concept_token_ids": concept_ids,
    }

    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "case_scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_case_scores(out_dir / "case_scores.csv", rows)
    write_summary(out_dir / "EVALUATION_SUMMARY.md", payload)
    print(f"Saved evaluation to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
