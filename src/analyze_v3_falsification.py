"""Analyze held-out-template v3 results with strong text-only controls."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_probe_auc import (  # noqa: E402
    average_precision,
    bootstrap_auc_ci,
    fpr_at_tpr,
    roc_auc,
)


BASE_METHODS = [
    "keyword_target_present",
    "keyword_hit_count",
    "logit_lens",
    "dense_jlens",
    "jvp_lens",
    "linear_probe",
    "linear_probe_cross_risk",
]

TEXT_METHODS = [
    "text_tfidf_word",
    "text_tfidf_char",
    "text_tfidf_best",
    "text_tfidf_label_shuffle",
]

LABELS = [
    "semantic_risk",
    "generated_rollback",
    "generated_policy_violation",
    "generated_structural_rollback",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze held-out-template v3 evaluation.")
    parser.add_argument("--case-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--word-features", type=int, default=8000)
    parser.add_argument("--char-features", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def full_text(row: dict[str, Any]) -> str:
    return f"{row.get('system', '')}\n\n{row.get('prompt', '')}".strip()


def word_ngrams(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    grams = [f"w1:{word}" for word in words]
    grams.extend(f"w2:{left}_{right}" for left, right in zip(words, words[1:]))
    return grams


def char_ngrams(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    grams = []
    for width in (3, 4, 5):
        grams.extend(f"c{width}:{normalized[idx:idx + width]}" for idx in range(len(normalized) - width + 1))
    return grams


def build_tfidf(
    texts: list[str],
    train_mask: np.ndarray,
    extractor: Callable[[str], list[str]],
    *,
    max_features: int,
    min_df: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    documents = [extractor(text) for text in texts]
    train_indices = np.flatnonzero(train_mask).tolist()
    df: Counter[str] = Counter()
    for idx in train_indices:
        df.update(set(documents[idx]))
    candidates = [(token, count) for token, count in df.items() if count >= min_df]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected = candidates[:max_features]
    vocab = {token: idx for idx, (token, _count) in enumerate(selected)}
    n_train = max(1, len(train_indices))
    idf = np.ones(len(vocab), dtype=np.float32)
    for token, idx in vocab.items():
        idf[idx] = math.log((1.0 + n_train) / (1.0 + df[token])) + 1.0

    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row_idx, grams in enumerate(documents):
        counts = Counter(gram for gram in grams if gram in vocab)
        for gram, count in counts.items():
            matrix[row_idx, vocab[gram]] = (1.0 + math.log(float(count))) * idf[vocab[gram]]
        norm = float(np.linalg.norm(matrix[row_idx]))
        if norm > 0:
            matrix[row_idx] /= norm
    return matrix, {
        "feature_count": len(vocab),
        "min_df": min_df,
        "max_features": max_features,
        "train_documents": len(train_indices),
    }


def train_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[str],
    *,
    epochs: int,
    device: torch.device,
    shuffled_train_labels: bool = False,
    seed: int = 17,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.manual_seed(seed)
    train_mask = np.asarray([split == "train" for split in splits])
    val_mask = np.asarray([split == "val" for split in splits])
    test_mask = np.asarray([split == "test" for split in splits])
    x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(labels.astype(np.float32)).to(device=device)
    train_idx = torch.from_numpy(np.flatnonzero(train_mask)).to(device=device)
    y_train = y[train_idx].clone()
    if shuffled_train_labels:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        order = torch.randperm(len(y_train), generator=generator, device=device)
        y_train = y_train[order]

    weight = torch.zeros(features.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=0.08, weight_decay=0.01)
    pos = y_train.sum().clamp_min(1.0)
    neg = (len(y_train) - y_train.sum()).clamp_min(1.0)
    pos_weight = (neg / pos).detach()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x[train_idx] @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        scores = (x @ weight + bias).detach().cpu().numpy().astype(np.float64)
    payload = {
        "train_auc": roc_auc(scores[train_mask].tolist(), labels[train_mask].tolist()),
        "val_auc": roc_auc(scores[val_mask].tolist(), labels[val_mask].tolist()),
        "test_auc": roc_auc(scores[test_mask].tolist(), labels[test_mask].tolist()),
        "weight_norm": float(weight.detach().norm().item()),
        "shuffled_train_labels": shuffled_train_labels,
    }
    return scores, payload


def add_text_baselines(
    rows: list[dict[str, Any]],
    *,
    epochs: int,
    word_features: int,
    char_features: int,
    device: torch.device,
) -> dict[str, Any]:
    texts = [full_text(row) for row in rows]
    splits = [str(row["split"]) for row in rows]
    labels = np.asarray([int(bool(row.get("semantic_risk_label"))) for row in rows], dtype=np.int32)
    train_mask = np.asarray([split == "train" for split in splits])
    word_matrix, word_vectorizer = build_tfidf(
        texts,
        train_mask,
        word_ngrams,
        max_features=word_features,
        min_df=2,
    )
    char_matrix, char_vectorizer = build_tfidf(
        texts,
        train_mask,
        char_ngrams,
        max_features=char_features,
        min_df=3,
    )
    word_scores, word_model = train_logistic(word_matrix, labels, splits, epochs=epochs, device=device)
    char_scores, char_model = train_logistic(char_matrix, labels, splits, epochs=epochs, device=device)
    shuffled_scores, shuffled_model = train_logistic(
        word_matrix,
        labels,
        splits,
        epochs=epochs,
        device=device,
        shuffled_train_labels=True,
    )
    best_method = "text_tfidf_word"
    best_scores = word_scores
    if (char_model.get("val_auc") or float("-inf")) > (word_model.get("val_auc") or float("-inf")):
        best_method = "text_tfidf_char"
        best_scores = char_scores
    for idx, row in enumerate(rows):
        for method, scores in [
            ("text_tfidf_word", word_scores),
            ("text_tfidf_char", char_scores),
            ("text_tfidf_best", best_scores),
            ("text_tfidf_label_shuffle", shuffled_scores),
        ]:
            row.setdefault("scores", {})[method] = {
                "score": float(scores[idx]),
                "rank": None,
                "logit": float(scores[idx]),
                "layer": -1,
                "concept": "prompt_text_only",
            }
    return {
        "selected_by_val": best_method,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "word_model": word_model,
        "char_model": char_model,
        "label_shuffle_model": shuffled_model,
    }


def label_value(row: dict[str, Any], label: str) -> int:
    if label == "semantic_risk":
        return int(bool(row.get("semantic_risk_label")))
    if label == "generated_rollback":
        return int(row.get("generated_validator", {}).get("decision") == "rollback")
    if label == "generated_policy_violation":
        return int(row.get("generated_policy_validator", {}).get("decision") == "rollback")
    if label == "generated_structural_rollback":
        return int(row.get("generated_structural_validator", {}).get("decision") == "rollback")
    raise KeyError(label)


def score_value(row: dict[str, Any], method: str) -> float:
    value = row.get("scores", {}).get(method, {}).get("score")
    return float(value) if value is not None else float("nan")


def metric(rows: list[dict[str, Any]], method: str, label: str, samples: int) -> dict[str, Any]:
    finite_rows = [row for row in rows if math.isfinite(score_value(row, method))]
    scores = [score_value(row, method) for row in finite_rows]
    labels = [label_value(row, label) for row in finite_rows]
    pairs = [str(row["pair_id"]) for row in finite_rows]
    templates = [str(row.get("template_family", row["pair_id"])) for row in finite_rows]
    auc = roc_auc(scores, labels)
    return {
        "method": method,
        "label": label,
        "n": len(finite_rows),
        "positive": int(sum(labels)),
        "auc": auc,
        "orientation_free_auc": None if auc is None else max(auc, 1.0 - auc),
        "pair_ci95": bootstrap_auc_ci(scores, labels, samples, clusters=pairs),
        "template_ci95": bootstrap_auc_ci(scores, labels, samples, clusters=templates, seed=19),
        "auprc": average_precision(scores, labels),
        "fpr_at_90_tpr": fpr_at_tpr(scores, labels, 0.9),
    }


def bootstrap_auc_difference(
    rows: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    label: str,
    samples: int,
    seed: int = 23,
) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if math.isfinite(score_value(row, method_a)) and math.isfinite(score_value(row, method_b))
    ]
    by_pair: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(usable):
        by_pair[str(row["pair_id"])].append(idx)
    pairs = sorted(by_pair)
    labels = [label_value(row, label) for row in usable]
    scores_a = [score_value(row, method_a) for row in usable]
    scores_b = [score_value(row, method_b) for row in usable]
    auc_a = roc_auc(scores_a, labels)
    auc_b = roc_auc(scores_b, labels)
    if auc_a is None or auc_b is None or not pairs:
        return {"difference": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(samples):
        sampled = rng.integers(0, len(pairs), size=len(pairs))
        indices = [idx for pair_idx in sampled for idx in by_pair[pairs[int(pair_idx)]]]
        draw_labels = [labels[idx] for idx in indices]
        draw_a = roc_auc([scores_a[idx] for idx in indices], draw_labels)
        draw_b = roc_auc([scores_b[idx] for idx in indices], draw_labels)
        if draw_a is not None and draw_b is not None:
            differences.append(draw_a - draw_b)
    ci = [None, None]
    if differences:
        ci = [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))]
    return {"difference": auc_a - auc_b, "ci95": ci, "auc_a": auc_a, "auc_b": auc_b}


CONDITION_BITS = {
    "concept_absent_target_absent": (0, 0),
    "concept_present_target_absent": (1, 0),
    "target_present_concept_absent": (0, 1),
    "concept_target_present": (1, 1),
}


def bootstrap_mean(values: list[float], samples: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(samples)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def factorial_effects(rows: list[dict[str, Any]], method: str, samples: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[tuple[int, int], float]] = defaultdict(dict)
    for row in rows:
        score = score_value(row, method)
        if not math.isfinite(score) or row.get("condition") not in CONDITION_BITS:
            continue
        grouped[(str(row["risk_type"]), str(row["pair_id"]))][CONDITION_BITS[row["condition"]]] = score
    effects: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"concept": [], "target": [], "interaction": []}
    )
    for (risk, _pair), cells in grouped.items():
        if len(cells) != 4:
            continue
        y00, y10, y01, y11 = cells[(0, 0)], cells[(1, 0)], cells[(0, 1)], cells[(1, 1)]
        effects[risk]["concept"].append(((y10 - y00) + (y11 - y01)) / 2.0)
        effects[risk]["target"].append(((y01 - y00) + (y11 - y10)) / 2.0)
        effects[risk]["interaction"].append((y11 - y10) - (y01 - y00))
    out = []
    for risk, values in sorted(effects.items()):
        for effect_name, effect_values in values.items():
            out.append(
                {
                    "risk_type": risk,
                    "method": method,
                    "effect": effect_name,
                    "pairs": len(effect_values),
                    "mean": float(np.mean(effect_values)) if effect_values else None,
                    "ci95": bootstrap_mean(effect_values, samples, seed=31 + len(out)),
                }
            )
    return out


def validator_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["risk_type"]), str(row["condition"]))].append(row)
    out = []
    for (risk, condition), group in sorted(grouped.items()):
        out.append(
            {
                "risk_type": risk,
                "condition": condition,
                "n": len(group),
                "lexical_rollback": float(np.mean([label_value(row, "generated_rollback") for row in group])),
                "policy_violation": float(
                    np.mean([label_value(row, "generated_policy_violation") for row in group])
                ),
                "structural_rollback": float(
                    np.mean([label_value(row, "generated_structural_rollback") for row in group])
                ),
            }
        )
    return out


def template_metrics(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["risk_type"]), str(row.get("template_family", "unknown")))].append(row)
    out = []
    for (risk, family), group in sorted(grouped.items()):
        for method in methods:
            scores = [score_value(row, method) for row in group]
            if not any(math.isfinite(value) for value in scores):
                continue
            out.append(
                {
                    "risk_type": risk,
                    "template_family": family,
                    "method": method,
                    "auc": roc_auc(scores, [label_value(row, "semantic_risk") for row in group]),
                }
            )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def ci_text(ci: list[float | None]) -> str:
    return "-" if not ci or ci[0] is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    gate = payload["scale_gate"]
    lines = [
        "# Held-Out-Template v3 Falsification Report",
        "",
        "## Bottom Line",
        "",
        f"- template-robust semantic signal: `{gate['template_signal_pass']}`",
        f"- residual added value over text: `{gate['internal_added_value_pass']}`",
        f"- generated-policy prediction gate: `{gate['policy_prediction_pass']}`",
        f"- proceed to local model replication: `{gate['proceed_local_replication']}`",
        f"- proceed to cloud dense scale: `{gate['proceed_cloud_dense_scale']}`",
        "",
        "Cloud scaling remains frozen unless the residual probe survives held-out templates, "
        "beats the strongest prompt-text baseline, and predicts generated policy violations.",
        "",
        "## Test Metrics",
        "",
        "| method | label | n | pos | AUC | pair CI | template CI | orientation-free AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["overall_test_metrics"]:
        lines.append(
            f"| `{row['method']}` | {row['label']} | {row['n']} | {row['positive']} | "
            f"{fmt(row['auc'])} | {ci_text(row['pair_ci95'])} | {ci_text(row['template_ci95'])} | "
            f"{fmt(row['orientation_free_auc'])} |"
        )

    lines.extend(
        [
            "",
            "## Residual vs Text",
            "",
            f"- selected text baseline: `{payload['text_baselines']['selected_by_val']}`",
            f"- residual minus text AUC: `{fmt(payload['residual_minus_text']['difference'])}`",
            f"- paired 95% CI: `{ci_text(payload['residual_minus_text']['ci95'])}`",
            "",
            "## Per-Risk Semantic AUC",
            "",
            "| risk | method | AUC | pair CI | template CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["per_risk_test_metrics"]:
        if row["label"] != "semantic_risk":
            continue
        lines.append(
            f"| {row['risk_type']} | `{row['method']}` | {fmt(row['auc'])} | "
            f"{ci_text(row['pair_ci95'])} | {ci_text(row['template_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Validator Outcomes",
            "",
            "| risk | condition | n | lexical rollback | policy violation | structural rollback |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["validator_rates"]:
        lines.append(
            f"| {row['risk_type']} | `{row['condition']}` | {row['n']} | "
            f"{row['lexical_rollback']:.3f} | {row['policy_violation']:.3f} | "
            f"{row['structural_rollback']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 2x2 Factorial Effects on Test Templates",
            "",
            "Positive concept effects support semantic-risk sensitivity; positive target effects support lexical sensitivity.",
            "",
            "| risk | method | effect | pairs | mean | pair CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["factorial_effects"]:
        lines.append(
            f"| {row['risk_type']} | `{row['method']}` | {row['effect']} | {row['pairs']} | "
            f"{fmt(row['mean'])} | {ci_text(row['ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "- `semantic_risk` is the single constructed primary label; `expected_rollback` is not treated as independent evidence.",
            "- `generated_rollback` is the legacy lexical gate outcome.",
            "- `generated_policy_violation` checks risk-specific unsafe behavior.",
            "- `generated_structural_rollback` additionally rejects malformed or out-of-contract JSON.",
            "- A below-chance AUC is reported with an orientation-free diagnostic, but it fails the preregistered score direction.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.case_scores)
    device = torch.device(args.device)
    text_payload = add_text_baselines(
        rows,
        epochs=args.epochs,
        word_features=args.word_features,
        char_features=args.char_features,
        device=device,
    )
    methods = [
        method
        for method in BASE_METHODS + TEXT_METHODS
        if any(method in row.get("scores", {}) for row in rows)
    ]
    test_rows = [row for row in rows if row.get("split") == "test"]
    overall_test_metrics = [
        metric(test_rows, method, label, args.bootstrap_samples)
        for label in LABELS
        for method in methods
    ]
    per_risk_test_metrics = []
    for risk in sorted({str(row["risk_type"]) for row in test_rows}):
        risk_rows = [row for row in test_rows if row["risk_type"] == risk]
        for label in LABELS:
            for method in methods:
                per_risk_test_metrics.append(
                    {"risk_type": risk, **metric(risk_rows, method, label, args.bootstrap_samples)}
                )

    factorial = []
    for method in methods:
        factorial.extend(factorial_effects(test_rows, method, args.bootstrap_samples))
    residual_minus_text = bootstrap_auc_difference(
        test_rows,
        "linear_probe",
        "text_tfidf_best",
        "semantic_risk",
        args.bootstrap_samples,
    )

    metric_index = {
        (row["method"], row["label"]): row
        for row in overall_test_metrics
    }
    residual_semantic = metric_index.get(("linear_probe", "semantic_risk"), {})
    residual_policy = metric_index.get(("linear_probe", "generated_policy_violation"), {})
    residual_auc = residual_semantic.get("auc")
    residual_ci = residual_semantic.get("pair_ci95", [None, None])
    policy_auc = residual_policy.get("auc")
    policy_ci = residual_policy.get("pair_ci95", [None, None])
    difference = residual_minus_text.get("difference")
    difference_ci = residual_minus_text.get("ci95", [None, None])
    template_signal_pass = bool(
        residual_auc is not None
        and residual_auc >= 0.80
        and residual_ci[0] is not None
        and residual_ci[0] >= 0.65
    )
    internal_added_value_pass = bool(
        difference is not None
        and difference >= 0.03
        and difference_ci[0] is not None
        and difference_ci[0] > 0.0
    )
    policy_prediction_pass = bool(
        policy_auc is not None
        and policy_auc >= 0.75
        and policy_ci[0] is not None
        and policy_ci[0] >= 0.60
    )
    proceed_local = template_signal_pass and internal_added_value_pass and policy_prediction_pass
    scale_gate = {
        "thresholds_frozen_before_v3_run": {
            "semantic_auc": 0.80,
            "semantic_pair_ci_lower": 0.65,
            "residual_minus_text_auc": 0.03,
            "residual_minus_text_ci_lower": 0.0,
            "policy_auc": 0.75,
            "policy_pair_ci_lower": 0.60,
        },
        "template_signal_pass": template_signal_pass,
        "internal_added_value_pass": internal_added_value_pass,
        "policy_prediction_pass": policy_prediction_pass,
        "proceed_local_replication": proceed_local,
        "proceed_cloud_dense_scale": False,
    }
    payload = {
        "source": str(args.case_scores),
        "case_count": len(rows),
        "test_case_count": len(test_rows),
        "methods": methods,
        "text_baselines": text_payload,
        "overall_test_metrics": overall_test_metrics,
        "per_risk_test_metrics": per_risk_test_metrics,
        "template_metrics": template_metrics(test_rows, methods),
        "factorial_effects": factorial,
        "validator_rates": validator_rates(test_rows),
        "residual_minus_text": residual_minus_text,
        "scale_gate": scale_gate,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "v3_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (args.out_dir / "case_scores_with_text.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    write_report(args.out_dir / "V3_FALSIFICATION_REPORT.md", payload)
    print(f"Saved v3 analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
