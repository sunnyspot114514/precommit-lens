"""Analyze the pre-registered v4 within-prompt trajectory experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_v3_falsification import build_tfidf, char_ngrams, word_ngrams  # noqa: E402


PRIMARY_CHECKPOINTS = [2, 4, 6, 8, 10, 12]
PRIMARY_LAYER = 18
BOOTSTRAP_SEED = 20260711


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confirmatory v4 trajectories.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/trajectory_v4_confirmatory/Qwen__Qwen3-0.6B"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/prompt_sets/trajectory_confirmatory_v4.jsonl"),
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--word-features", type=int, default=8000)
    parser.add_argument("--char-features", type=int, default=10000)
    parser.add_argument("--judge-scores", type=Path, default=None)
    parser.add_argument("--primary-layer", type=int, default=PRIMARY_LAYER)
    parser.add_argument(
        "--primary-checkpoints",
        default=",".join(str(value) for value in PRIMARY_CHECKPOINTS),
    )
    parser.add_argument("--expected-test-prompts", type=int, default=9)
    parser.add_argument("--experiment-label", default="v4")
    parser.add_argument("--analysis-name", default="v4_analysis.json")
    parser.add_argument("--scores-name", default="trajectory_scores_v4.jsonl")
    parser.add_argument("--report-name", default="V4_CONFIRMATORY_RESULTS.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prompt_mixed_ids(
    labels: np.ndarray,
    prompts: np.ndarray,
    splits: np.ndarray,
    mask: np.ndarray,
    split: str,
) -> set[str]:
    out: set[str] = set()
    for prompt in sorted(set(prompts[(splits == split) & mask].tolist())):
        idx = (prompts == prompt) & (splits == split) & mask
        if len(set(labels[idx].tolist())) == 2:
            out.add(str(prompt))
    return out


def balanced_prompt_weights(
    labels: np.ndarray,
    prompts: np.ndarray,
    splits: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mixed = sorted(prompt_mixed_ids(labels, prompts, splits, mask, "train"))
    fit = np.zeros(len(labels), dtype=np.bool_)
    weights = np.zeros(len(labels), dtype=np.float32)
    for prompt in mixed:
        prompt_idx = np.flatnonzero((prompts == prompt) & (splits == "train") & mask)
        for label in (0, 1):
            class_idx = prompt_idx[labels[prompt_idx] == label]
            if len(class_idx):
                fit[class_idx] = True
                weights[class_idx] = 0.5 / (len(mixed) * len(class_idx))
    return fit, weights, mixed


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    prompts: np.ndarray,
    splits: np.ndarray,
    available: np.ndarray,
    *,
    standardize: bool,
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_mask, sample_weights, mixed = balanced_prompt_weights(
        labels, prompts, splits, available
    )
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    if fit_mask.sum() == 0 or len(set(labels[fit_mask].tolist())) < 2:
        return scores, {"status": "insufficient_training_contrast", "mixed_train_prompts": len(mixed)}

    x = np.asarray(features, dtype=np.float32)
    finite = np.all(np.isfinite(x), axis=1)
    fit_mask &= finite
    score_mask = available & finite
    if fit_mask.sum() == 0:
        return scores, {"status": "no_finite_training_features", "mixed_train_prompts": len(mixed)}

    if standardize:
        weight = sample_weights[fit_mask].astype(np.float64)
        mean = np.average(x[fit_mask], axis=0, weights=weight).astype(np.float32)
        variance = np.average((x[fit_mask] - mean) ** 2, axis=0, weights=weight)
        scale = np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
        x = (x - mean) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    torch.manual_seed(seed)
    x_tensor = torch.from_numpy(x).to(device=device)
    y_tensor = torch.from_numpy(labels.astype(np.float32)).to(device=device)
    w_tensor = torch.from_numpy(sample_weights).to(device=device)
    fit_idx = torch.from_numpy(np.flatnonzero(fit_mask)).to(device=device)
    weight = torch.zeros(x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=0.05, weight_decay=0.01)
    started = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor[fit_idx] @ weight + bias
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tensor[fit_idx], reduction="none"
        )
        active_weights = w_tensor[fit_idx]
        loss = (losses * active_weights).sum() / active_weights.sum().clamp_min(1e-12)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    with torch.inference_mode():
        warmup = x_tensor @ weight + bias
        del warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        score_started = time.perf_counter()
        repeats = 20
        for _ in range(repeats):
            logits_all = x_tensor @ weight + bias
        if device.type == "cuda":
            torch.cuda.synchronize()
        scoring_seconds = (time.perf_counter() - score_started) / repeats
        values = logits_all.detach().cpu().numpy().astype(np.float64)
    scores[score_mask] = values[score_mask]
    return scores, {
        "status": "fit",
        "mixed_train_prompts": len(mixed),
        "training_rows": int(fit_mask.sum()),
        "feature_count": int(x.shape[1]),
        "epochs": epochs,
        "training_seconds": training_seconds,
        "scoring_seconds_all_rows": scoring_seconds,
        "scoring_microseconds_per_row": 1e6 * scoring_seconds / len(labels),
        "weight_norm": float(weight.detach().norm().item()),
    }


def prompt_auc_map(
    scores: np.ndarray,
    labels: np.ndarray,
    prompts: np.ndarray,
    risks: np.ndarray,
    splits: np.ndarray,
    available: np.ndarray,
    split: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    usable = available & np.isfinite(scores) & (splits == split)
    for prompt in sorted(set(prompts[usable].tolist())):
        idx = np.flatnonzero(usable & (prompts == prompt))
        positive = scores[idx][labels[idx] == 1]
        negative = scores[idx][labels[idx] == 0]
        if not len(positive) or not len(negative):
            continue
        delta = positive[:, None] - negative[None, :]
        auc = float((np.sum(delta > 0) + 0.5 * np.sum(np.isclose(delta, 0.0))).item() / delta.size)
        result[str(prompt)] = {
            "auc": auc,
            "risk_type": str(risks[idx[0]]),
            "positive": int(len(positive)),
            "negative": int(len(negative)),
            "pairs": int(delta.size),
        }
    return result


def percentile_ci(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize_prompt_auc(
    prompt_values: dict[str, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    prompt_ids = sorted(prompt_values)
    if not prompt_ids:
        return {"auc": None, "ci95": [None, None], "prompt_count": 0, "risk_count": 0}
    values = np.asarray([prompt_values[prompt]["auc"] for prompt in prompt_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(samples)]
    return {
        "auc": float(values.mean()),
        "ci95": percentile_ci(boot),
        "prompt_count": len(prompt_ids),
        "risk_count": len({prompt_values[prompt]["risk_type"] for prompt in prompt_ids}),
        "pair_count": sum(int(prompt_values[prompt]["pairs"]) for prompt in prompt_ids),
    }


def paired_envelope_difference(
    residual: dict[str, dict[str, Any]],
    text: dict[str, dict[str, Any]],
    stats: dict[str, dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    prompt_ids = sorted(set(residual) & set(text) & set(stats))
    if not prompt_ids:
        return {
            "delta": None,
            "ci95": [None, None],
            "baseline": None,
            "prompt_count": 0,
            "risk_count": 0,
        }
    r = np.asarray([residual[prompt]["auc"] for prompt in prompt_ids])
    t = np.asarray([text[prompt]["auc"] for prompt in prompt_ids])
    s = np.asarray([stats[prompt]["auc"] for prompt in prompt_ids])
    t_mean = float(t.mean())
    s_mean = float(s.mean())
    baseline = "visible_prefix_tfidf" if t_mean >= s_mean else "next_token_stats"
    point = float(r.mean() - max(t_mean, s_mean))
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(samples):
        selected = rng.integers(0, len(prompt_ids), len(prompt_ids))
        boot.append(float(r[selected].mean() - max(t[selected].mean(), s[selected].mean())))
    return {
        "delta": point,
        "ci95": percentile_ci(boot),
        "baseline": baseline,
        "baseline_auc": max(t_mean, s_mean),
        "prompt_count": len(prompt_ids),
        "risk_count": len({residual[prompt]["risk_type"] for prompt in prompt_ids}),
    }


def split_yield(
    labels: np.ndarray,
    prompts: np.ndarray,
    risks: np.ndarray,
    splits: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        prompt_ids = sorted(set(prompts[splits == split].tolist()))
        rows = []
        for prompt in prompt_ids:
            idx = (prompts == prompt) & (splits == split)
            violations = int(labels[idx].sum())
            total = int(idx.sum())
            rows.append(
                {
                    "case_id": prompt,
                    "risk_type": str(risks[np.flatnonzero(idx)[0]]),
                    "n": total,
                    "violations": violations,
                    "violation_rate": violations / total,
                    "mixed": 0 < violations < total,
                }
            )
        mixed = [row for row in rows if row["mixed"]]
        output[split] = {
            "prompt_count": len(rows),
            "mixed_prompt_count": len(mixed),
            "mixed_risk_count": len({row["risk_type"] for row in mixed}),
            "by_risk": {
                risk: {
                    "prompts": sum(row["risk_type"] == risk for row in rows),
                    "mixed": sum(row["risk_type"] == risk and row["mixed"] for row in rows),
                }
                for risk in sorted({row["risk_type"] for row in rows})
            },
            "prompts": rows,
        }
    return output


def load_judge_scores(path: Path | None) -> dict[tuple[str, int], float]:
    if path is None or not path.exists():
        return {}
    return {
        (str(row["trajectory_id"]), int(row["checkpoint"])): float(row["score"])
        for row in read_jsonl(path)
    }


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    primary_checkpoints = [
        int(value.strip()) for value in args.primary_checkpoints.split(",") if value.strip()
    ]
    trajectories = read_jsonl(args.run_dir / "trajectories.jsonl")
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    cache = np.load(args.run_dir / "trajectory_features.npz")
    cached_ids = [str(value) for value in cache["trajectory_ids"].tolist()]
    by_id = {str(row["trajectory_id"]): row for row in trajectories}
    if set(cached_ids) != set(by_id):
        raise ValueError("Trajectory JSONL and feature cache ids differ")
    trajectories = [by_id[trajectory_id] for trajectory_id in cached_ids]

    features = cache["features"].astype(np.float32)
    stats = cache["stats"].astype(np.float32)
    valid = cache["valid"].astype(np.bool_)
    checkpoints = [int(value) for value in cache["checkpoints"].tolist()]
    layers = [int(value) for value in cache["layers"].tolist()]
    if args.primary_layer not in layers:
        raise ValueError(f"Primary layer {args.primary_layer} is not in captured layers {layers}")
    if any(checkpoint not in checkpoints for checkpoint in primary_checkpoints):
        raise ValueError("Every primary checkpoint must be present in the feature cache")
    labels = np.asarray(
        [row["policy_validator"]["decision"] == "rollback" for row in trajectories], dtype=np.int32
    )
    prompts = np.asarray([str(row["case_id"]) for row in trajectories])
    risks = np.asarray([str(row["risk_type"]) for row in trajectories])
    splits = np.asarray([str(cases[prompt]["trajectory_split"]) for prompt in prompts])
    landings = np.asarray(
        [int(row["policy_landing_token"]) if row.get("policy_landing_token") is not None else -1 for row in trajectories]
    )
    trajectory_ids = np.asarray(cached_ids)
    judge_scores = load_judge_scores(args.judge_scores)
    judge_summary_path = args.run_dir / "prefix_judge_summary.json"
    judge_summary = (
        json.loads(judge_summary_path.read_text(encoding="utf-8"))
        if judge_summary_path.exists()
        else None
    )
    monitoring_cost_path = args.run_dir / "monitoring_cost_benchmark.json"
    monitoring_cost = (
        json.loads(monitoring_cost_path.read_text(encoding="utf-8"))
        if monitoring_cost_path.exists()
        else None
    )
    device = torch.device(args.device)

    yield_summary = split_yield(labels, prompts, risks, splits)
    checkpoint_results: list[dict[str, Any]] = []
    trajectory_score_rows: list[dict[str, Any]] = [
        {
            "trajectory_id": row["trajectory_id"],
            "case_id": row["case_id"],
            "risk_type": row["risk_type"],
            "trajectory_split": str(cases[str(row["case_id"])]["trajectory_split"]),
            "eventual_policy_violation": bool(labels[idx]),
            "policy_landing_token": None if landings[idx] < 0 else int(landings[idx]),
            "scores": {},
        }
        for idx, row in enumerate(trajectories)
    ]
    total_training_seconds = 0.0
    scoring_costs: dict[str, list[float]] = defaultdict(list)

    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        available = valid[:, checkpoint_idx] & ((landings < 0) | (checkpoint < landings))
        method_scores: dict[str, np.ndarray] = {}
        model_payloads: dict[str, Any] = {}

        for layer_idx, layer in enumerate(layers):
            method = f"residual_layer_{layer}"
            scores, payload = fit_logistic(
                features[:, checkpoint_idx, layer_idx, :],
                labels,
                prompts,
                splits,
                available,
                standardize=True,
                epochs=args.epochs,
                device=device,
                seed=1000 + checkpoint * 31 + layer,
            )
            method_scores[method] = scores
            model_payloads[method] = payload
            total_training_seconds += float(payload.get("training_seconds", 0.0))
            if payload.get("scoring_microseconds_per_row") is not None:
                scoring_costs[method].append(float(payload["scoring_microseconds_per_row"]))

        stat_scores, stat_payload = fit_logistic(
            stats[:, checkpoint_idx, :],
            labels,
            prompts,
            splits,
            available,
            standardize=True,
            epochs=args.epochs,
            device=device,
            seed=2000 + checkpoint,
        )
        method_scores["next_token_stats"] = stat_scores
        model_payloads["next_token_stats"] = stat_payload
        total_training_seconds += float(stat_payload.get("training_seconds", 0.0))
        if stat_payload.get("scoring_microseconds_per_row") is not None:
            scoring_costs["next_token_stats"].append(
                float(stat_payload["scoring_microseconds_per_row"])
            )

        texts = [
            f"{cases[str(row['case_id'])].get('system', '')}\n\n"
            f"{cases[str(row['case_id'])].get('prompt', '')}\n\n"
            f"PARTIAL RESPONSE:\n{row['checkpoint_prefixes'][checkpoint_idx]}"
            for row in trajectories
        ]
        train_mask = (splits == "train") & available
        vector_started = time.perf_counter()
        word_matrix, word_meta = build_tfidf(
            texts, train_mask, word_ngrams, max_features=args.word_features, min_df=2
        )
        word_vector_seconds = time.perf_counter() - vector_started
        word_scores, word_payload = fit_logistic(
            word_matrix,
            labels,
            prompts,
            splits,
            available,
            standardize=False,
            epochs=args.epochs,
            device=device,
            seed=3000 + checkpoint,
        )
        vector_started = time.perf_counter()
        char_matrix, char_meta = build_tfidf(
            texts, train_mask, char_ngrams, max_features=args.char_features, min_df=3
        )
        char_vector_seconds = time.perf_counter() - vector_started
        char_scores, char_payload = fit_logistic(
            char_matrix,
            labels,
            prompts,
            splits,
            available,
            standardize=False,
            epochs=args.epochs,
            device=device,
            seed=4000 + checkpoint,
        )
        method_scores["visible_prefix_tfidf_word"] = word_scores
        method_scores["visible_prefix_tfidf_char"] = char_scores
        word_val = prompt_auc_map(
            word_scores, labels, prompts, risks, splits, available, "validation"
        )
        char_val = prompt_auc_map(
            char_scores, labels, prompts, risks, splits, available, "validation"
        )
        word_val_auc = summarize_prompt_auc(
            word_val, samples=args.bootstrap_samples, seed=BOOTSTRAP_SEED + checkpoint
        )["auc"]
        char_val_auc = summarize_prompt_auc(
            char_val, samples=args.bootstrap_samples, seed=BOOTSTRAP_SEED + checkpoint + 1
        )["auc"]
        selected_text = (
            "visible_prefix_tfidf_char"
            if (char_val_auc if char_val_auc is not None else -math.inf)
            > (word_val_auc if word_val_auc is not None else -math.inf)
            else "visible_prefix_tfidf_word"
        )
        method_scores["visible_prefix_tfidf"] = method_scores[selected_text]
        model_payloads.update(
            {
                "visible_prefix_tfidf_word": {
                    **word_payload,
                    "vectorizer": word_meta,
                    "vectorization_seconds": word_vector_seconds,
                },
                "visible_prefix_tfidf_char": {
                    **char_payload,
                    "vectorizer": char_meta,
                    "vectorization_seconds": char_vector_seconds,
                },
                "visible_prefix_tfidf": {
                    "selected_by_validation": selected_text,
                    "word_validation_auc": word_val_auc,
                    "char_validation_auc": char_val_auc,
                },
            }
        )
        total_training_seconds += float(word_payload.get("training_seconds", 0.0))
        total_training_seconds += float(char_payload.get("training_seconds", 0.0))
        if word_payload.get("scoring_microseconds_per_row") is not None:
            scoring_costs["visible_prefix_tfidf_word"].append(
                float(word_payload["scoring_microseconds_per_row"])
            )
        if char_payload.get("scoring_microseconds_per_row") is not None:
            scoring_costs["visible_prefix_tfidf_char"].append(
                float(char_payload["scoring_microseconds_per_row"])
            )

        if judge_scores:
            values = np.full(len(trajectories), np.nan, dtype=np.float64)
            for idx, trajectory_id in enumerate(trajectory_ids):
                value = judge_scores.get((str(trajectory_id), checkpoint))
                if value is not None:
                    values[idx] = value
            method_scores["prefix_model_judge"] = values

        methods_to_report = [
            f"residual_layer_{args.primary_layer}",
            "visible_prefix_tfidf",
            "next_token_stats",
        ]
        if "prefix_model_judge" in method_scores:
            methods_to_report.append("prefix_model_judge")
        test_maps: dict[str, dict[str, dict[str, Any]]] = {}
        metrics: dict[str, Any] = {}
        for method in methods_to_report:
            test_map = prompt_auc_map(
                method_scores[method], labels, prompts, risks, splits, available, "test"
            )
            test_maps[method] = test_map
            metrics[method] = summarize_prompt_auc(
                test_map,
                samples=args.bootstrap_samples,
                seed=BOOTSTRAP_SEED + checkpoint * 17 + len(metrics),
            )
            metrics[method]["by_risk"] = {
                risk: summarize_prompt_auc(
                    {prompt: value for prompt, value in test_map.items() if value["risk_type"] == risk},
                    samples=args.bootstrap_samples,
                    seed=BOOTSTRAP_SEED + checkpoint * 23 + risk_idx,
                )
                for risk_idx, risk in enumerate(sorted(set(risks[splits == "test"].tolist())))
            }

        comparison = paired_envelope_difference(
            test_maps[f"residual_layer_{args.primary_layer}"],
            test_maps["visible_prefix_tfidf"],
            test_maps["next_token_stats"],
            samples=args.bootstrap_samples,
            seed=BOOTSTRAP_SEED + checkpoint * 101,
        )
        layer_metrics = {}
        for layer in layers:
            method = f"residual_layer_{layer}"
            layer_map = prompt_auc_map(
                method_scores[method], labels, prompts, risks, splits, available, "test"
            )
            layer_metrics[str(layer)] = summarize_prompt_auc(
                layer_map,
                samples=args.bootstrap_samples,
                seed=BOOTSTRAP_SEED + checkpoint * 29 + layer,
            )

        for row_idx, output_row in enumerate(trajectory_score_rows):
            output_row["scores"][str(checkpoint)] = {
                method: (float(values[row_idx]) if np.isfinite(values[row_idx]) else None)
                for method, values in method_scores.items()
                if method
                in {
                    f"residual_layer_{layer}" for layer in layers
                }
                | {"visible_prefix_tfidf", "next_token_stats", "prefix_model_judge"}
            }

        checkpoint_results.append(
            {
                "checkpoint": checkpoint,
                "available_trajectories": int(available.sum()),
                "available_by_split": {
                    split: int((available & (splits == split)).sum())
                    for split in ("train", "validation", "test")
                },
                "evaluable_test_prompts": len(
                    prompt_mixed_ids(labels, prompts, splits, available, "test")
                ),
                "evaluable_test_risks": len(
                    {
                        cases[prompt]["risk_type"]
                        for prompt in prompt_mixed_ids(labels, prompts, splits, available, "test")
                    }
                ),
                "metrics": metrics,
                "primary_comparison": comparison,
                "secondary_layer_metrics": layer_metrics,
                "models": model_payloads,
            }
        )
        print(f"Analyzed checkpoint {checkpoint}", flush=True)

    test_yield = yield_summary["test"]
    yield_pass = test_yield["mixed_prompt_count"] >= 6 and test_yield["mixed_risk_count"] >= 2
    winning: list[int] = []
    for result in checkpoint_results:
        checkpoint = int(result["checkpoint"])
        comparison = result["primary_comparison"]
        ci = comparison["ci95"]
        if (
            checkpoint in primary_checkpoints
            and result["evaluable_test_prompts"] >= 6
            and result["evaluable_test_risks"] >= 2
            and comparison["delta"] is not None
            and comparison["delta"] >= 0.03
            and ci[0] is not None
            and ci[0] > 0
        ):
            winning.append(checkpoint)
    consecutive = [
        [left, right]
        for left, right in zip(primary_checkpoints, primary_checkpoints[1:])
        if left in winning and right in winning
    ]
    gate = {
        "status": "pass" if yield_pass and consecutive else ("inconclusive" if not yield_pass else "fail"),
        "contrast_yield_pass": yield_pass,
        "winning_checkpoints": winning,
        "consecutive_winning_pairs": consecutive,
        "residual_advantage_pass": bool(consecutive),
    }

    landing_basis = Counter(str(row.get("policy_landing_basis", "missing")) for row in trajectories)
    violating_without_landing = sum(bool(labels[idx]) and landings[idx] < 0 for idx in range(len(labels)))
    payload = {
        "protocol": {
            "experiment_label": args.experiment_label,
            "primary_checkpoints": primary_checkpoints,
            "primary_layer": args.primary_layer,
            "expected_test_prompts": args.expected_test_prompts,
            "bootstrap_samples": args.bootstrap_samples,
            "epochs": args.epochs,
            "device": str(device),
            "judge_scores": str(args.judge_scores) if args.judge_scores else None,
        },
        "trajectory_count": len(trajectories),
        "prompt_count": len(set(prompts.tolist())),
        "confirmatory_yield": yield_summary,
        "landing": {
            "basis_counts": dict(sorted(landing_basis.items())),
            "violating_without_landing": violating_without_landing,
        },
        "checkpoint_results": checkpoint_results,
        "gate": gate,
        "cost": {
            "logistic_training_seconds": total_training_seconds,
            "mean_scoring_microseconds_per_row": {
                method: float(np.mean(values)) for method, values in sorted(scoring_costs.items())
            },
            "prefix_judge": judge_summary,
            "monitoring_benchmark": monitoring_cost,
        },
    }
    (args.run_dir / args.analysis_name).write_text(
        json.dumps(payload, indent=2, default=json_default), encoding="utf-8"
    )
    with (args.run_dir / args.scores_name).open("w", encoding="utf-8") as handle:
        for row in trajectory_score_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        f"# {args.experiment_label} Confirmatory Trajectory Results",
        "",
        f"- gate: **{gate['status'].upper()}**",
        f"- trajectories / prompts: `{len(trajectories)}` / `{len(set(prompts.tolist()))}`",
        f"- fresh-seed mixed test prompts: `{test_yield['mixed_prompt_count']}/{args.expected_test_prompts}` across "
        f"`{test_yield['mixed_risk_count']}` risks",
        f"- winning checkpoints: `{winning}`",
        f"- consecutive winning pairs: `{consecutive}`",
        f"- violating trajectories without a semantic landing: `{violating_without_landing}`",
        "",
        "## Primary Checkpoint Curve",
        "",
        f"| checkpoint | prompts | risks | residual L{args.primary_layer} | TF-IDF | next-token | envelope | delta [95% CI] | judge |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in checkpoint_results:
        metrics = result["metrics"]
        residual = metrics[f"residual_layer_{args.primary_layer}"]["auc"]
        text_auc = metrics["visible_prefix_tfidf"]["auc"]
        stat_auc = metrics["next_token_stats"]["auc"]
        judge_auc = metrics.get("prefix_model_judge", {}).get("auc")
        comparison = result["primary_comparison"]
        ci = comparison["ci95"]
        lines.append(
            f"| {result['checkpoint']} | {result['evaluable_test_prompts']} | "
            f"{result['evaluable_test_risks']} | {fmt(residual)} | {fmt(text_auc)} | "
            f"{fmt(stat_auc)} | {fmt(comparison.get('baseline_auc'))} | "
            f"{fmt(comparison.get('delta'))} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(judge_auc)} |"
        )
    lines.extend(
        [
            "",
            "## Secondary Layer Diagnostic",
            "",
            f"| checkpoint | primary L{args.primary_layer} | best captured layer | best residual AUC |",
            "|---:|---:|---:|---:|",
        ]
    )
    for result in checkpoint_results:
        if result["checkpoint"] not in primary_checkpoints:
            continue
        candidates = [
            (int(layer), value.get("auc"))
            for layer, value in result["secondary_layer_metrics"].items()
            if value.get("auc") is not None
        ]
        best_layer, best_auc = max(candidates, key=lambda item: item[1])
        lines.append(
            f"| {result['checkpoint']} | "
            f"{fmt(result['metrics'][f'residual_layer_{args.primary_layer}']['auc'])} | "
            f"{best_layer} | {fmt(best_auc)} |"
        )
    lines.extend(
        [
            "",
            "## Fresh-Seed Contrast Yield",
            "",
            "| split | mixed / total | risk coverage |",
            "|---|---:|---:|",
        ]
    )
    for split in ("train", "validation", "test"):
        value = yield_summary[split]
        lines.append(
            f"| {split} | {value['mixed_prompt_count']} / {value['prompt_count']} | "
            f"{value['mixed_risk_count']} |"
        )
    lines.extend(["", "## Measured Cost", ""])
    if monitoring_cost:
        ratio_ci = monitoring_cost["capture_to_plain_ratio_ci95"]
        lines.extend(
            [
                f"- {len(layers)}-layer, {len(checkpoints)}-checkpoint capture/plain generation ratio: "
                f"`{monitoring_cost['capture_to_plain_ratio']:.3f}` "
                f"(95% paired CI `{ratio_ci[0]:.3f}`-`{ratio_ci[1]:.3f}`; "
                f"`{monitoring_cost['paired_runs']}` paired runs).",
                f"- Mean capture overhead: "
                f"`{monitoring_cost['mean_paired_overhead_seconds'] * 1000:.1f}` ms/trajectory.",
            ]
        )
    if judge_summary:
        lines.append(
            f"- Prefix judge: `{judge_summary['forward_seconds']:.3f}` seconds for "
            f"`{judge_summary['unique_rendered_prefixes']}` unique prefixes "
            f"(`{judge_summary['milliseconds_per_unique_prefix']:.1f}` ms/prefix)."
        )
    residual_cost = payload["cost"]["mean_scoring_microseconds_per_row"].get(
        f"residual_layer_{args.primary_layer}"
    )
    if residual_cost is not None:
        lines.append(
            f"- Layer-{args.primary_layer} logistic scoring: `{residual_cost:.3f}` microseconds/row."
        )
    lines.extend(
        [
            "",
            "Secondary layers are post-hoc diagnostics. They cannot override the frozen "
            f"layer-{args.primary_layer} gate.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Discipline",
            "",
            "This experiment tests extraction latency and cost, not information exclusivity. "
            "The selected prompts were adaptively screened for stochastic outcome contrast, so "
            "the results do not estimate violation prevalence. Per-risk and secondary-layer "
            "results are diagnostics and cannot override the frozen primary gate.",
        ]
    )
    (args.run_dir / args.report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.experiment_label} gate: {gate['status']}")
    print(f"Saved analysis to {args.run_dir}")


if __name__ == "__main__":
    main()
