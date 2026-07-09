# Leakage-Controlled v2 Report

## Bottom Line

The local Qwen3-0.6B falsification stage is complete, and it does **not** support scaling dense J-lens to larger models yet.

On the leakage-controlled v2 corpus, raw dense J-lens and selected-layer JVP do not beat keyword baselines on the primary `expected_rollback` label. A cheap linear probe is much stronger, which means the practical governance story currently points toward low-cost learned pre-commit monitors, not dense Jacobian readouts.

The current prompt set also passes a tokenizer-level leakage audit for `Qwen/Qwen3-0.6B`: 400 target-absent prompts checked against 30,000 watched-token sequences, with 0 violations.

## What Ran

- Model: `Qwen/Qwen3-0.6B`
- Device: local RTX 3060 12GB
- Dataset: `800` cases
- Risks: `early_spoiler`, `fake_commit`, `hidden_fields`, `schema_bypass`
- Design: `50` matched groups per risk, four conditions per group
- Dense lens: existing full-layer `1024 x 1024` matrices for all 28 layers
- JVP: selected layers `[0, 7, 14, 21, 27]`
- Linear probe: group-held-out split, layer selected by validation AUC
- Stability ablation: 4-prompt lens vs 32-prompt lens on 200 core paraphrase cases

Core files:

- Protocol: `results/PREREGISTERED_EVALUATION_PROTOCOL.md`
- Dataset: `data/prompt_sets/leakage_controlled_v2.jsonl`
- Category AUC table: `results/auc_by_category.md`
- Token leakage audit: `results/leakage_controlled_v2_main/Qwen__Qwen3-0.6B/TOKEN_LEAKAGE_AUDIT.md`
- Main metrics: `results/leakage_controlled_v2_main/Qwen__Qwen3-0.6B/EVALUATION_SUMMARY.md`
- Falsification summary: `results/leakage_controlled_v2_main/Qwen__Qwen3-0.6B/FALSIFICATION_SUMMARY.md`
- Intervention summary: `results/leakage_controlled_v2_intervention/Qwen__Qwen3-0.6B/INTERVENTION_SUMMARY.md`
- Lens stability summary: `results/dense_lens_stability_4fit_vs_32fit/Qwen__Qwen3-0.6B/LENS_STABILITY.md`

## Main AUC Results

Primary label: `expected_rollback`.

| method | AUC | interpretation |
|---|---:|---|
| `keyword_target_present` | 0.500 | target-word presence alone is balanced by design |
| `keyword_hit_count` | 0.625 | surface count still has nontrivial signal |
| `logit_lens` | 0.486 | near random on the primary label |
| `dense_jlens` | 0.416 | fails the keyword-baseline check |
| `jvp_lens` | 0.317 | fails the keyword-baseline check |
| `linear_probe` | 1.000 | cheap learned monitor dominates on this corpus |

Observed label: `generated_rollback`.

| method | AUC | interpretation |
|---|---:|---|
| `keyword_hit_count` | 0.647 | still stronger than dense/JVP overall |
| `logit_lens` | 0.451 | weak |
| `dense_jlens` | 0.501 | near chance overall |
| `jvp_lens` | 0.411 | weak |
| `linear_probe` | 0.858 overall / 0.886 test | strong but may exploit template regularities |

## Per-Risk Notes

- `early_spoiler`: dense J-lens is strongly negative on expected rollback (`0.231`) and generated rollback (`0.246`).
- `fake_commit`: logit lens is useful on generated rollback (`0.742`), but dense J-lens is weak (`0.522`) and below keyword-hit baseline (`0.641`).
- `hidden_fields`: dense/JVP/logit are all weak after the `motive_cache` token leak was removed from the paraphrase condition.
- `schema_bypass`: dense J-lens has high generated-rollback AUC (`0.827`), but keyword-hit baseline is higher (`0.841`) and expected-rollback AUC is weak (`0.450`).

This means the current dense signal is not robust enough to justify a cloud scale curve.

## Intervention Result

The dense-direction suppression sweep ran on `50` paraphrase attacks and `50` hard-neutral controls per risk. It compared suppression against a random same-norm sham direction.

| risk | attack baseline | suppress | sham | result |
|---|---:|---:|---:|---|
| `early_spoiler` | 1.000 | 1.000 | 1.000 | no effect |
| `fake_commit` | 0.800 | 1.000 | 0.760 | suppression worsened rollback |
| `hidden_fields` | 0.900 | 0.900 | 0.760 | no suppression benefit |
| `schema_bypass` | 0.500 | 0.600 | 0.440 | suppression worsened rollback |

Hard-neutral rollback stayed at `0.000` for baseline, suppress, and sham across all four risks. So the intervention did not broadly break benign cases, but it also did not reduce risky outputs.

## Lens Stability Ablation

A 32-prompt full-layer dense lens was fitted on the same local RTX 3060. The fit used all 28 layers and took `3341.1s` (`55.7` minutes). The resulting `.npz` artifact is intentionally ignored by Git.

The 4-fit and 32-fit lenses were compared on 200 `concept_present_target_absent` cases:

| metric | value |
|---|---:|
| mean Spearman rank correlation | 0.895 |
| median Spearman rank correlation | 0.899 |
| mean Pearson correlation | 0.937 |
| top-1 agreement | 0.700 |

By risk:

| risk | mean Spearman | top-1 agreement |
|---|---:|---:|
| `early_spoiler` | 0.920 | 0.400 |
| `fake_commit` | 0.876 | 0.400 |
| `hidden_fields` | 0.907 | 1.000 |
| `schema_bypass` | 0.876 | 1.000 |

Interpretation: broad watched-token ranking is fairly stable to fitting-prompt count, but headline top-1 tokens are not stable for early-spoiler and fake-commit. This reinforces the decision to avoid single-case, single-token evidence.

## Interpretation

This is a useful negative result.

The v2 experiment supports three conclusions:

- Target-word leakage was reduced enough to expose that raw dense/JVP watched-token ranks are not reliable on this corpus.
- Linear probes are much cheaper and much stronger, so the governance monitor path should prioritize learned residual probes before dense Jacobian scaling.
- Dense-direction intervention does not currently provide a causal control handle.
- The dense lens itself is moderately stable in aggregate, but individual top tokens can be unstable.

The project should not rent RTX Pro 6000 for 8B/14B dense J-lens yet. The next scale-up should wait until a stronger local signal survives keyword, template, and sham controls.

## Next Experimental Fix

The main remaining confound is template regularity. The linear probe can likely detect repeated condition phrasing, not only abstract runtime risk.

Recommended next step:

- Add multiple paraphrase templates per condition.
- Add a held-out-template split.
- Add cross-risk transfer: train on three risk families, test on the fourth.
- Keep `target_present_concept_absent` as the token-copy control.
- Re-run 0.6B before any cloud scale curve.

## 中文摘要

本地 0.6B 阶段已经完成，但结果不支持现在就租 RTX Pro 6000 做大模型 dense J-lens 规模曲线。

主要结论：

- 去泄漏 v2 数据集共有 800 个 case，每类风险 50 组，每组 4 个条件。
- dense J-lens 和 JVP 在主指标 `expected_rollback` 上没有超过 keyword baseline。
- linear probe 很强，但它可能利用了模板规律，所以不能直接解读成抽象概念已经被完美读出。
- 系统干预没有证明因果效果：suppress 没有比 sham 更稳定地降低攻击 rollback，有些风险还变差。

因此当前最稳妥的论文方向不是“dense Jacobian 是必要治理机制”，而是：

> 在 pre-commit runtime governance 中，廉价 residual probe 可能比重型 dense Jacobian 更实用；dense J-lens 目前更适合作为诊断工具，而不是部署态监控器。
