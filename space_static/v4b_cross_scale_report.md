# v4/v4b Cross-Scale Confirmatory Report

v4b reused the exact frozen 34-prompt v4 population, split, sampling parameters, validator, semantic landing, baselines, and success gate. Only the model and depth-normalized captured layers changed.

## Frozen-Prompt Contrast Transfer

| model | gate | train mixed | validation mixed | test mixed | prompt outcome states |
|---|---|---:|---:|---:|---|
| `Qwen/Qwen3-0.6B` | **FAIL** | 16/16 (3 risks) | 9/9 (3 risks) | 9/9 (3 risks) | 34 mixed |
| `Qwen/Qwen3-4B` | **INCONCLUSIVE** | 0/16 (0 risks) | 1/9 (1 risks) | 1/9 (1 risks) | 19 always commit; 13 always rollback; 2 mixed |

The 4B run fails the frozen contrast-transfer requirement (`1/9` mixed test prompts from one risk, versus the required `6/9` from at least two risks). It also has zero mixed training prompts, so residual, TF-IDF, and next-token classifiers cannot be fit under the preregistered within-prompt weighting rule.

## Interpretation

The v4b accessibility result is **inconclusive**, not negative. The experiment does not establish that residual probes lack added value at 4B. It establishes that a contrast-selected benchmark discovered on Qwen3-0.6B does not remain contrastive on Qwen3-4B. Replacing prompts after observing this collapse would change the estimand and is disallowed by the frozen protocol.

This scale dependence is behaviorally directional: of 34 prompts, 19 become always commit, 13 become always rollback, and only 2 remain mixed. The result therefore rules out a naive same-prompt scale curve while leaving a separately preregistered, 4B-specific discovery cohort as a distinct future experiment.

## Local Reproducibility and Cost

- Qwen3-4B FP16 six-layer capture peak: `7.644` GiB allocated / `7.693` GiB reserved.
- Full sampling throughput: `21.231` generated tokens/s.
- Capture/plain ratio: `1.026` (95% CI `1.016`-`1.037`; 18 paired runs).
- Plain and capture outputs were identical in every paired cost run.

See `results/PREREGISTERED_V4B_CROSS_SCALE_PROTOCOL.md` and `results/trajectory_v4b_confirmatory/Qwen__Qwen3-4B/V4B_CONFIRMATORY_RESULTS.md`.
