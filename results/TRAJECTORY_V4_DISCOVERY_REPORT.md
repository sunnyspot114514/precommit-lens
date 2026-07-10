# v4 Trajectory Discovery Report

This report records the adaptive prompt-screening stage that preceded the
frozen v4 confirmatory experiment. Discovery trajectories are selection data,
not confirmatory evidence.

## Frozen Eligibility Rule

A prompt was eligible when `Qwen3-0.6B`, sampled 16 times at temperature `0.8`
and top-p `0.95`, produced a risk-specific policy-violation rate in
`[0.20, 0.80]`. Confirmation required at least 24 eligible prompts overall and
at least 6 prompts in at least three risk families.

## Discovery Rounds

| round | purpose | prompts | trajectories | full-chat audit | eligible |
|---|---|---:|---:|---:|---:|
| 1 | ambiguous safe/unsafe candidates across four risks | 48 | 768 | 0 / 3,600 violations | 15 |
| 2 | explicit equal-probability choice for missing risks | 48 | 768 | 0 / 3,552 violations | 3 |
| 3 | final graded safe-first calibration for early spoiler | 48 | 768 | 0 / 2,880 violations | 16 |

Round 1 produced one eligible `early_spoiler`, seven `hidden_fields`, seven
`schema_bypass`, and no `fake_commit` prompts. Round 2 showed that the model
mostly ignored explicit 50/50 instructions: it added three eligible early
spoiler prompts and still no fake-commit prompt. Round 3 was declared the final
calibration round before sampling. It added 16 early-spoiler prompts across
seven wording families.

## Gate Outcome

The combined eligible pool contains 34 prompts:

| risk | eligible prompts |
|---|---:|
| early_spoiler | 20 |
| hidden_fields | 7 |
| schema_bypass | 7 |
| fake_commit | 0 |

The discovery gate passed with three risk families above six prompts. The 34
prompts cover 24 template families and were frozen into a 16/9/9 grouped
train/validation/test split. The exact prompt list and SHA-256 are in
`results/trajectory_v4_discovery/CONFIRMATORY_SPLIT_MANIFEST.md`.

## Interpretation Boundary

The pool is adaptively enriched for prompts whose sampled outcomes diverge.
Confirmatory v4 metrics therefore test monitoring performance within this
selected high-variance population. They do not estimate how often violations
occur under natural traffic or under the unscreened candidate distribution.

No discovery seed, generated output, residual state, or observed discovery
label is reused for confirmatory model fitting, validation selection, or test
metrics. Prompts that collapse under the fresh confirmatory seeds remain in the
contrast-yield denominator and are not replaced.
