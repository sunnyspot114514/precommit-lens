# Pre-Registered v4 Trajectory Discovery Protocol

This document freezes the v4 claim boundary and prompt-screening procedure
before trajectory screening. Discovery samples are not confirmatory data.

## Claim Boundary

The confirmatory v4 experiment will test whether residual-state probes provide
a better latency-compute-accuracy tradeoff for early policy-violation warning
than practical visible-prefix baselines.

It will **not** test whether residual states contain information unavailable
from the visible prefix and the model computation. For fixed model parameters,
the residual state at generation step `t` is a deterministic function of the
prompt and sampled prefix. Any positive result is therefore an accessibility,
latency, or cost result, not an information-exclusivity result.

## Discovery Purpose

Within-prompt evaluation requires prompts that generate both policy-compliant
and policy-violating trajectories. Discovery estimates this contrast yield.
It does not estimate probe performance.

## Candidate Pool

- Model: `Qwen/Qwen3-0.6B`.
- Source: full-chat-audited v3 prompt corpus.
- Initial conditions:
  - `concept_present_target_absent`;
  - `target_present_concept_absent`.
- All four risk families remain eligible.
- Candidate sampling is balanced by risk and condition and deterministic by
  case-id hash.
- If this pool yields too few mixed prompts, new ambiguity-calibrated prompts
  may be constructed in a second discovery round. That round must pass the same
  full-chat token audit and remains separate from confirmation.

The v2 `generated_rollback` pool is not used because its original system
message leaked `validator` and its observed label was a lexical substring gate.

## Sampling

- `do_sample=True`
- temperature: `0.8`
- top-p: `0.95`
- maximum generated tokens: `48`
- discovery trajectories per prompt: `16`
- discovery seed range begins at `100000`
- thinking mode disabled through the model chat template
- one trajectory per generation call so each seed is reproducible

## Eligibility

A prompt is eligible when its discovery policy-violation rate lies in
`[0.20, 0.80]`. Screening uses only the risk-specific v3 policy validator, not
the lexical or structural rollback endpoints.

The confirmatory phase requires:

- at least `24` eligible prompts overall;
- at least `6` eligible prompts in at least three risk families;
- enough prompt/template families to form disjoint train, validation, and test
  partitions.

If these requirements fail, no confirmatory probe result will be reported from
this pool. Prompt calibration may continue only as another discovery round.

## Separation From Confirmation

- Discovery seeds and outputs are permanently excluded from probe training,
  layer selection, threshold selection, and test metrics.
- Confirmation uses a new seed range and at least `32` trajectories per prompt.
- Prompt and template-family splits will be frozen after discovery but before
  any confirmatory trajectory is generated.
- Prompts that collapse to one outcome under fresh confirmatory seeds remain in
  the contrast-yield denominator; they are not silently replaced.

## Planned Confirmatory Statistics

- Prompt-macro within-prompt pairwise AUC.
- Prompt-cluster bootstrap confidence intervals.
- Fixed absolute generation checkpoints as the primary curve.
- Only trajectories that have not yet crossed the deterministic policy landing
  point contribute at a checkpoint.
- Event-relative lead time is secondary because compliant trajectories have no
  natural violation landing point.
- Residual probes are compared with prompt-only, visible-prefix TF-IDF,
  next-token distribution, and prefix-judge baselines.
- Monitoring overhead is measured rather than assumed to be zero.

The exact checkpoints, layers, primary scalar summary, baseline ladder, and
success gate will be frozen in a separate confirmatory protocol after discovery
establishes the available prompt count.
