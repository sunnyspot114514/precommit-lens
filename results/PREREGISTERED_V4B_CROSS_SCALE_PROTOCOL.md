# Pre-Registered v4b Cross-Scale Confirmatory Protocol

This protocol was frozen before any `Qwen/Qwen3-4B` trajectory was generated
or inspected. v4b is a single planned cross-scale replication of v4. It changes
the monitored model and depth-normalized residual layers while retaining the
frozen prompt population, split, sampling parameters, estimands, baselines, and
success gate.

## Claim Under Test

The practical claim is unchanged: at fixed pre-landing generation checkpoints,
a residual-state probe may predict the eventual policy-validator outcome more
accurately than the cheap visible-prefix baseline envelope.

v4b tests whether the v4 result is specific to a `0.6B` model. It does not test
information exclusivity, natural violation prevalence, or arbitrary model-judge
prompts.

## Frozen Model and Data

- Model: `Qwen/Qwen3-4B`.
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Precision: unquantized `float16` weights.
- Prompt file: `data/prompt_sets/trajectory_confirmatory_v4.jsonl`.
- Prompt-file SHA-256:
  `29763735accfb0e03d99fb81fff7da5cf5a303d626dc7803154185104d136907`.
- Frozen prompts / template families: `34 / 24`.
- Frozen train / validation / test prompts: `16 / 9 / 9`.
- The v4 split and all complete template-family assignments are reused.

The same prompt population is the primary cross-scale estimand. A prompt that
loses within-prompt outcome contrast under the 4B model remains in every yield
denominator and is not replaced. No 4B-specific discovery cohort may rescue the
primary result.

## Sampling and Landing

- `32` trajectories per prompt (`1,088` planned trajectories).
- Seed start: `5,000,000`, with the same deterministic seed mapping as v4.
- `temperature=0.8`, `top_p=0.95`, `max_new_tokens=48`.
- Maximum rendered input length: `256` tokens.
- Thinking mode disabled through the chat template.
- One trajectory per generation call.
- The risk-specific semantic landing and incremental-validator fallback are
  unchanged from v4.

## Checkpoints and Depth Mapping

- Primary checkpoints: `2, 4, 6, 8, 10, 12`.
- Sanity checkpoint: `0`.
- Secondary checkpoints: `16, 24`.
- Qwen3-0.6B reference layers: `0, 6, 12, 18, 24, 27` of `28` layers.
- Qwen3-4B captured layers: `0, 8, 16, 23, 31, 35` of `36` layers.
- Frozen mapping rule: `round(reference_layer / 27 * 35)`.
- Primary layer: `23`, the depth-normalized counterpart of v4 layer `18`.

Secondary layers remain diagnostics and cannot override the layer-23 gate.

## Models, Metrics, and Gate

The residual probe, validation-selected word/character TF-IDF baseline,
next-token-statistics baseline, same-model forced-choice prefix judge, balanced
training weights, prompt-macro within-prompt pairwise AUC, and `2,000`
prompt-cluster bootstrap replicates are unchanged from v4.

The cheap-baseline envelope remains the checkpoint-wise larger AUC of the
validation-selected TF-IDF model and next-token-statistics model, recomputed in
each paired bootstrap replicate.

The result is positive only if all frozen v4 conditions hold:

1. At least `6/9` test prompts across at least two risks retain fresh-seed
   within-prompt contrast.
2. At a winning checkpoint, at least six test prompts across at least two risks
   contain both pre-landing outcomes.
3. Layer-23 residual AUC exceeds the cheap-baseline envelope by at least `0.03`,
   with the paired 95% interval entirely above zero.
4. Condition 3 holds at two consecutive primary checkpoints.

Failure of condition 1 makes the result inconclusive for accessibility; failure
of conditions 2-4 after condition 1 passes makes it negative. Per-risk,
secondary-layer, and judge results cannot rescue the gate.

## Feasibility and Cost Discipline

An unquantized FP16 one-prompt, two-seed smoke test may be run before the full
experiment. It checks only loading, deterministic capture, and peak memory; its
behavioral outputs are not used for prompt selection, model selection, or
threshold changes. If FP16 cannot run locally, execution moves unchanged to a
larger GPU rather than silently changing quantization.

The paired plain/capture benchmark uses the same six-layer, nine-checkpoint
research path and reports output identity, latency ratio, and paired bootstrap
interval. The 4B prefix judge uses the same frozen forced-choice prompt as v4;
its failure or success is interpreted only for that baseline specification.
