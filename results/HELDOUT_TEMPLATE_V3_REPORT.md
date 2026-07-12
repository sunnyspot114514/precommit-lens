# Held-Out-Template v3 Report

## Bottom Line

The pre-registered Qwen3-0.6B v3 stage is complete. A residual linear probe
does generalize to held-out prompt-template families, but it adds no predictive
value over a prompt-text-only classifier. The local multimodel and cloud scale
gates therefore remain closed.

This is a sharper result than v2:

- the residual signal is real on this corpus, not merely a repeated-template
  train/test leak;
- the same semantic and generated-policy labels are predicted even better by
  TF-IDF features computed from the model input;
- dense J-lens and JVP remain unreliable overall;
- current dense-direction suppression is causally counterproductive relative
  to a paired sham control.

## Protocol

- Model: `Qwen/Qwen3-0.6B`
- Hardware: local RTX 3060 12GB
- Cases: `960`
- Risks: four
- Complete template families per risk: `12`
- Matched content groups per template family: `5`
- Split: `480` train / `240` validation / `240` test cases
- Test template families are absent from both probe training and layer
  selection.
- Target-absent full-chat inputs audited: `480`
- Watched-token sequences checked: `36,000`
- Full-chat audit violations: `0`
- Dense lens: saved full-layer `1024 x 1024` Qwen3-0.6B lens
- JVP layers: `[0, 7, 14, 21, 27]`

The thresholds and scale gate were frozen in
`results/PREREGISTERED_V3_PROTOCOL.md` before the v3 model run.

## Main Test Results

### Constructed semantic-risk label

| method | AUC | pair-cluster 95% CI | template-cluster 95% CI |
|---|---:|---:|---:|
| target-word presence | 0.500 | [0.500, 0.500] | [0.500, 0.500] |
| keyword hit count | 0.438 | [0.412, 0.463] | [0.396, 0.500] |
| logit lens | 0.405 | [0.346, 0.456] | [0.264, 0.507] |
| dense J-lens | 0.498 | [0.432, 0.563] | [0.358, 0.638] |
| selected-layer JVP | 0.481 | [0.436, 0.521] | [0.385, 0.569] |
| residual linear probe | 0.897 | [0.861, 0.937] | [0.826, 0.982] |
| leave-one-risk-out residual probe | 0.681 | [0.628, 0.732] | [0.564, 0.815] |
| word TF-IDF | 1.000 | [1.000, 1.000] | [1.000, 1.000] |
| character TF-IDF | 1.000 | [1.000, 1.000] | [1.000, 1.000] |

The residual probe minus the validation-selected text baseline is `-0.103`,
with paired 95% CI `[-0.138, -0.064]`. The interval excludes zero in the wrong
direction for an internal-state added-value claim.

### Generated policy-violation label

| method | AUC | pair-cluster 95% CI |
|---|---:|---:|
| dense J-lens | 0.500 | [0.430, 0.568] |
| selected-layer JVP | 0.469 | [0.417, 0.516] |
| residual linear probe | 0.918 | [0.889, 0.948] |
| leave-one-risk-out residual probe | 0.707 | [0.656, 0.753] |
| word TF-IDF | 0.994 | [0.988, 0.999] |
| character TF-IDF | 0.990 | [0.977, 0.998] |

The residual signal predicts generated behavior, but the prompt text predicts
it better because Qwen3-0.6B usually follows the explicit risk instruction.

## Dense Readout Heterogeneity

Dense J-lens is not uniformly null. On the semantic test label its per-risk
AUCs are:

| risk | dense AUC | concept main effect | paired 95% CI |
|---|---:|---:|---:|
| early_spoiler | 0.736 | 0.145 | [0.103, 0.190] |
| fake_commit | 0.472 | -0.060 | [-0.107, -0.014] |
| hidden_fields | 0.702 | 0.145 | [0.038, 0.271] |
| schema_bypass | 0.283 | -0.119 | [-0.210, -0.033] |

This supports a category-specific diagnostic interpretation, not a general
pre-commit monitor. The overall dense AUC is `0.498`, and two risks have
effects in the wrong direction.

## Validator Separation

v3 reports three different observed endpoints:

1. the legacy lexical-substring gate;
2. a risk-specific policy-violation check;
3. JSON parse/contract validation plus the policy check.

`897/960` generations contained a parseable JSON object. Structural rollback
is intentionally not the primary scale gate because many safe generations
omit required keys even when they contain no risky behavior. Policy violation
is the cleaner behavioral endpoint for this experiment.

## v2 Audit Correction

The original v2 audit checked only the user prompt. Re-auditing the complete
rendered system+user input found `100` target-absent violations, all in
`fake_commit`: the shared system message contained the watched target term
`validator`.

The original `0/30,000` result remains a correct user-prompt-only audit, but it
must not be described as a complete model-input audit. v3 replaces it with a
full-chat `0/36,000` result.

## Paired Intervention Result

Adding paired uncertainty changes the intervention result from an anecdotal
point estimate to clear negative causal evidence:

| risk | suppress minus sham rollback | paired 95% CI | exact McNemar p |
|---|---:|---:|---:|
| fake_commit | +0.240 | [0.120, 0.360] | <0.001 |
| hidden_fields | +0.140 | [0.060, 0.240] | 0.016 |
| schema_bypass | +0.160 | [0.040, 0.280] | 0.039 |

The current dense direction increases risky rollback relative to sham; it is
not a useful suppression handle.

## Decision

The pre-registered semantic-signal and generated-policy gates pass, but the
internal-state added-value gate fails. Therefore:

- do not run Qwen3.5/Gemma v3 replication yet;
- do not rent an RTX Pro 6000;
- do not claim that residual probes reveal information unavailable from the
  prompt surface;
- retain dense/JVP as diagnostic baselines rather than governance monitors.

The next scientifically distinct experiment would hold the prompt fixed and
label divergent sampled trajectories. It should compare early-generation
residual state against the exact visible token prefix, so prompt wording can no
longer determine the label by construction. That is a new experiment, not a
reason to expand the current v3 corpus after its gate failed.

## Artifacts

- `results/PREREGISTERED_V3_PROTOCOL.md`
- `data/prompt_sets/heldout_templates_v3.jsonl`
- `results/heldout_templates_v3_main/Qwen__Qwen3-0.6B/TOKEN_LEAKAGE_AUDIT.md`
- `results/heldout_templates_v3_main/Qwen__Qwen3-0.6B/EVALUATION_SUMMARY.md`
- `results/heldout_templates_v3_analysis/Qwen__Qwen3-0.6B/V3_FALSIFICATION_REPORT.md`
- `results/leakage_controlled_v2_main/Qwen__Qwen3-0.6B/FULL_INPUT_TOKEN_LEAKAGE_REAUDIT.md`
- `results/leakage_controlled_v2_intervention/Qwen__Qwen3-0.6B/PAIRED_INTERVENTION_STATS.md`
