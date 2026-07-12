# v4d Pre-Landing Identity Diagnostic

**Status: post-hoc mechanistic diagnostic; the frozen v4d gate is unchanged.**

| checkpoint | evaluable prompts | risks | identical visible prefix | identical L21 residual | identical all captured layers | max L21 span |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 3 | 8 | 8 | 8 | 0.000000 |
| 2 | 8 | 3 | 8 | 8 | 8 | 0.000000 |
| 4 | 8 | 3 | 8 | 8 | 8 | 0.000000 |
| 6 | 5 | 2 | 5 | 5 | 5 | 0.000000 |
| 8 | 3 | 1 | 3 | 3 | 3 | 0.000000 |
| 10 | 3 | 1 | 3 | 3 | 3 | 0.000000 |
| 12 | 3 | 1 | 3 | 3 | 3 | 0.000000 |
| 16 | 3 | 1 | 3 | 3 | 3 | 0.000000 |
| 24 | 2 | 1 | 0 | 0 | 0 | 8.660156 |

## First Divergence Relative to Policy Landing

| risk | before landing | at landing | after landing |
|---|---:|---:|---:|
| `early_spoiler` | 0 | 412 | 0 |
| `fake_commit` | 0 | 602 | 0 |
| `hidden_fields` | 714 | 0 | 0 |

Across test prompts, `1014` cross-label pairs first diverge exactly at policy landing and `714` diverge before landing. All early-spoiler and fake-commit pairs diverge at landing. Every pre-landing divergence is a two-token lead in hidden-fields (`714` pairs).

At frozen checkpoints 0-16, every evaluable prompt has one visible prefix and zero within-prompt span at every captured layer. Checkpoint 24 contains state variation, but only two test prompts from one risk remain pre-landing, below the frozen support requirement. No intermediate checkpoint was sampled post hoc.
