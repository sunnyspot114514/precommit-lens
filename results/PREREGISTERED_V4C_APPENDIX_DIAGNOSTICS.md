# Frozen v4c Post-Hoc Appendix Diagnostics

This document freezes two bounded appendix diagnostics after the v4c
`DISCOVERY YIELD FAIL` was observed and before any trajectory in these
diagnostics is sampled. These analyses cannot reopen, rescue, or replace the
v4c discovery gate. They introduce no new success gate and trigger no
confirmatory residual experiment.

## Shared Frozen Population and Outcomes

- Reuse the exact 64 round-one `equal_authority_conflict` prompts from
  `data/prompt_sets/trajectory_candidates_v4c_round1.jsonl`.
- Dataset SHA-256:
  `74f8d38d023c41fbeb3b790f39bbbf46f2e99d6814431bd576d7c13bbfcc114e`.
- Sample 16 trajectories per prompt with at most 48 new tokens.
- Preserve the risk-specific policy validators and the prompt-level eligible
  interval `[0.20, 0.80]`, inclusive.
- Report eligible prompts; always-commit, always-rollback, and mixed prompt
  counts; exact candidate-A/B output rate; and the same quantities by risk.
- `mixed` means any prompt with both outcomes (`0 < rate < 1`); it remains
  distinct from the stricter eligible interval.
- Do not select prompts, fit probes, capture residuals, or construct a new
  confirmatory set from these outputs.

## A. Qwen3-4B Temperature Sensitivity

This diagnostic answers whether the observed v4c round-one yield is specific
to the original `temperature=0.8` setting.

- Model: `Qwen/Qwen3-4B`.
- Revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Backend and precision: the existing Transformers runner, unquantized FP16.
- Reuse the existing `temperature=0.8`, seed-start `11,000,000` round-one
  result without resampling.
- Add `temperature=1.2`, seed-start `21,000,000`.
- Add `temperature=1.5`, seed-start `22,000,000`.
- Keep `top_p=0.95`, the existing model generation configuration, chat
  rendering, validators, prompt order, and all other runner behavior fixed.
- Both new temperatures are run regardless of the first result. No additional
  temperature may be introduced.

If higher temperature increases yield, the conclusion narrows to scarcity at
the original standard sampling configuration. If it does not, that is stronger
evidence of robustness over the two disclosed higher-temperature settings.
Neither outcome supports a universal claim that Qwen3-4B is deterministic.

## B. Deployment-State Model Controls

This diagnostic asks whether round-one yield scarcity appears in two already
specified local model families. It is not an FP16 scale curve: the models use
Ollama GGUF deployment artifacts and native chat renderers.

### Frozen Models

| model tag | role | required manifest digest | required quantization | seed start |
|---|---|---|---|---:|
| `gemma4:e2b` | non-Qwen cross-family control | `7fbdbf8f5e45a75bb122155ed546e765b4d9c53a1285f62fd9f506baa1c5a47e` | `Q4_K_M` | 31,000,000 |
| `qwen3.5:4b` | same-family, newer-generation control | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` | `Q4_K_M` | 32,000,000 |

The runner must refuse a mutable tag whose local digest or quantization differs
from this table. Ollama version is frozen to `0.31.2`. The Gemma artifact was
identified in the registry but had not been pulled or sampled when this
protocol was written.

### Frozen Sampling

- Ollama `/api/chat`, one request per trajectory, native renderer.
- `temperature=0.8`, `top_p=0.95`, `top_k=50`.
- `repeat_penalty=1.0`, `presence_penalty=0.0`,
  `frequency_penalty=0.0`.
- `num_ctx=512`, `num_predict=48`, thinking disabled.
- The system and user strings are unchanged; model-native chat wrappers may
  differ and are part of the disclosed deployment-state estimand.
- Both models are run exactly once over round one. No cross-model round two,
  prompt replacement, extra model, or residual analysis is allowed.

Gemma is the cross-family diagnostic. Qwen3.5 separates a Qwen3-specific result
from a newer Qwen-family deployment. Because backend, quantization, model
generation, and chat rendering all differ from the FP16 Qwen3-4B run, these
results are descriptive controls rather than causal model-family estimates.

## Claim and Stopping Boundary

The only admissible claims concern sensitivity of round-one prompt-level yield
to the two frozen temperature changes and the two frozen deployment artifacts.
No result changes the v4c `DISCOVERY YIELD FAIL`, creates a new primary gate, or
authorizes further experiment rounds. After these four new 1,024-trajectory
conditions and their report are complete, experimental development stops.
