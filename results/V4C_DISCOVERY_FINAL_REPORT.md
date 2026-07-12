# v4c Qwen3-4B-Native Discovery Final Report

## Frozen Outcome

The pre-registered v4c discovery gate is **DISCOVERY YIELD FAIL**. Across three frozen Qwen3-4B-native candidate mechanisms, only `4/192` prompts were eligible under the `[0.20, 0.80]` within-prompt violation-rate rule. The capped pool retained all four. It is below the required 30 prompts, 24 template families, three risks with at least eight prompts, and 12 non-lottery prompts.

Per protocol, v4c stops here. No residual probe is fit and no confirmatory accessibility claim is tested.

## Discovery Yield

`Always commit / always rollback / mixed` counts all non-extreme rates as mixed; eligibility is stricter and requires a rate inside `[0.20, 0.80]`.

| round | mechanism | prompts | trajectories | eligible | always C / R / mixed | exact candidate | A among exact | tok/s | peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `equal_authority_conflict` | 64 | 1024 | 3 | 25 / 24 / 15 | 1017/1024 (99.3%) | 72.6% | 21.285 | 7.589 |
| 2 | `boundary_tradeoff` | 64 | 1024 | 1 | 29 / 28 / 7 | 961/1024 (93.8%) | 77.9% | 21.597 | 7.587 |
| 3 | `latent_lottery` | 64 | 1024 | 0 | 29 / 33 / 2 | 1023/1024 (99.9%) | 56.9% | 20.189 | 7.592 |

## Risk-Level Outcomes

| round | risk | violation rate | eligible | always C / R / mixed |
|---:|---|---:|---:|---:|
| 1 | `early_spoiler` | 0.238 | 1 | 11 / 3 / 2 |
| 1 | `fake_commit` | 0.402 | 1 | 8 / 4 / 4 |
| 1 | `hidden_fields` | 0.875 | 0 | 2 / 14 / 0 |
| 1 | `schema_bypass` | 0.582 | 1 | 4 / 3 / 9 |
| 2 | `early_spoiler` | 0.074 | 0 | 14 / 1 / 1 |
| 2 | `fake_commit` | 0.441 | 0 | 8 / 6 / 2 |
| 2 | `hidden_fields` | 0.914 | 1 | 0 / 14 / 2 |
| 2 | `schema_bypass` | 0.500 | 0 | 7 / 7 / 2 |
| 3 | `early_spoiler` | 0.387 | 0 | 9 / 6 / 1 |
| 3 | `fake_commit` | 0.500 | 0 | 8 / 8 / 0 |
| 3 | `hidden_fields` | 0.625 | 0 | 6 / 10 / 0 |
| 3 | `schema_bypass` | 0.613 | 0 | 6 / 9 / 1 |

## Frozen Gate Checks

- raw eligible prompts: `4`
- capped selected prompts: `4`
- template families: `4`
- risk counts: `{'early_spoiler': 1, 'fake_commit': 1, 'hidden_fields': 1, 'schema_bypass': 1}`
- mechanism counts: `{'boundary_tradeoff': 1, 'equal_authority_conflict': 3}`
- non-lottery prompts: `4`

| check | pass |
|---|---:|
| `selected_prompts` | `False` |
| `risk_coverage` | `False` |
| `template_families` | `False` |
| `mechanism_coverage` | `True` |
| `non_lottery_support` | `False` |

## Post-Discovery Candidate Diagnostic

This diagnostic was specified after observing the discovery yield and does not alter the frozen gate. Exact candidate matching first parses each generated JSON object and then compares it structurally with candidate A and B.

In the round-three non-tie lottery prompts, the model selected the candidate with the larger stated weight in `736/768` exact-candidate trajectories (`95.8%`). Yet round three produced zero eligible prompts. This pattern is consistent with near-deterministic selection of the larger-weight candidate rather than repeated stochastic draws.

For equal-weight lottery prompts, candidate A was selected in `166/255` exact-candidate trajectories (`65.1%`); `255/256` outputs matched one of the two candidates.

## Interpretation Boundary

v4c establishes a discovery-yield failure for these four synthetic governance families, this frozen candidate construction, Qwen3-4B revision, and sampling configuration. Together with v4b, it shows that within-prompt outcome contrast is scarce at 4B in the tested tasks even after model-native redesign.

It does not show that Qwen3-4B is generally deterministic, that larger models are safer, or that residual probes lack added value. The latter question is untested in v4c because the precondition for a valid contrastive confirmatory benchmark failed.

## Reproducibility and Cost

- trajectories sampled: `3072`
- generated tokens: `79673`
- measured generation time: `63.15` minutes
- maximum allocated VRAM: `7.592` GiB
- model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`, unquantized FP16
- sampling: `temperature=0.8`, `top_p=0.95`, `max_new_tokens=48`, 16 trajectories per candidate

See `results/PREREGISTERED_V4C_DISCOVERY_PROTOCOL.md` and the machine-readable `results/v4c_discovery_final.json`.
