# v4c Post-Hoc Appendix Diagnostics

These diagnostics were frozen after the v4c discovery result. They do not reopen the failed gate, select a new prompt pool, or authorize residual probes.

## Temperature Sensitivity

The same 64 round-one prompts were evaluated on the same Qwen3-4B FP16 model. The original T=0.8 run is reused; T=1.2 and T=1.5 use fresh seeds.

| temperature | seed start | eligible | always C / R / mixed | candidate match | A/B-switch prompts | tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 11000000 | 3/64 | 25 / 24 / 15 | 1017/1024 (99.3%) | 15/64 | 21.285 |
| 1.2 | 21000000 | 9/64 | 18 / 22 / 24 | 1015/1024 (99.1%) | 23/64 | 21.299 |
| 1.5 | 22000000 | 11/64 | 17 / 21 / 26 | 1008/1024 (98.4%) | 26/64 | 21.800 |

## Deployment-State Model Controls

These are Ollama Q4_K_M deployment artifacts with native chat renderers. They are descriptive model controls, not an unconfounded FP16 scale curve.

| model | role | digest | eligible | always C / R / mixed | candidate match | A/B-switch prompts | tok/s |
|---|---|---|---:|---:|---:|---:|---:|
| `gemma4:e2b` | cross-family | `7fbdbf8f5e45` | 3/64 | 29 / 25 / 10 | 973/1024 (95.0%) | 6/64 | 40.463 |
| `qwen3.5:4b` | same-family newer generation | `2a654d98e6fb` | 34/64 | 7 / 1 / 56 | 962/1024 (93.9%) | 52/64 | 38.364 |

## Risk-Level Outcomes

| condition | risk | violation rate | eligible | always C / R / mixed |
|---|---|---:|---:|---:|
| `qwen3_4b_t0p8` | `early_spoiler` | 0.238 | 1 | 11 / 3 / 2 |
| `qwen3_4b_t0p8` | `fake_commit` | 0.402 | 1 | 8 / 4 / 4 |
| `qwen3_4b_t0p8` | `hidden_fields` | 0.875 | 0 | 2 / 14 / 0 |
| `qwen3_4b_t0p8` | `schema_bypass` | 0.582 | 1 | 4 / 3 / 9 |
| `qwen3_4b_t1p2` | `early_spoiler` | 0.227 | 1 | 11 / 2 / 3 |
| `qwen3_4b_t1p2` | `fake_commit` | 0.426 | 3 | 4 / 3 / 9 |
| `qwen3_4b_t1p2` | `hidden_fields` | 0.879 | 0 | 1 / 14 / 1 |
| `qwen3_4b_t1p2` | `schema_bypass` | 0.547 | 5 | 2 / 3 / 11 |
| `qwen3_4b_t1p5` | `early_spoiler` | 0.199 | 2 | 11 / 2 / 3 |
| `qwen3_4b_t1p5` | `fake_commit` | 0.445 | 5 | 3 / 4 / 9 |
| `qwen3_4b_t1p5` | `hidden_fields` | 0.883 | 0 | 1 / 13 / 2 |
| `qwen3_4b_t1p5` | `schema_bypass` | 0.547 | 4 | 2 / 2 / 12 |
| `gemma4_e2b_t0p8` | `early_spoiler` | 0.141 | 1 | 12 / 0 / 4 |
| `gemma4_e2b_t0p8` | `fake_commit` | 0.805 | 2 | 2 / 11 / 3 |
| `gemma4_e2b_t0p8` | `hidden_fields` | 0.941 | 0 | 0 / 14 / 2 |
| `gemma4_e2b_t0p8` | `schema_bypass` | 0.004 | 0 | 15 / 0 / 1 |
| `qwen35_4b_t0p8` | `early_spoiler` | 0.598 | 12 | 0 / 0 / 16 |
| `qwen35_4b_t0p8` | `fake_commit` | 0.461 | 8 | 1 / 0 / 15 |
| `qwen35_4b_t0p8` | `hidden_fields` | 0.641 | 10 | 1 / 1 / 14 |
| `qwen35_4b_t0p8` | `schema_bypass` | 0.180 | 4 | 5 / 0 / 11 |

## Interpretation Boundary

The eligible counts across Qwen3-4B temperatures 0.8, 1.2, and 1.5 are `3`, `9`, and `11`. Mixed counts are `15`, `24`, and `26`.

At T=0.8, the deployment controls yield `3/64` eligible prompts for Gemma and `34/64` for Qwen3.5. These counts describe the frozen deployments only.

A post-hoc output-fidelity audit finds both exact candidate A and B in `6/64` Gemma prompts and `52/64` Qwen3.5 prompts. The Qwen3.5 contrast therefore cannot be explained only by its non-candidate outputs.

No appendix result changes the v4c `DISCOVERY YIELD FAIL`. Temperature, backend, quantization, native rendering, and model generation prevent a universal determinism claim or a causal family comparison.

See `results/PREREGISTERED_V4C_APPENDIX_DIAGNOSTICS.md` and `results/v4c_appendix_diagnostics.json`.
