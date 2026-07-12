# v4d Confirmatory Trajectory Results

- gate: **FAIL**
- trajectories / prompts: `1056` / `33`
- fresh-seed mixed test prompts: `8/8` across `3` risks
- winning checkpoints: `[]`
- consecutive winning pairs: `[]`
- violating trajectories without a semantic landing: `0`

## Primary Checkpoint Curve

| checkpoint | prompts | risks | residual L21 | TF-IDF | next-token | envelope | delta [95% CI] | judge |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 2 | 8 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 4 | 8 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 6 | 5 | 2 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 8 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 10 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 12 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 16 | 3 | 1 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | NA |
| 24 | 2 | 1 | NA | NA | NA | NA | NA [NA, NA] | NA |

## Secondary Layer Diagnostic

| checkpoint | primary L21 | best captured layer | best residual AUC |
|---:|---:|---:|---:|
| 2 | 0.500 | 0 | 0.500 |
| 4 | 0.500 | 0 | 0.500 |
| 6 | 0.500 | 0 | 0.500 |
| 8 | 0.500 | 0 | 0.500 |
| 10 | 0.500 | 0 | 0.500 |
| 12 | 0.500 | 0 | 0.500 |

## Fresh-Seed Contrast Yield

| split | mixed / total | risk coverage |
|---|---:|---:|
| train | 17 / 17 | 3 |
| validation | 8 / 8 | 3 |
| test | 8 / 8 | 3 |

## Measured Cost

- 6-layer, 9-checkpoint capture/plain generation ratio: `1.005` (95% paired CI `0.953`-`1.062`; `16` paired runs).
- Mean capture overhead: `7.3` ms/trajectory.
- Prefix judge: `14.332` seconds for `62` unique prefixes (`231.2` ms/prefix).
- Layer-21 logistic scoring: `0.061` microseconds/row.

Secondary layers are post-hoc diagnostics. They cannot override the frozen layer-21 gate.

## Interpretation Discipline

This experiment tests extraction latency and cost, not information exclusivity. The selected prompts were adaptively screened for stochastic outcome contrast, so the results do not estimate violation prevalence. Per-risk and secondary-layer results are diagnostics and cannot override the frozen primary gate.
