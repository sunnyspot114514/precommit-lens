# v4 Confirmatory Trajectory Results

- gate: **FAIL**
- trajectories / prompts: `1088` / `34`
- fresh-seed mixed test prompts: `9/9` across `3` risks
- winning checkpoints: `[]`
- consecutive winning pairs: `[]`
- violating trajectories without a semantic landing: `0`

## Primary Checkpoint Curve

| checkpoint | prompts | risks | residual L18 | TF-IDF | next-token | envelope | delta [95% CI] | judge |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 3 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 [0.000, 0.000] | 0.500 |
| 2 | 9 | 3 | 0.527 | 0.558 | 0.500 | 0.558 | -0.031 [-0.095, 0.019] | 0.509 |
| 4 | 9 | 3 | 0.580 | 0.610 | 0.512 | 0.610 | -0.030 [-0.093, 0.009] | 0.458 |
| 6 | 9 | 3 | 0.676 | 0.661 | 0.449 | 0.661 | 0.015 [-0.095, 0.126] | 0.381 |
| 8 | 9 | 3 | 0.823 | 0.817 | 0.545 | 0.817 | 0.006 [0.000, 0.017] | 0.482 |
| 10 | 9 | 3 | 0.817 | 0.814 | 0.739 | 0.814 | 0.003 [0.000, 0.008] | 0.413 |
| 12 | 5 | 3 | 0.718 | 0.715 | 0.600 | 0.715 | 0.003 [-0.037, 0.045] | 0.425 |
| 16 | 4 | 2 | 0.600 | 0.577 | 0.500 | 0.577 | 0.023 [0.000, 0.047] | NA |
| 24 | 3 | 2 | 0.582 | 0.582 | 0.554 | 0.582 | 0.000 [0.000, 0.000] | NA |

## Secondary Layer Diagnostic

| checkpoint | primary L18 | best captured layer | best residual AUC |
|---:|---:|---:|---:|
| 2 | 0.527 | 27 | 0.568 |
| 4 | 0.580 | 24 | 0.610 |
| 6 | 0.676 | 0 | 0.713 |
| 8 | 0.823 | 6 | 0.825 |
| 10 | 0.817 | 27 | 0.819 |
| 12 | 0.718 | 0 | 0.718 |

## Fresh-Seed Contrast Yield

| split | mixed / total | risk coverage |
|---|---:|---:|
| train | 16 / 16 | 3 |
| validation | 9 / 9 | 3 |
| test | 9 / 9 | 3 |

## Measured Cost

- Six-layer, nine-checkpoint capture/plain generation ratio: `1.014` (95% paired CI `0.999`-`1.029`; `18` paired runs).
- Mean capture overhead: `11.8` ms/trajectory.
- Prefix judge: `4.043` seconds for `160` unique prefixes (`25.3` ms/prefix).
- Layer-18 logistic scoring: `0.050` microseconds/row.

Secondary layers are post-hoc diagnostics. They cannot override the frozen layer-18 gate.

## Interpretation Discipline

This experiment tests extraction latency and cost, not information exclusivity. The selected prompts were adaptively screened for stochastic outcome contrast, so the results do not estimate violation prevalence. Per-risk and secondary-layer results are diagnostics and cannot override the frozen primary gate.
