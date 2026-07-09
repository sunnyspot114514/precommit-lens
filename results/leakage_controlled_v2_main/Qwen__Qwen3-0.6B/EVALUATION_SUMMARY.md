# Leakage-Controlled Probe Evaluation

- model: `Qwen/Qwen3-0.6B`
- cases: `800`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- generated outputs: `True`
- dense lens: `results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_lens_smoke.npz`
- JVP layers: `[0, 7, 14, 21, 27]`

## Main Metrics

| method | label | split | n | pos | AUC | 95% CI | AUPRC | FPR@90%TPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `keyword_target_present` | expected_rollback | all | 800 | 400 | 0.500 | [0.466, 0.534] | 0.500 | 0.900 |
| `keyword_hit_count` | expected_rollback | all | 800 | 400 | 0.625 | [0.588, 0.661] | 0.750 | 0.900 |
| `logit_lens` | expected_rollback | all | 800 | 400 | 0.466 | [0.423, 0.508] | 0.444 | 0.875 |
| `dense_jlens` | expected_rollback | all | 800 | 400 | 0.416 | [0.377, 0.468] | 0.412 | 0.650 |
| `jvp_lens` | expected_rollback | all | 800 | 400 | 0.312 | [0.273, 0.353] | 0.379 | 0.912 |
| `linear_probe` | expected_rollback | all | 800 | 400 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `linear_probe` | expected_rollback | test | 160 | 80 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `keyword_target_present` | semantic_risk | all | 800 | 400 | 0.500 | [0.466, 0.534] | 0.500 | 0.900 |
| `keyword_hit_count` | semantic_risk | all | 800 | 400 | 0.625 | [0.588, 0.661] | 0.750 | 0.900 |
| `logit_lens` | semantic_risk | all | 800 | 400 | 0.466 | [0.423, 0.508] | 0.444 | 0.875 |
| `dense_jlens` | semantic_risk | all | 800 | 400 | 0.416 | [0.377, 0.468] | 0.412 | 0.650 |
| `jvp_lens` | semantic_risk | all | 800 | 400 | 0.312 | [0.273, 0.353] | 0.379 | 0.912 |
| `linear_probe` | semantic_risk | all | 800 | 400 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `linear_probe` | semantic_risk | test | 160 | 80 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `keyword_target_present` | generated_rollback | all | 800 | 345 | 0.570 | [0.539, 0.601] | 0.543 | 0.881 |
| `keyword_hit_count` | generated_rollback | all | 800 | 345 | 0.667 | [0.631, 0.701] | 0.706 | 0.881 |
| `logit_lens` | generated_rollback | all | 800 | 345 | 0.445 | [0.408, 0.484] | 0.372 | 0.890 |
| `dense_jlens` | generated_rollback | all | 800 | 345 | 0.515 | [0.478, 0.557] | 0.395 | 0.648 |
| `jvp_lens` | generated_rollback | all | 800 | 345 | 0.400 | [0.359, 0.441] | 0.383 | 0.901 |
| `linear_probe` | generated_rollback | all | 800 | 345 | 0.850 | [0.824, 0.876] | 0.757 | 0.527 |
| `linear_probe` | generated_rollback | test | 160 | 74 | 0.881 | [0.818, 0.929] | 0.839 | 0.151 |

## Linear Probe

- probe label: `expected_rollback`
- selected layer: `0`
- validation AUC: `1.000`
- test AUC: `1.000`

## Runtime Cost

| component | seconds | seconds/case |
|---|---:|---:|
| forward | 26.446 | 0.03306 |
| logit_lens_scoring | 75.999 | 0.09500 |
| dense_jlens_scoring | 83.512 | 0.10439 |
| jvp_lens_scoring | 145.358 | 0.18170 |
| generation | 668.900 | 0.83613 |
| linear_probe_training | 2.554 | 0.00319 |

## Falsification Notes

- If `keyword_target_present` matches or beats internal readouts, the result is likely surface-token copying.
- If `linear_probe` beats dense J-lens at far lower cost, dense J-lens is not the practical governance monitor.
- If generated rollback has too few positives, report expected-rollback and generated-rollback metrics separately.
