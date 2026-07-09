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
| `logit_lens` | expected_rollback | all | 800 | 400 | 0.486 | [0.443, 0.529] | 0.453 | 0.875 |
| `dense_jlens` | expected_rollback | all | 800 | 400 | 0.416 | [0.377, 0.468] | 0.412 | 0.650 |
| `jvp_lens` | expected_rollback | all | 800 | 400 | 0.317 | [0.277, 0.357] | 0.381 | 0.912 |
| `linear_probe` | expected_rollback | all | 800 | 400 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `linear_probe` | expected_rollback | test | 160 | 80 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `keyword_target_present` | semantic_risk | all | 800 | 400 | 0.500 | [0.466, 0.534] | 0.500 | 0.900 |
| `keyword_hit_count` | semantic_risk | all | 800 | 400 | 0.625 | [0.588, 0.661] | 0.750 | 0.900 |
| `logit_lens` | semantic_risk | all | 800 | 400 | 0.486 | [0.443, 0.529] | 0.453 | 0.875 |
| `dense_jlens` | semantic_risk | all | 800 | 400 | 0.416 | [0.377, 0.468] | 0.412 | 0.650 |
| `jvp_lens` | semantic_risk | all | 800 | 400 | 0.317 | [0.277, 0.357] | 0.381 | 0.912 |
| `linear_probe` | semantic_risk | all | 800 | 400 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `linear_probe` | semantic_risk | test | 160 | 80 | 1.000 | [1.000, 1.000] | 1.000 | 0.000 |
| `keyword_target_present` | generated_rollback | all | 800 | 360 | 0.551 | [0.518, 0.580] | 0.547 | 0.873 |
| `keyword_hit_count` | generated_rollback | all | 800 | 360 | 0.647 | [0.611, 0.678] | 0.704 | 0.873 |
| `logit_lens` | generated_rollback | all | 800 | 360 | 0.451 | [0.415, 0.492] | 0.390 | 0.886 |
| `dense_jlens` | generated_rollback | all | 800 | 360 | 0.501 | [0.464, 0.542] | 0.404 | 0.636 |
| `jvp_lens` | generated_rollback | all | 800 | 360 | 0.411 | [0.374, 0.451] | 0.403 | 0.875 |
| `linear_probe` | generated_rollback | all | 800 | 360 | 0.858 | [0.832, 0.886] | 0.774 | 0.170 |
| `linear_probe` | generated_rollback | test | 160 | 75 | 0.886 | [0.826, 0.935] | 0.849 | 0.141 |

## Linear Probe

- probe label: `expected_rollback`
- selected layer: `0`
- validation AUC: `1.000`
- test AUC: `1.000`

## Runtime Cost

| component | seconds | seconds/case |
|---|---:|---:|
| forward | 27.271 | 0.03409 |
| logit_lens_scoring | 80.475 | 0.10059 |
| dense_jlens_scoring | 88.494 | 0.11062 |
| jvp_lens_scoring | 149.775 | 0.18722 |
| generation | 685.524 | 0.85690 |
| linear_probe_training | 2.726 | 0.00341 |

## Falsification Notes

- If `keyword_target_present` matches or beats internal readouts, the result is likely surface-token copying.
- If `linear_probe` beats dense J-lens at far lower cost, dense J-lens is not the practical governance monitor.
- If generated rollback has too few positives, report expected-rollback and generated-rollback metrics separately.
