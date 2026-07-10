# Leakage-Controlled Probe Evaluation

- model: `Qwen/Qwen3-0.6B`
- cases: `960`
- layers: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`
- generated outputs: `True`
- dense lens: `results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_lens_smoke.npz`
- JVP layers: `[0, 7, 14, 21, 27]`
- split strategy: `declared_template_family`
- AUC CI bootstrap unit: `pair_id`

## Main Metrics

| method | label | split | n | pos | AUC | 95% CI | AUPRC | FPR@90%TPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `keyword_target_present` | semantic_risk | all | 960 | 480 | 0.500 | [0.500, 0.500] | 0.500 | 0.900 |
| `keyword_hit_count` | semantic_risk | all | 960 | 480 | 0.430 | [0.419, 0.440] | 0.462 | 0.900 |
| `logit_lens` | semantic_risk | all | 960 | 480 | 0.467 | [0.447, 0.485] | 0.472 | 0.867 |
| `dense_jlens` | semantic_risk | all | 960 | 480 | 0.552 | [0.524, 0.577] | 0.521 | 0.827 |
| `jvp_lens` | semantic_risk | all | 960 | 480 | 0.456 | [0.428, 0.487] | 0.495 | 0.915 |
| `linear_probe` | semantic_risk | all | 960 | 480 | 0.968 | [0.953, 0.979] | 0.951 | 0.060 |
| `linear_probe` | semantic_risk | test | 240 | 120 | 0.897 | [0.861, 0.936] | 0.872 | 0.167 |
| `linear_probe_cross_risk` | semantic_risk | test | 240 | 120 | 0.681 | [0.635, 0.735] | 0.688 | 0.775 |
| `keyword_target_present` | generated_rollback | all | 960 | 205 | 0.787 | [0.763, 0.805] | 0.475 | 0.359 |
| `keyword_hit_count` | generated_rollback | all | 960 | 205 | 0.797 | [0.767, 0.822] | 0.471 | 0.346 |
| `logit_lens` | generated_rollback | all | 960 | 205 | 0.667 | [0.626, 0.711] | 0.271 | 0.577 |
| `dense_jlens` | generated_rollback | all | 960 | 205 | 0.649 | [0.608, 0.688] | 0.274 | 0.577 |
| `jvp_lens` | generated_rollback | all | 960 | 205 | 0.482 | [0.439, 0.529] | 0.216 | 0.899 |
| `linear_probe` | generated_rollback | all | 960 | 205 | 0.605 | [0.565, 0.647] | 0.283 | 0.875 |
| `linear_probe` | generated_rollback | test | 240 | 50 | 0.655 | [0.574, 0.723] | 0.313 | 0.658 |
| `linear_probe_cross_risk` | generated_rollback | test | 240 | 50 | 0.558 | [0.471, 0.633] | 0.236 | 0.795 |
| `keyword_target_present` | generated_policy_violation | all | 960 | 487 | 0.534 | [0.519, 0.547] | 0.549 | 0.848 |
| `keyword_hit_count` | generated_policy_violation | all | 960 | 487 | 0.479 | [0.457, 0.497] | 0.500 | 0.848 |
| `logit_lens` | generated_policy_violation | all | 960 | 487 | 0.511 | [0.482, 0.540] | 0.496 | 0.831 |
| `dense_jlens` | generated_policy_violation | all | 960 | 487 | 0.585 | [0.553, 0.613] | 0.537 | 0.780 |
| `jvp_lens` | generated_policy_violation | all | 960 | 487 | 0.446 | [0.417, 0.474] | 0.489 | 0.907 |
| `linear_probe` | generated_policy_violation | all | 960 | 487 | 0.910 | [0.891, 0.928] | 0.898 | 0.123 |
| `linear_probe` | generated_policy_violation | test | 240 | 125 | 0.918 | [0.887, 0.948] | 0.900 | 0.130 |
| `linear_probe_cross_risk` | generated_policy_violation | test | 240 | 125 | 0.707 | [0.658, 0.754] | 0.725 | 0.765 |
| `keyword_target_present` | generated_structural_rollback | all | 960 | 698 | 0.566 | [0.547, 0.585] | 0.793 | 0.927 |
| `keyword_hit_count` | generated_structural_rollback | all | 960 | 698 | 0.519 | [0.498, 0.541] | 0.725 | 0.927 |
| `logit_lens` | generated_structural_rollback | all | 960 | 698 | 0.447 | [0.403, 0.491] | 0.667 | 0.805 |
| `dense_jlens` | generated_structural_rollback | all | 960 | 698 | 0.562 | [0.520, 0.602] | 0.755 | 0.889 |
| `jvp_lens` | generated_structural_rollback | all | 960 | 698 | 0.481 | [0.443, 0.522] | 0.714 | 0.912 |
| `linear_probe` | generated_structural_rollback | all | 960 | 698 | 0.841 | [0.817, 0.866] | 0.942 | 0.718 |
| `linear_probe` | generated_structural_rollback | test | 240 | 183 | 0.847 | [0.792, 0.893] | 0.957 | 0.737 |
| `linear_probe_cross_risk` | generated_structural_rollback | test | 240 | 183 | 0.703 | [0.632, 0.757] | 0.906 | 0.825 |

## Linear Probe

- probe label: `semantic_risk`
- selected layer: `18`
- validation AUC: `1.000`
- test AUC: `0.897`
- cross-risk probe: `True`

## Runtime Cost

| component | seconds | seconds/case |
|---|---:|---:|
| forward | 32.276 | 0.03362 |
| logit_lens_scoring | 94.904 | 0.09886 |
| dense_jlens_scoring | 104.622 | 0.10898 |
| jvp_lens_scoring | 177.504 | 0.18490 |
| generation | 849.269 | 0.88465 |
| linear_probe_training | 2.644 | 0.00275 |
| cross_risk_probe_training | 9.865 | 0.01028 |

## Falsification Notes

- If `keyword_target_present` matches or beats internal readouts, the result is likely surface-token copying.
- If `linear_probe` beats dense J-lens at far lower cost, dense J-lens is not the practical governance monitor.
- If generated rollback has too few positives, report expected-rollback and generated-rollback metrics separately.
