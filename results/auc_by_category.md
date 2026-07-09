# AUC by Category

Leakage-controlled falsification summary by risk category, method, and label.

## Per-Risk AUC

| risk | method | label | n | pos | AUC | AUPRC | FPR@90%TPR |
|---|---|---|---:|---:|---:|---:|---:|
| early_spoiler | `keyword_target_present` | expected_rollback | 200 | 100 | 0.500 | 0.510 | 0.890 |
| early_spoiler | `keyword_hit_count` | expected_rollback | 200 | 100 | 0.625 | 0.751 | 0.890 |
| early_spoiler | `logit_lens` | expected_rollback | 200 | 100 | 0.500 | 0.454 | 0.500 |
| early_spoiler | `dense_jlens` | expected_rollback | 200 | 100 | 0.231 | 0.358 | 1.000 |
| early_spoiler | `jvp_lens` | expected_rollback | 200 | 100 | 0.367 | 0.405 | 0.950 |
| early_spoiler | `linear_probe` | expected_rollback | 200 | 100 | 1.000 | 1.000 | 0.000 |
| early_spoiler | `keyword_target_present` | generated_rollback | 200 | 95 | 0.475 | 0.462 | 0.914 |
| early_spoiler | `keyword_hit_count` | generated_rollback | 200 | 95 | 0.588 | 0.660 | 0.914 |
| early_spoiler | `logit_lens` | generated_rollback | 200 | 95 | 0.496 | 0.430 | 0.524 |
| early_spoiler | `dense_jlens` | generated_rollback | 200 | 95 | 0.246 | 0.341 | 1.000 |
| early_spoiler | `jvp_lens` | generated_rollback | 200 | 95 | 0.333 | 0.369 | 1.000 |
| early_spoiler | `linear_probe` | generated_rollback | 200 | 95 | 0.975 | 0.964 | 0.048 |
| fake_commit | `keyword_target_present` | expected_rollback | 200 | 100 | 0.500 | 0.510 | 0.890 |
| fake_commit | `keyword_hit_count` | expected_rollback | 200 | 100 | 0.625 | 0.751 | 0.890 |
| fake_commit | `logit_lens` | expected_rollback | 200 | 100 | 0.736 | 0.789 | 0.550 |
| fake_commit | `dense_jlens` | expected_rollback | 200 | 100 | 0.551 | 0.519 | 0.500 |
| fake_commit | `jvp_lens` | expected_rollback | 200 | 100 | 0.530 | 0.587 | 0.900 |
| fake_commit | `linear_probe` | expected_rollback | 200 | 100 | 1.000 | 1.000 | 0.000 |
| fake_commit | `keyword_target_present` | generated_rollback | 200 | 85 | 0.526 | 0.449 | 0.878 |
| fake_commit | `keyword_hit_count` | generated_rollback | 200 | 85 | 0.641 | 0.676 | 0.878 |
| fake_commit | `logit_lens` | generated_rollback | 200 | 85 | 0.742 | 0.719 | 0.565 |
| fake_commit | `dense_jlens` | generated_rollback | 200 | 85 | 0.522 | 0.444 | 0.565 |
| fake_commit | `jvp_lens` | generated_rollback | 200 | 85 | 0.550 | 0.490 | 0.957 |
| fake_commit | `linear_probe` | generated_rollback | 200 | 85 | 0.934 | 0.838 | 0.130 |
| hidden_fields | `keyword_target_present` | expected_rollback | 200 | 100 | 0.500 | 0.510 | 0.890 |
| hidden_fields | `keyword_hit_count` | expected_rollback | 200 | 100 | 0.625 | 0.751 | 0.890 |
| hidden_fields | `logit_lens` | expected_rollback | 200 | 100 | 0.500 | 0.454 | 0.500 |
| hidden_fields | `dense_jlens` | expected_rollback | 200 | 100 | 0.450 | 0.433 | 0.550 |
| hidden_fields | `jvp_lens` | expected_rollback | 200 | 100 | 0.193 | 0.346 | 0.900 |
| hidden_fields | `linear_probe` | expected_rollback | 200 | 100 | 1.000 | 1.000 | 0.000 |
| hidden_fields | `keyword_target_present` | generated_rollback | 200 | 70 | 0.390 | 0.304 | 0.946 |
| hidden_fields | `keyword_hit_count` | generated_rollback | 200 | 70 | 0.459 | 0.402 | 0.946 |
| hidden_fields | `logit_lens` | generated_rollback | 200 | 70 | 0.484 | 0.313 | 0.615 |
| hidden_fields | `dense_jlens` | generated_rollback | 200 | 70 | 0.420 | 0.289 | 0.654 |
| hidden_fields | `jvp_lens` | generated_rollback | 200 | 70 | 0.313 | 0.256 | 0.885 |
| hidden_fields | `linear_probe` | generated_rollback | 200 | 70 | 0.898 | 0.798 | 0.192 |
| schema_bypass | `keyword_target_present` | expected_rollback | 200 | 100 | 0.500 | 0.510 | 0.890 |
| schema_bypass | `keyword_hit_count` | expected_rollback | 200 | 100 | 0.625 | 0.751 | 0.890 |
| schema_bypass | `logit_lens` | expected_rollback | 200 | 100 | 0.905 | 0.785 | 0.150 |
| schema_bypass | `dense_jlens` | expected_rollback | 200 | 100 | 0.450 | 0.433 | 0.550 |
| schema_bypass | `jvp_lens` | expected_rollback | 200 | 100 | 0.030 | 0.314 | 1.000 |
| schema_bypass | `linear_probe` | expected_rollback | 200 | 100 | 1.000 | 1.000 | 0.000 |
| schema_bypass | `keyword_target_present` | generated_rollback | 200 | 110 | 0.803 | 0.844 | 0.622 |
| schema_bypass | `keyword_hit_count` | generated_rollback | 200 | 110 | 0.841 | 0.891 | 0.622 |
| schema_bypass | `logit_lens` | generated_rollback | 200 | 110 | 0.710 | 0.696 | 0.333 |
| schema_bypass | `dense_jlens` | generated_rollback | 200 | 110 | 0.827 | 0.828 | 0.278 |
| schema_bypass | `jvp_lens` | generated_rollback | 200 | 110 | 0.367 | 0.589 | 1.000 |
| schema_bypass | `linear_probe` | generated_rollback | 200 | 110 | 0.765 | 0.729 | 0.278 |

## Condition Means

| risk | condition | n | gen rollback | target present | keyword hits | logit | dense | JVP | linear |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| early_spoiler | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | -5.718 | -3.138 | -7.271 | -95.971 |
| early_spoiler | `concept_present_target_absent` | 50 | 1.000 | 0.000 | 0.000 | -5.426 | -3.798 | -7.515 | 21.506 |
| early_spoiler | `concept_target_present` | 50 | 0.900 | 1.000 | 6.000 | -5.254 | -2.639 | -6.756 | 64.241 |
| early_spoiler | `target_present_concept_absent` | 50 | 0.000 | 1.000 | 4.000 | -2.127 | 0.000 | -4.153 | -78.508 |
| fake_commit | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | -4.919 | -5.889 | -8.408 | -84.464 |
| fake_commit | `concept_present_target_absent` | 50 | 0.800 | 0.000 | 0.000 | -4.840 | -5.416 | -7.429 | 32.914 |
| fake_commit | `concept_target_present` | 50 | 0.900 | 1.000 | 7.000 | -4.547 | -1.622 | -6.448 | 96.737 |
| fake_commit | `target_present_concept_absent` | 50 | 0.000 | 1.000 | 5.000 | -4.693 | -1.212 | -6.720 | -19.544 |
| hidden_fields | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | -4.951 | -6.429 | -5.861 | -41.231 |
| hidden_fields | `concept_present_target_absent` | 50 | 0.900 | 0.000 | 0.000 | -4.896 | -5.454 | -6.722 | 64.115 |
| hidden_fields | `concept_target_present` | 50 | 0.500 | 1.000 | 6.000 | -4.864 | -5.366 | -7.032 | 68.569 |
| hidden_fields | `target_present_concept_absent` | 50 | 0.000 | 1.000 | 5.000 | -3.218 | 0.000 | -4.646 | -32.025 |
| schema_bypass | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | -5.026 | -6.098 | -3.798 | -90.949 |
| schema_bypass | `concept_present_target_absent` | 50 | 0.500 | 0.000 | 0.000 | -4.782 | -5.633 | -6.307 | 89.383 |
| schema_bypass | `concept_target_present` | 50 | 1.000 | 1.000 | 8.000 | -4.845 | -5.266 | -7.236 | 67.372 |
| schema_bypass | `target_present_concept_absent` | 50 | 0.700 | 1.000 | 5.000 | -4.883 | -2.617 | -2.270 | -67.985 |

## Interpretation Checks

- early_spoiler: dense J-lens does not beat the keyword-hit baseline on expected rollback (dense 0.231, keyword 0.625).
- early_spoiler: linear probe dominates dense J-lens on expected rollback (linear 1.000, dense 0.231).
- fake_commit: dense J-lens does not beat the keyword-hit baseline on expected rollback (dense 0.551, keyword 0.625).
- fake_commit: linear probe dominates dense J-lens on expected rollback (linear 1.000, dense 0.551).
- hidden_fields: dense J-lens does not beat the keyword-hit baseline on expected rollback (dense 0.450, keyword 0.625).
- hidden_fields: linear probe dominates dense J-lens on expected rollback (linear 1.000, dense 0.450).
- schema_bypass: dense J-lens does not beat the keyword-hit baseline on expected rollback (dense 0.450, keyword 0.625).
- schema_bypass: linear probe dominates dense J-lens on expected rollback (linear 1.000, dense 0.450).
