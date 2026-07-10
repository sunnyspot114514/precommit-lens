# Paired Intervention Statistics

All differences preserve the per-prompt baseline/suppress/sham pairing. The p-value is the exact two-sided McNemar test on discordant outcomes.

| risk | condition | n | baseline | suppress | sham | suppress-sham | 95% CI | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| early_spoiler | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 |
| early_spoiler | `concept_present_target_absent` | 50 | 1.000 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 |
| fake_commit | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 |
| fake_commit | `concept_present_target_absent` | 50 | 0.800 | 1.000 | 0.760 | 0.240 | [0.120, 0.360] | 0.000 |
| hidden_fields | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 |
| hidden_fields | `concept_present_target_absent` | 50 | 0.900 | 0.900 | 0.760 | 0.140 | [0.060, 0.240] | 0.016 |
| schema_bypass | `concept_absent_target_absent` | 50 | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] | 1.000 |
| schema_bypass | `concept_present_target_absent` | 50 | 0.500 | 0.600 | 0.440 | 0.160 | [0.040, 0.280] | 0.039 |

A suppression effect requires a negative suppress-minus-sham difference whose paired interval excludes zero.
