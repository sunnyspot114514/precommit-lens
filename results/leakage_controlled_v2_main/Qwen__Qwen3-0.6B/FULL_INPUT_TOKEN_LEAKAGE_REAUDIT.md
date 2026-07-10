# Token-Level Model-Input Leakage Audit

- model tokenizer: `Qwen/Qwen3-0.6B`
- input scope: `full_chat`
- target-absent cases: `400`
- token sequences checked: `30000`
- violations: `100`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 100 | 0 |
| fake_commit | 100 | 100 |
| hidden_fields | 100 | 0 |
| schema_bypass | 100 | 0 |

## Violations

- `fake_commit_000_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_000_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_001_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_001_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_002_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_002_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_003_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_003_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_004_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_004_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_005_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_005_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_006_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_006_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_007_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_007_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_008_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_008_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_009_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_009_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_010_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_010_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_011_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_011_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_012_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_012_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_013_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_013_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_014_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_014_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_015_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_015_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_016_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_016_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_017_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_017_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_018_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_018_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_019_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_019_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_020_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_020_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_021_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_021_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_022_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_022_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_023_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_023_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_024_concept_present_target_absent` target_term ` validator` tokens=[22935] at=24
- `fake_commit_024_concept_absent_target_absent` target_term ` validator` tokens=[22935] at=24
- ... 50 more
