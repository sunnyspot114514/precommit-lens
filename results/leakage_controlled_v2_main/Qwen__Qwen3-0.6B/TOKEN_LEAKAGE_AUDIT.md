# Token-Level Prompt Leakage Audit

- model tokenizer: `Qwen/Qwen3-0.6B`
- target-absent cases: `400`
- token sequences checked: `30000`
- violations: `0`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 100 | 0 |
| fake_commit | 100 | 0 |
| hidden_fields | 100 | 0 |
| schema_bypass | 100 | 0 |
