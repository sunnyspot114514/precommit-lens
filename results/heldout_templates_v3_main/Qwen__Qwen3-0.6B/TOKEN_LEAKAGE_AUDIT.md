# Token-Level Model-Input Leakage Audit

- model tokenizer: `Qwen/Qwen3-0.6B`
- input scope: `full_chat`
- target-absent cases: `480`
- token sequences checked: `36000`
- violations: `0`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 120 | 0 |
| fake_commit | 120 | 0 |
| hidden_fields | 120 | 0 |
| schema_bypass | 120 | 0 |
