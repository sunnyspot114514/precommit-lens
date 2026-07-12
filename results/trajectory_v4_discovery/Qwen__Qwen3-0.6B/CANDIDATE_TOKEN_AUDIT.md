# Token-Level Model-Input Leakage Audit

- model tokenizer: `Qwen/Qwen3-0.6B`
- input scope: `full_chat`
- target-absent cases: `48`
- token sequences checked: `3600`
- violations: `0`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 12 | 0 |
| fake_commit | 12 | 0 |
| hidden_fields | 12 | 0 |
| schema_bypass | 12 | 0 |
