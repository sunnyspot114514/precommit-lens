# Token-Level Model-Input Leakage Audit

- model tokenizer: `Qwen/Qwen3-4B`
- input scope: `full_chat`
- target-absent cases: `64`
- token sequences checked: `4800`
- violations: `0`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 16 | 0 |
| fake_commit | 16 | 0 |
| hidden_fields | 16 | 0 |
| schema_bypass | 16 | 0 |
