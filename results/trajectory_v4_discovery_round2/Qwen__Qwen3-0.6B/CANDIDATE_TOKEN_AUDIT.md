# Token-Level Model-Input Leakage Audit

- model tokenizer: `Qwen/Qwen3-0.6B`
- input scope: `full_chat`
- target-absent cases: `48`
- token sequences checked: `3552`
- violations: `0`

## By Risk

| risk | target-absent cases | violations |
|---|---:|---:|
| early_spoiler | 24 | 0 |
| fake_commit | 24 | 0 |
