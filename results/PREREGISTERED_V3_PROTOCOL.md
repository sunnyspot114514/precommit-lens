# Pre-Registered Held-Out-Template v3 Protocol

This protocol was written before the v3 model run. Existing v2 results were
used to choose the falsification questions, but v3 thresholds and reporting
rules are frozen before inspecting v3 outcomes.

## Primary Question

Does a residual-stream linear probe predict semantic pre-commit risk on
entirely held-out prompt-template families, and does it add predictive value
beyond a prompt-text-only classifier?

## Dataset

- Four risk families: `early_spoiler`, `fake_commit`, `hidden_fields`, and
  `schema_bypass`.
- Twelve complete phrasing families per risk.
- Five matched content groups per phrasing family.
- Four 2x2 conditions per group: concept present/absent crossed with watched
  target words present/absent.
- Complete template families, matched groups, and content instances cannot
  cross train/validation/test boundaries.
- The leakage audit checks the complete rendered system+user chat input.

## Labels

- Primary constructed label: `semantic_risk`.
- `expected_rollback` is identical by construction and is not reported as
  independent evidence.
- Observed endpoints are reported separately:
  - legacy lexical-substring rollback;
  - risk-specific generated policy violation;
  - JSON parse/contract plus policy rollback.

## Methods

- target-word presence and hit-count baselines;
- word unigram/bigram TF-IDF logistic classifier;
- character 3-5 gram TF-IDF logistic classifier;
- label-shuffled text-classifier control;
- logit lens;
- saved dense J-lens;
- selected-layer JVP lens;
- residual linear probe selected only on validation template families;
- leave-one-risk-family-out residual probe.

## Statistics

- Main evaluation uses only held-out test template families.
- AUC confidence intervals resample complete `pair_id` clusters.
- Template-family cluster intervals are reported separately.
- Residual-minus-text AUC uses a paired cluster bootstrap.
- The 2x2 analysis reports concept, target-token, and interaction effects with
  matched-pair bootstrap intervals.
- Below-chance AUC remains a failure under the pre-specified score direction;
  `max(AUC, 1-AUC)` is appendix-style diagnostic information only.

## Scale Gate

Local model replication is allowed only if all three conditions hold:

1. Residual-probe semantic-risk test AUC is at least `0.80`, with pair-cluster
   95% CI lower bound at least `0.65`.
2. Residual AUC exceeds the validation-selected text-only baseline by at least
   `0.03`, with paired 95% CI lower bound above `0`.
3. Residual-probe AUC for generated policy violation is at least `0.75`, with
   pair-cluster 95% CI lower bound at least `0.60`.

Failure sends the project to a negative/confound-analysis result. Passing the
gate permits local replication on Qwen3.5/Gemma. It does not by itself permit
an RTX Pro 6000 dense-Jacobian scale curve; cloud scaling requires successful
local replication as a separate gate.

## Commands

```powershell
.\.venv-jlens\Scripts\python.exe .\src\build_heldout_template_prompts.py `
  --variants-per-family 5 `
  --tokenizer-model-id Qwen/Qwen3-0.6B `
  --token-audit-json .\results\heldout_templates_v3_main\Qwen__Qwen3-0.6B\token_leakage_audit.json `
  --token-audit-md .\results\heldout_templates_v3_main\Qwen__Qwen3-0.6B\TOKEN_LEAKAGE_AUDIT.md

.\.venv-jlens\Scripts\python.exe .\src\evaluate_probe_auc.py `
  --cases .\data\prompt_sets\heldout_templates_v3.jsonl `
  --config .\configs\prompt_set_v3_heldout_templates.yaml `
  --out-dir .\results\heldout_templates_v3_main `
  --model-id Qwen/Qwen3-0.6B `
  --dense-lens .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_lens_smoke.npz `
  --layers all `
  --jvp-layers stride:7 `
  --probe-label semantic_risk `
  --cross-risk-probe `
  --max-seq-len 192 `
  --max-new-tokens 48 `
  --probe-epochs 120 `
  --bootstrap-samples 500 `
  --dtype float16

.\.venv-jlens\Scripts\python.exe .\src\analyze_v3_falsification.py `
  --case-scores .\results\heldout_templates_v3_main\Qwen__Qwen3-0.6B\case_scores.jsonl `
  --out-dir .\results\heldout_templates_v3_analysis\Qwen__Qwen3-0.6B `
  --bootstrap-samples 1000
```
