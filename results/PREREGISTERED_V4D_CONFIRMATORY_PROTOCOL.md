# Pre-Registered v4d Confirmatory Protocol

This concrete protocol was generated after the frozen stage-one gate passed and before any
confirmatory trajectory was sampled.

## Frozen Population

- Model: `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, unquantized FP16 Transformers.
- Prompt file: `data/prompt_sets/trajectory_confirmatory_v4d.jsonl`.
- Prompt SHA-256: `f2de5815ef24fe70bac4793500592f36a467937dad26e280f092d62ec51acd24`.
- Prompts / families: `33` / `22`.
- Train / validation / test: `17 / 8 / 8`.
- Included risks: `early_spoiler, fake_commit, hidden_fields`.
- Stage-one trajectories are excluded from fitting, selection, and metrics.

## Confirmatory Sampling and Features

- `32` fresh trajectories per prompt; seed start `34,000,000`.
- `temperature=0.8`, `top_p=0.95`, `max_new_tokens=48`, thinking disabled.
- Checkpoints: `0, 2, 4, 6, 8, 10, 12, 16, 24`.
- Captured layers: `0, 7, 14, 21, 28, 31`; frozen primary layer: `21`.
- Fresh-seed collapse remains in every denominator; prompts are never replaced.

## Metric and Gate

The residual probe, validation-selected word/character TF-IDF, next-token statistics,
same-model prefix judge, balanced prompt/class weights, prompt-macro pairwise AUC, and 2,000
prompt-cluster bootstrap replicates are unchanged from v4.

The primary result is positive only if all conditions hold:

1. At least `6/8` test prompts across at least two risks retain fresh-seed contrast.
2. A winning checkpoint has at least six evaluable test prompts across at least two risks.
3. Layer-21 residual AUC exceeds the cheap TF-IDF/next-token envelope by at least `0.03`, with
   the paired 95% interval entirely above zero.
4. Condition 3 holds at two consecutive primary checkpoints (`2, 4, 6, 8, 10, 12`).

Failure of condition 1 is `INCONCLUSIVE`; otherwise failure of conditions 2-4 is `FAIL`.
Per-risk, secondary-layer, and judge results cannot rescue the gate. v4d is the final
pre-submission experiment and cannot derive a v4e.
