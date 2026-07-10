---
title: PreCommitLens
sdk: static
pinned: false
license: mit
models:
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-4B
tags:
- interpretability
- jacobian-lens
- runtime-governance
- ai-safety
short_description: Static PreCommitLens result browser.
---

# PreCommitLens

Static result browser for the PreCommitLens experiment.

This Space does not download models, fit Jacobians, or run live inference. It
serves precomputed dense Jacobian-lens summaries, paired attack/control deltas,
and the pre-registered v4/v4b fixed-prompt trajectory results.

The v4 confirmatory run contains 1,088 fresh trajectories over 34 prompts. Its
primary residual-added-value gate failed: internal residual prediction became
accurate before policy landing, but a visible-prefix TF-IDF baseline matched it.

The frozen-prompt Qwen3-4B replication is inconclusive for probe added value:
only 2/34 prompts remain outcome-divergent, including 1/9 test prompts. This
shows that the 0.6B contrast-selected benchmark does not directly transfer as a
4B scale point.

For the full code and reproduction scripts, see the GitHub repository:

<https://github.com/sunnyspot114514/precommit-lens>
