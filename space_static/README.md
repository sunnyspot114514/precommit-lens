---
title: PreCommitLens
sdk: static
pinned: false
license: mit
models:
- Qwen/Qwen3-0.6B
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
and the pre-registered v4 fixed-prompt trajectory result.

The v4 confirmatory run contains 1,088 fresh trajectories over 34 prompts. Its
primary residual-added-value gate failed: internal residual prediction became
accurate before policy landing, but a visible-prefix TF-IDF baseline matched it.

For the full code and reproduction scripts, see the GitHub repository:

<https://github.com/sunnyspot114514/precommit-lens>
