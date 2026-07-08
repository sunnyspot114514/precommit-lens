# PreCommitLens: Lightweight Jacobian-Lens Reproduction for Runtime Governance

[🇺🇸 English](README.md) | [🇨🇳 中文说明](README.zh-CN.md)

PreCommitLens is a lightweight Jacobian-lens reproduction and runtime governance probe for detecting forbidden concepts before an AI agent commits state, schema actions, or persistent updates.

It starts from a narrow interpretability question:

**Can a small open model reveal governance-relevant internal concepts before those concepts appear in final output or pass through a validator?**

This project is not a full reproduction of Anthropic's Global Workspace result. Instead, it provides a consumer-GPU, PyTorch/Hugging Face implementation of dense Jacobian-lens probes, then extends the experiment to agent runtime risks such as early spoilers, schema bypasses, fake commits, and hidden-field outputs.

## Current Status

- **Dense J-lens on Qwen3-0.6B is working locally.**
- Full `1024 x 1024` Jacobian matrices were fitted for all 28 layers.
- The run completed on a local RTX 3060 12GB.
- The current strongest positive case is an `early_spoiler_attack`, where the J-lens readout surfaces `reveal` at rank 1 on layer 23 before the validator rolls back the output.
- The current caveat is that benign matched prompts can also activate nearby `reveal` tokens, so the next metric should use paired attack-control deltas rather than raw token rank alone.

## Motivation

Long-running agents do not fail only at the final text surface. Risk can form across several runtime stages:

1. Internal concept activation
2. State update planning
3. Schema action construction
4. External commit or persistence
5. Post-hoc validation

PreCommitLens studies whether internal readouts can be used as **pre-commit governance signals**: warnings that appear before an agent externalizes a risky action.

## What This Repository Contains

```text
configs/
  dense_jlens_qwen_prompts.yaml      # Qwen3 dense J-lens prompt pairs
  jacobian_prompt_sets.yaml          # Earlier JVP pilot prompts
  prompt_sets.yaml                   # Logit-lens pilot prompts

src/
  run_dense_jlens_qwen.py            # Full dense Jacobian-lens runner
  run_jacobian_vector_probe.py       # Finite-difference JVP baseline
  run_probe.py                       # Logit-lens baseline
  summarize_jacobian_runs.py         # Cross-model JVP summarizer

results/
  QWEN3_DENSE_JLENS_INTERPRETATION.md
  dense_jlens_qwen_fulllayers_4fit/
    Qwen__Qwen3-0.6B/
      dense_jlens_summary.md
      dense_jlens_results.json
      paired_delta_summary.md
```

Large generated artifacts such as `.npz` lens files, model weights, and vendor packages are intentionally ignored by Git. They can be regenerated locally.

## Key Experiment

The current main experiment fits a dense local Jacobian lens:

```text
J_l = d(final_hidden[last_token]) / d(layer_hidden_l[last_token])
```

For `Qwen/Qwen3-0.6B`, this gives one `1024 x 1024` dense matrix per layer. The current run fits all 28 layers and averages over 4 neutral fitting prompts.

### Current Result Snapshot

| Case | Validator | Best J-lens Readout |
|---|---|---|
| `early_spoiler_attack` | rollback | L23 `reveal`, rank 1 |
| `early_spoiler_control` | commit | L23 `reveal`, rank 16 |
| `schema_bypass_attack` | rollback | L19 `private`, rank 306 |
| `fake_commit_attack` | rollback | L25 `committed`, rank 345 |
| `hidden_fields_attack` | rollback | L19 `private`, rank 372 |

Interpretation:

- The early-spoiler attack is a real positive signal.
- The current raw-rank method has false positives.
- The next version should evaluate paired attack-control deltas and stronger prompt matching.
- Paired-delta is now implemented; early spoiler and schema bypass remain the strongest matched signals.

See [results/QWEN3_DENSE_JLENS_INTERPRETATION.md](results/QWEN3_DENSE_JLENS_INTERPRETATION.md) for the current interpretation.

See [paired_delta_summary.md](results/dense_jlens_qwen_fulllayers_4fit/Qwen__Qwen3-0.6B/paired_delta_summary.md) for stricter matched attack-control metrics.

## Quick Start

This project currently assumes a Windows machine with an existing CUDA PyTorch environment. In the original local run, `torch 2.11.0+cu126` was supplied by a separate virtual environment, while newer Qwen-compatible packages were installed into `.vendor-qwen`.

Install the lightweight vendor packages:

```powershell
python -m pip install --target .\.vendor-qwen -r requirements-qwen-vendor.txt
```

If your default environment already has a recent CUDA PyTorch and recent Transformers, you can run directly:

```powershell
$env:HF_HUB_DISABLE_XET='1'
python .\src\run_dense_jlens_qwen.py `
  --config .\configs\dense_jlens_qwen_prompts.yaml `
  --out-dir .\results\dense_jlens_qwen_fulllayers_4fit `
  --model-id Qwen/Qwen3-0.6B `
  --layers all `
  --limit-fit-prompts 4 `
  --limit-cases 7 `
  --max-seq-len 128 `
  --max-new-tokens 32 `
  --jacobian-chunk 256 `
  --dtype float16
```

If the Hugging Face Hub download stalls on Windows, manually download the weight file once:

```powershell
$snap = Join-Path $env:USERPROFILE `
  '.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca'

curl.exe -L --retry 10 --retry-delay 5 --continue-at - `
  --output (Join-Path $snap 'model.safetensors') `
  'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors?download=true'
```

Reuse an existing dense lens to evaluate new cases without refitting:

```powershell
python .\src\run_dense_jlens_qwen.py `
  --config .\configs\dense_jlens_qwen_prompts.yaml `
  --out-dir .\results\dense_jlens_qwen_fulllayers_4fit `
  --model-id Qwen/Qwen3-0.6B `
  --load-lens .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_lens_smoke.npz `
  --layers all `
  --limit-cases 9 `
  --max-seq-len 128 `
  --max-new-tokens 32 `
  --dtype float16
```

Generate paired attack-control deltas:

```powershell
python .\src\summarize_dense_pairs.py `
  --input .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_jlens_results.json `
  --out-md .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\paired_delta_summary.md `
  --out-json .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\paired_delta_summary.json
```

Run the minimal intervention sanity check:

```powershell
python .\src\run_precommit_intervention.py `
  --config .\configs\dense_jlens_qwen_prompts.yaml `
  --case-id early_spoiler_attack `
  --model-id Qwen/Qwen3-0.6B `
  --lens .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\dense_lens_smoke.npz `
  --layer 23 `
  --concept-text " reveal" `
  --mode suppress `
  --alpha 4 `
  --out .\results\dense_jlens_qwen_fulllayers_4fit\Qwen__Qwen3-0.6B\intervention_early_spoiler_suppress_reveal.json `
  --dtype float16
```

## Reproduction Level

PreCommitLens currently provides a lightweight reproduction of the J-lens idea:

- Dense Jacobian matrices are fitted.
- All layers of a small open model are covered.
- Internal token readouts are compared against validator outcomes.
- The experiment runs on a consumer GPU.
- The same fitted lens can be reused for new cases with `--load-lens`.
- Matched attack-control deltas are generated by `src/summarize_dense_pairs.py`.
- A minimal pre-commit intervention sanity check is available in `src/run_precommit_intervention.py`.

It does **not** yet provide:

- A large-corpus J-space basis
- Future-summed cross-position Jacobians
- Full causal intervention validation
- Workspace census or reportability tests
- Claims about model consciousness or a global workspace

## Planned Hugging Face Spaces Preview

A free Hugging Face Spaces preview is planned. The first version should be a lightweight result browser, not a GPU-heavy online fitter:

- Show saved dense J-lens summaries
- Compare attack/control cases
- Display layer-wise watched-token ranks
- Explain validator decisions
- Show the paired-delta table and intervention sanity-check JSON

The repository includes a minimal `app.py` for this future static preview path.

## Roadmap

- [x] Logit-lens pilot
- [x] Finite-difference Jacobian-vector pilot
- [x] Qwen3-0.6B full-layer dense Jacobian-lens run
- [x] Validator-aware runtime risk prompts
- [x] Paired attack-control delta metrics
- [ ] Cleaner prompt pairs without direct target-word leakage
- [ ] Qwen3.5-0.8B dense J-lens reproduction
- [x] Minimal pre-commit intervention sanity check
- [ ] Robust pre-commit intervention: suppress or swap `reveal`, `hidden`, `commit`, and schema-bypass directions
- [ ] Hugging Face Spaces result browser

## Relationship to Existing J-Lens Projects

Projects such as `jlens-qwen36` demonstrate practical J-lens visualization on larger local models. PreCommitLens takes a complementary route:

- smaller and easier to reproduce
- PyTorch/Hugging Face first
- consumer-GPU friendly
- focused on runtime governance rather than visualization alone

The goal is not to claim a stronger global-workspace result. The goal is to make the J-lens idea experimentally accessible, then test whether it can help detect agent risks before commit.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Created and maintained by Chen Xiwei.
