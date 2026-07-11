param(
    [string]$Python = ".\.venv-jlens\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$cases = ".\data\prompt_sets\trajectory_candidates_v4c_round1.jsonl"
$expectedCasesHash = "74f8d38d023c41fbeb3b790f39bbbf46f2e99d6814431bd576d7c13bbfcc114e"
$actualCasesHash = (Get-FileHash $cases -Algorithm SHA256).Hash.ToLower()
if ($actualCasesHash -ne $expectedCasesHash) {
    throw "v4c round-one dataset SHA-256 mismatch: $actualCasesHash"
}

$env:HF_HOME = (Resolve-Path ".\.cache\huggingface").Path
$commonHf = @(
    ".\src\run_trajectory_sampling.py",
    "--cases", $cases,
    "--model-id", "Qwen/Qwen3-4B",
    "--revision", "1cfa9a7208912126459214e8b04321603b3df60c",
    "--conditions", "trajectory_v4c_equal_authority_conflict",
    "--samples-per-prompt", "16",
    "--top-p", "0.95",
    "--max-new-tokens", "48",
    "--dtype", "float16"
)

& $Python @commonHf `
    --out-dir ".\results\trajectory_v4c_appendix_temperature_t1p2" `
    --seed-start 21000000 `
    --temperature 1.2
if ($LASTEXITCODE -ne 0) { throw "Qwen3-4B T=1.2 diagnostic failed" }

& $Python @commonHf `
    --out-dir ".\results\trajectory_v4c_appendix_temperature_t1p5" `
    --seed-start 22000000 `
    --temperature 1.5
if ($LASTEXITCODE -ne 0) { throw "Qwen3-4B T=1.5 diagnostic failed" }

$commonOllama = @(
    ".\src\run_ollama_trajectory_sampling.py",
    "--cases", $cases,
    "--conditions", "trajectory_v4c_equal_authority_conflict",
    "--samples-per-prompt", "16",
    "--temperature", "0.8",
    "--top-p", "0.95",
    "--top-k", "50",
    "--repeat-penalty", "1.0",
    "--presence-penalty", "0.0",
    "--frequency-penalty", "0.0",
    "--num-ctx", "512",
    "--max-new-tokens", "48"
)

& $Python @commonOllama `
    --model-id "gemma4:e2b" `
    --expected-digest "7fbdbf8f5e45a75bb122155ed546e765b4d9c53a1285f62fd9f506baa1c5a47e" `
    --expected-quantization "Q4_K_M" `
    --seed-start 31000000 `
    --out-dir ".\results\trajectory_v4c_appendix_gemma4_e2b"
if ($LASTEXITCODE -ne 0) { throw "Gemma 4 E2B diagnostic failed" }

& $Python @commonOllama `
    --model-id "qwen3.5:4b" `
    --expected-digest "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd" `
    --expected-quantization "Q4_K_M" `
    --seed-start 32000000 `
    --out-dir ".\results\trajectory_v4c_appendix_qwen35_4b"
if ($LASTEXITCODE -ne 0) { throw "Qwen3.5-4B diagnostic failed" }

& $Python ".\src\summarize_v4c_appendix_diagnostics.py"
if ($LASTEXITCODE -ne 0) { throw "v4c appendix summarization failed" }
