param(
  [string]$Python = ".\.venv-jlens\Scripts\python.exe",
  [string]$Config = ".\configs\dense_jlens_qwen_prompts.yaml",
  [string]$OutDir = ".\results\dense_jlens_multimodel_selected",
  [int]$Cases = 0,
  [int]$JacobianChunk = 128
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoPath = $repo.Path
Set-Location -LiteralPath $repoPath
$env:HF_HUB_CACHE = Join-Path $repo ".cache\huggingface\hub"
$env:TRANSFORMERS_CACHE = $env:HF_HUB_CACHE
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:HF_HUB_DISABLE_XET = "1"

$pythonPath = if ([System.IO.Path]::IsPathRooted($Python)) {
  $Python
} else {
  Join-Path $repoPath $Python
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
  throw "Missing isolated Python at $pythonPath. Create .venv-jlens before running."
}

$modelRuns = @(
  @{
    Model = "Qwen/Qwen3.5-0.8B"
    Layers = "0,6,12,18,23"
    FitPrompts = 1
  },
  @{
    Model = "unsloth/gemma-3-270m-it"
    Layers = "all"
    FitPrompts = 4
  }
)

foreach ($run in $modelRuns) {
  $model = $run.Model
  $layers = $run.Layers
  $fitPrompts = $run.FitPrompts
  Write-Host "=== Running dense J-lens: $model ==="
  & $pythonPath .\src\run_dense_jlens_qwen.py `
    --config $Config `
    --out-dir $OutDir `
    --model-id $model `
    --allow-download `
    --layers $layers `
    --limit-fit-prompts $fitPrompts `
    --limit-cases $Cases `
    --jacobian-chunk $JacobianChunk `
    --max-seq-len 160 `
    --max-new-tokens 48 `
    --top-k 10

  $safe = $model -replace "[^A-Za-z0-9_.-]+", "__"
  & $pythonPath .\src\summarize_dense_pairs.py `
    --input (Join-Path $OutDir "$safe\dense_jlens_results.json") `
    --out-md (Join-Path $OutDir "$safe\paired_delta_summary.md") `
    --out-json (Join-Path $OutDir "$safe\paired_delta_summary.json")
}
