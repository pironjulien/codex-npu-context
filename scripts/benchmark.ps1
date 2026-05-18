param(
    [string[]]$Devices = @("NPU", "CPU"),
    [int]$Iterations = 20,
    [int]$Warmup = 2,
    [double]$SustainSeconds = 0,
    [int]$TopK = 3,
    [int[]]$BatchSizes = @(1, 4, 8, 16),
    [string[]]$Queries = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @(
    $Script,
    "bench",
    "--devices"
) + $Devices + @(
    "--iterations", $Iterations,
    "--warmup", $Warmup,
    "--sustain-seconds", $SustainSeconds,
    "--top-k", $TopK,
    "--batch-sizes"
) + $BatchSizes

if ($Queries.Count -gt 0) {
    $Args += "--queries"
    $Args += $Queries
}

& $Python @Args
