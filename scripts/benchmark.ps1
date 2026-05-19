param(
    [string]$Device = "NPU",
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
    "--device", $Device,
    "bench",
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

$Output = & $Python @Args
$Output | Write-Output

try {
    $Payload = $Output | ConvertFrom-Json
} catch {
    throw "Benchmark did not return valid JSON."
}

if (!$Payload.ok) {
    throw "Benchmark payload reported ok=false."
}

$Runs = @()
foreach ($DevicePayload in @($Payload.devices)) {
    $Runs += @($DevicePayload.batch_runs)
}

if ($Runs.Count -eq 0) {
    throw "Benchmark returned no batch runs."
}

$FailedRuns = @($Runs | Where-Object { $_.error })
$SuccessfulRuns = @($Runs | Where-Object { !$_.error -and $_.queries_per_second })

if ($FailedRuns.Count -gt 0 -and $SuccessfulRuns.Count -eq 0) {
    throw "Benchmark failed: all batch runs returned errors."
}
