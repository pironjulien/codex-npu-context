param(
    [Parameter(Mandatory = $true)]
    [string[]]$Roots,
    [string]$Device = "NPU",
    [int]$MaxChunks = 500,
    [int]$MaxChunksPerFile = 120,
    [int]$BatchSize = 8,
    [int]$Parallelism = 1,
    [switch]$FailOnSecret,
    [switch]$NoIncremental
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "--device", $Device, "index", "--roots") + $Roots + @(
    "--max-chunks", $MaxChunks,
    "--max-chunks-per-file", $MaxChunksPerFile,
    "--batch-size", $BatchSize,
    "--parallelism", $Parallelism
)
if ($FailOnSecret) {
    $Args += "--fail-on-secret"
}
if ($NoIncremental) {
    $Args += "--no-incremental"
}
& $Python @Args
