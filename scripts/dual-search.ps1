param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [string[]]$Roots = @(),
    [string]$Rg = "",
    [string]$Device = "NPU",
    [int]$TopK = 8,
    [double]$MinScore = -999
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "--device", $Device, "dual-search", $Query, "--top-k", $TopK)
if ($Roots.Count -gt 0) {
    $Args += @("--roots") + $Roots
}
if ($Rg) {
    $Args += @("--rg", $Rg)
}
if ($MinScore -ne -999) {
    $Args += @("--min-score", $MinScore)
}

& $Python @Args
