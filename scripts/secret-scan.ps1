param(
    [Parameter(Mandatory = $true)]
    [string[]]$Roots,
    [int]$LimitMb = 12,
    [switch]$FailOnSecret
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "secret-scan", "--roots") + $Roots + @("--limit-mb", $LimitMb)
if ($FailOnSecret) {
    $Args += "--fail-on-secret"
}

& $Python @Args
