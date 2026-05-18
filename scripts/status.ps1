param(
    [string]$Device = "NPU",
    [switch]$DeviceNames
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "--device", $Device, "status")
if ($DeviceNames) {
    $Args += "--device-names"
}

& $Python @Args
