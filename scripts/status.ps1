param(
    [string]$Device = "NPU",
    [switch]$DeviceNames
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "--device", $Device, "status")
if ($DeviceNames) {
    $Args += "--device-names"
}

& $Python @Args
