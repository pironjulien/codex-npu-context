param(
    [string]$Device = "NPU"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "codex_npu_context.py"

& $Python $Script --device $Device status
