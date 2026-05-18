param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Query,
    [string]$Device = "NPU",
    [int]$TopK = 8
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "codex_npu_context.py"

& $Python $Script --device $Device search $Query --top-k $TopK
