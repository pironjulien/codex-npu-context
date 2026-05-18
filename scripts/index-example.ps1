param(
    [Parameter(Mandatory = $true)]
    [string[]]$Roots,
    [string]$Device = "NPU",
    [int]$MaxChunks = 500
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Script = Join-Path $Root "codex_npu_context.py"

$Args = @($Script, "--device", $Device, "index", "--roots") + $Roots + @("--max-chunks", $MaxChunks)
& $Python @Args
