param(
    [switch]$Strict
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } elseif (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { "python" }
$Script = Join-Path $Root "codex_npu_context.py"

$DoctorArgs = @($Script, "doctor")
if ($Strict) {
    $DoctorArgs += "--fail"
}

& $Python @DoctorArgs
$AstCheck = @"
import ast
from pathlib import Path
files = [Path("codex_npu_context.py"), Path("tests/test_core.py"), *Path("codex_npu_context_core").glob("*.py")]
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("AST parse complete.")
"@
& $Python -c $AstCheck
& $Python -m unittest discover -s (Join-Path $Root "tests")
& $Python $Script "secret-scan" "--roots" (Join-Path $Root "README.md") (Join-Path $Root "AGENTS.md") (Join-Path $Root "skills\codex-npu-context") "--fail-on-secret"
node --check (Join-Path $Root "mcp\index.js") | Out-Null

Write-Host "Self-test complete."
