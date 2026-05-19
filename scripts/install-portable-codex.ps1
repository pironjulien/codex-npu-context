param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [string]$RuntimeDir = "",
    [string]$Device = "NPU",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$SourceRoot = Split-Path -Parent $PSScriptRoot
if (!$RuntimeDir) {
    $RuntimeDir = Join-Path $CodexHome "mcp\codex-npu-context"
}

$RuntimeDir = [System.IO.Path]::GetFullPath($RuntimeDir)
$CodexHome = [System.IO.Path]::GetFullPath($CodexHome)

function Invoke-NativeChecked {
    & $args[0] @($args[1..($args.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

if ((Test-Path $RuntimeDir) -and $Force) {
    $Resolved = Resolve-Path $RuntimeDir
    if (!$Resolved.Path.StartsWith($CodexHome, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove runtime outside CodexHome: $($Resolved.Path)"
    }
    Remove-Item -LiteralPath $Resolved.Path -Recurse -Force
}

New-Item -ItemType Directory -Force $RuntimeDir | Out-Null

$Files = @(
    ".gitignore",
    "AGENTS.md",
    "codex_npu_context.py",
    "LICENSE",
    "package.json",
    "package-lock.json",
    "pyproject.toml",
    "README.md",
    "requirements.txt"
)

foreach ($File in $Files) {
    $Source = Join-Path $SourceRoot $File
    if (Test-Path $Source) {
        Copy-Item -LiteralPath $Source -Destination (Join-Path $RuntimeDir $File) -Force
    }
}

$Dirs = @("codex_npu_context_core", "examples", "mcp", "scripts", "skills")
foreach ($Dir in $Dirs) {
    $Source = Join-Path $SourceRoot $Dir
    $Destination = Join-Path $RuntimeDir $Dir
    if (Test-Path $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
}

$SourceModel = Join-Path $SourceRoot "models\qwen3-embedding-0.6b-int8-ov"
$RuntimeModel = Join-Path $RuntimeDir "models\qwen3-embedding-0.6b-int8-ov"
if (Test-Path $SourceModel) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $RuntimeModel) | Out-Null
    if (Test-Path $RuntimeModel) {
        Remove-Item -LiteralPath $RuntimeModel -Recurse -Force
    }
    Copy-Item -LiteralPath $SourceModel -Destination $RuntimeModel -Recurse -Force
}

Push-Location $RuntimeDir
try {
    $Npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (!$Npm) {
        $Npm = (Get-Command npm -ErrorAction Stop).Source
    }
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\install-python311.ps1")
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\install.ps1") -Device $Device
    Invoke-NativeChecked $Npm install
    Invoke-NativeChecked (Join-Path $RuntimeDir ".venv\Scripts\python.exe") -m pip install -e .
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\install-codex.ps1") `
        -CodexHome $CodexHome `
        -Device $Device `
        -Python (Join-Path $RuntimeDir ".venv\Scripts\python.exe") `
        -ModelDir $RuntimeModel `
        -IndexDir (Join-Path $RuntimeDir "index") `
        -OvCacheDir (Join-Path $RuntimeDir "ov_cache")
    $SafeReadme = Join-Path $RuntimeDir "README.md"
    $SafeSkill = Join-Path $RuntimeDir "skills\codex-npu-context"
    & (Join-Path $RuntimeDir "scripts\secret-scan.ps1") `
        -Roots @($SafeReadme, $SafeSkill) `
        -FailOnSecret
    if (!$?) {
        throw "Command failed: secret-scan safe roots"
    }
    & (Join-Path $RuntimeDir "scripts\index-example.ps1") `
        -Roots @($SafeReadme, $SafeSkill) `
        -MaxChunks 80 `
        -MaxChunksPerFile 40 `
        -FailOnSecret
    if (!$?) {
        throw "Command failed: index-example safe roots"
    }
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\verify-codex-install.ps1") `
        -CodexHome $CodexHome `
        -IndexDir (Join-Path $RuntimeDir "index")
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\mcp-smoke.ps1")
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\mcp-warm-benchmark.ps1") -Iterations 8
    Invoke-NativeChecked (Join-Path $RuntimeDir ".venv\Scripts\codex-npu-context.exe") doctor --fail --codex-home $CodexHome
    Invoke-NativeChecked powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RuntimeDir "scripts\benchmark.ps1") -Iterations 10
} finally {
    Pop-Location
}

Write-Host "Portable Codex install complete: $RuntimeDir"
