param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [string]$Device = "NPU",
    [int]$MaxChunks = 1200,
    [int]$MaxChunksPerFile = 120,
    [int]$BatchSize = 8,
    [int]$Parallelism = 8,
    [string[]]$ExtraRoots = @(),
    [switch]$IncludeSessions,
    [switch]$SkipRuntimeDocs,
    [switch]$AllowSecretFindings,
    [switch]$NoIncremental,
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$Script = Join-Path $Root "codex_npu_context.py"
$CodexHome = [System.IO.Path]::GetFullPath($CodexHome)

$Roots = New-Object System.Collections.Generic.List[string]
$RuntimeReadme = Join-Path $Root "README.md"
$RuntimeSkill = Join-Path $Root "skills\codex-npu-context"
if (!$SkipRuntimeDocs) {
    if (Test-Path $RuntimeReadme) {
        $Roots.Add([System.IO.Path]::GetFullPath($RuntimeReadme))
    }
    if (Test-Path $RuntimeSkill) {
        $Roots.Add([System.IO.Path]::GetFullPath($RuntimeSkill))
    }
}

$Memories = Join-Path $CodexHome "memories"
if (Test-Path $Memories) {
    $Roots.Add([System.IO.Path]::GetFullPath($Memories))
}

if ($IncludeSessions) {
    $Sessions = Join-Path $CodexHome "sessions"
    if (Test-Path $Sessions) {
        $Roots.Add([System.IO.Path]::GetFullPath($Sessions))
    }
}

foreach ($ExtraRoot in $ExtraRoots) {
    if (Test-Path $ExtraRoot) {
        $Roots.Add([System.IO.Path]::GetFullPath($ExtraRoot))
    } else {
        throw "Extra root does not exist: $ExtraRoot"
    }
}

if ($Roots.Count -eq 0) {
    throw "No Codex memory roots found under $CodexHome. Expected at least: $Memories"
}

$UniqueRoots = @($Roots | Select-Object -Unique)
$Plan = [ordered]@{
    ok = $true
    codex_home = $CodexHome
    roots = $UniqueRoots
    include_sessions = [bool]$IncludeSessions
    include_runtime_docs = -not [bool]$SkipRuntimeDocs
    extra_roots = $ExtraRoots
    fail_on_secret = -not [bool]$AllowSecretFindings
    max_chunks = $MaxChunks
    max_chunks_per_file = $MaxChunksPerFile
    batch_size = $BatchSize
    parallelism = $Parallelism
    incremental = -not [bool]$NoIncremental
}

if ($WhatIfOnly) {
    $Plan | ConvertTo-Json -Depth 8
    exit 0
}

if (!(Test-Path $Python)) {
    throw "Python runtime not found: $Python. Run .\scripts\install.ps1 first."
}

$ScanArgs = @($Script, "secret-scan", "--roots") + $UniqueRoots
if (!$AllowSecretFindings) {
    $ScanArgs += "--fail-on-secret"
}
& $Python @ScanArgs
if ($LASTEXITCODE -ne 0) {
    throw "Secret scan failed. Review findings or rerun with -AllowSecretFindings only after explicit approval."
}

$IndexArgs = @(
    $Script,
    "--device", $Device,
    "index",
    "--roots"
) + $UniqueRoots + @(
    "--max-chunks", $MaxChunks,
    "--max-chunks-per-file", $MaxChunksPerFile,
    "--batch-size", $BatchSize,
    "--parallelism", $Parallelism
)

if (!$AllowSecretFindings) {
    $IndexArgs += "--fail-on-secret"
}
if ($NoIncremental) {
    $IndexArgs += "--no-incremental"
}

& $Python @IndexArgs
if ($LASTEXITCODE -ne 0) {
    throw "Codex memory indexing failed."
}
