param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $Root "skills\codex-npu-context"
$SkillFile = Join-Path $Source "SKILL.md"

if (!(Test-Path $SkillFile)) {
    throw "Skill source not found: $SkillFile"
}

$SkillsDir = Join-Path $CodexHome "skills"
$Destination = Join-Path $SkillsDir "codex-npu-context"

if ((Test-Path $Destination) -and !$Force) {
    throw "Skill already exists at $Destination. Re-run with -Force to overwrite files."
}

New-Item -ItemType Directory -Force $SkillsDir | Out-Null
Copy-Item -LiteralPath $Source -Destination $SkillsDir -Recurse -Force

Write-Host "Installed codex-npu-context skill to $Destination"
Write-Host "Restart Codex to pick up new skills."
