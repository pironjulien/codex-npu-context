param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" })
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ExpectedMcp = Join-Path $Root "mcp\index.js"
$ExpectedSkill = Join-Path $Root "skills\codex-npu-context\SKILL.md"
$ActiveSkill = Join-Path $CodexHome "skills\codex-npu-context\SKILL.md"
$ConfigPath = Join-Path $CodexHome "config.toml"

function ConvertTo-TomlString([string]$Value) {
    return $Value.Replace("\", "\\").Replace('"', '\"')
}

$ConfigText = if (Test-Path $ConfigPath) { Get-Content -Raw -Path $ConfigPath } else { "" }
$ExpectedMcpToml = ConvertTo-TomlString $ExpectedMcp
$ConfigPointsToRepoMcp = (
    $ConfigText.Contains("[mcp_servers.codex-npu-context]") -and
    $ConfigText.Contains($ExpectedMcpToml)
)

$RepoSkillHash = if (Test-Path $ExpectedSkill) { (Get-FileHash -Algorithm SHA256 $ExpectedSkill).Hash } else { $null }
$ActiveSkillHash = if (Test-Path $ActiveSkill) { (Get-FileHash -Algorithm SHA256 $ActiveSkill).Hash } else { $null }
$SkillMatchesRepo = ($RepoSkillHash -ne $null -and $RepoSkillHash -eq $ActiveSkillHash)

$Payload = [ordered]@{
    ok = ($ConfigPointsToRepoMcp -and $SkillMatchesRepo)
    codex_home = $CodexHome
    config_path = $ConfigPath
    expected_mcp = $ExpectedMcp
    config_points_to_repo_mcp = $ConfigPointsToRepoMcp
    expected_skill = $ExpectedSkill
    active_skill = $ActiveSkill
    skill_matches_repo = $SkillMatchesRepo
    repo_skill_sha256 = $RepoSkillHash
    active_skill_sha256 = $ActiveSkillHash
}

$Payload | ConvertTo-Json -Depth 4

if (!$Payload.ok) {
    exit 1
}
