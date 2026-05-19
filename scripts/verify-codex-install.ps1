param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [string]$IndexDir = $(if ($env:CODEX_NPU_CONTEXT_INDEX_DIR) { $env:CODEX_NPU_CONTEXT_INDEX_DIR } else { Join-Path (Split-Path -Parent $PSScriptRoot) "index" }),
    [switch]$AllowNoPreload
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
$PreloadEnabled = $ConfigText.Contains('CODEX_NPU_CONTEXT_PRELOAD = "1"')

$RepoSkillHash = if (Test-Path $ExpectedSkill) { (Get-FileHash -Algorithm SHA256 $ExpectedSkill).Hash } else { $null }
$ActiveSkillHash = if (Test-Path $ActiveSkill) { (Get-FileHash -Algorithm SHA256 $ActiveSkill).Hash } else { $null }
$SkillMatchesRepo = ($RepoSkillHash -ne $null -and $RepoSkillHash -eq $ActiveSkillHash)
$IndexExists = (Test-Path (Join-Path $IndexDir "chunks.jsonl")) -and (Test-Path (Join-Path $IndexDir "embeddings.npy"))
$ManifestPath = Join-Path $IndexDir "manifest.json"
$ManifestExists = Test-Path $ManifestPath
$ChunksPath = Join-Path $IndexDir "chunks.jsonl"
$ChunksCount = if (Test-Path $ChunksPath) {
    [System.Linq.Enumerable]::Count([System.IO.File]::ReadLines($ChunksPath))
} else {
    0
}
$IndexHasChunks = $ChunksCount -gt 0

$Payload = [ordered]@{
    ok = ($ConfigPointsToRepoMcp -and $SkillMatchesRepo -and ($PreloadEnabled -or $AllowNoPreload) -and $IndexExists -and $ManifestExists -and $IndexHasChunks)
    codex_home = $CodexHome
    config_path = $ConfigPath
    expected_mcp = $ExpectedMcp
    config_points_to_repo_mcp = $ConfigPointsToRepoMcp
    preload_enabled = $PreloadEnabled
    preload_required = !$AllowNoPreload
    expected_skill = $ExpectedSkill
    active_skill = $ActiveSkill
    skill_matches_repo = $SkillMatchesRepo
    repo_skill_sha256 = $RepoSkillHash
    active_skill_sha256 = $ActiveSkillHash
    index_dir = $IndexDir
    index_exists = $IndexExists
    chunks_count = $ChunksCount
    index_has_chunks = $IndexHasChunks
    manifest_path = $ManifestPath
    manifest_exists = $ManifestExists
}

$Payload | ConvertTo-Json -Depth 4

if (!$Payload.ok) {
    exit 1
}
