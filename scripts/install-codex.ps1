param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [string]$Device = "NPU",
    [string]$Python = "",
    [string]$ModelDir = "",
    [string]$IndexDir = "",
    [string]$OvCacheDir = "",
    [switch]$NoSkill,
    [switch]$NoPreload
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$McpEntry = Join-Path $Root "mcp\index.js"
$SkillSource = Join-Path $Root "skills\codex-npu-context"
$ConfigPath = Join-Path $CodexHome "config.toml"

function ConvertTo-TomlString([string]$Value) {
    return $Value.Replace("\", "\\").Replace('"', '\"')
}

if (!(Test-Path $McpEntry)) {
    throw "MCP entrypoint not found: $McpEntry"
}

if (!$NoSkill) {
    $SkillFile = Join-Path $SkillSource "SKILL.md"
    if (!(Test-Path $SkillFile)) {
        throw "Skill source not found: $SkillFile"
    }
    $SkillsDir = Join-Path $CodexHome "skills"
    New-Item -ItemType Directory -Force $SkillsDir | Out-Null
    Copy-Item -LiteralPath $SkillSource -Destination $SkillsDir -Recurse -Force
}

New-Item -ItemType Directory -Force $CodexHome | Out-Null
$ConfigText = if (Test-Path $ConfigPath) { Get-Content -Raw -Path $ConfigPath } else { "" }

$ConfigText = [regex]::Replace(
    $ConfigText,
    "(?ms)^\[mcp_servers\.codex-npu-context\.env\]\r?\n.*?(?=^\[|\z)",
    ""
)
$ConfigText = [regex]::Replace(
    $ConfigText,
    "(?ms)^\[mcp_servers\.codex-npu-context\]\r?\n.*?(?=^\[|\z)",
    ""
).TrimEnd()

$Block = @(
    "",
    "[mcp_servers.codex-npu-context]",
    'command = "node"',
    ('args = [ "{0}" ]' -f (ConvertTo-TomlString $McpEntry)),
    "startup_timeout_sec = 30",
    "tool_timeout_sec = 300",
    "enabled = true",
    "",
    "[mcp_servers.codex-npu-context.env]",
    ('CODEX_NPU_CONTEXT_DEVICE = "{0}"' -f (ConvertTo-TomlString $Device))
)

if (!$NoPreload) {
    $Block += 'CODEX_NPU_CONTEXT_PRELOAD = "1"'
}

if ($Python) {
    $Block += ('CODEX_NPU_CONTEXT_PYTHON = "{0}"' -f (ConvertTo-TomlString $Python))
}
if ($ModelDir) {
    $Block += ('CODEX_NPU_CONTEXT_MODEL_DIR = "{0}"' -f (ConvertTo-TomlString $ModelDir))
}
if ($IndexDir) {
    $Block += ('CODEX_NPU_CONTEXT_INDEX_DIR = "{0}"' -f (ConvertTo-TomlString $IndexDir))
}
if ($OvCacheDir) {
    $Block += ('CODEX_NPU_CONTEXT_OV_CACHE_DIR = "{0}"' -f (ConvertTo-TomlString $OvCacheDir))
}

$NewConfig = ($ConfigText + ($Block -join [Environment]::NewLine) + [Environment]::NewLine)
Set-Content -Path $ConfigPath -Value $NewConfig -Encoding utf8

Write-Host "Configured Codex MCP: $McpEntry"
if (!$NoSkill) {
    Write-Host "Installed Codex skill: $(Join-Path $CodexHome "skills\codex-npu-context")"
}
if (!$NoPreload) {
    Write-Host "Enabled MCP preload: CODEX_NPU_CONTEXT_PRELOAD=1"
}
Write-Host "Restart Codex to reload MCP servers and skills."
