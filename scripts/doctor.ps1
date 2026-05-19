param(
    [string]$CodexHome = $(if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }),
    [string]$Device = "NPU"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:CODEX_NPU_CONTEXT_PYTHON) { $env:CODEX_NPU_CONTEXT_PYTHON } else { Join-Path $Root ".venv\Scripts\python.exe" }
$SystemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
$PyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
$Node = (Get-Command node -ErrorAction SilentlyContinue).Source
$Npm = (Get-Command npm -ErrorAction SilentlyContinue).Source
$Rg = (Get-Command rg -ErrorAction SilentlyContinue).Source
$Script = Join-Path $Root "codex_npu_context.py"
$ModelDir = if ($env:CODEX_NPU_CONTEXT_MODEL_DIR) { $env:CODEX_NPU_CONTEXT_MODEL_DIR } else { Join-Path $Root "models\qwen3-embedding-0.6b-int8-ov" }
$IndexDir = if ($env:CODEX_NPU_CONTEXT_INDEX_DIR) { $env:CODEX_NPU_CONTEXT_INDEX_DIR } else { Join-Path $Root "index" }
$ConfigPath = Join-Path $CodexHome "config.toml"
$SkillPath = Join-Path $CodexHome "skills\codex-npu-context\SKILL.md"

function Test-CommandVersion([string]$Command, [string[]]$Arguments) {
    try {
        $Output = & $Command @Arguments 2>$null
        return ($Output -join "`n").Trim()
    } catch {
        return $null
    }
}

function Get-PythonMajorMinor([string]$Command, [string[]]$Arguments = @()) {
    try {
        $Output = & $Command @Arguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($Output -join "").Trim()
    } catch {
        return $null
    }
}

$PythonForImport = $null
$PythonForImportArgs = @()
if (Test-Path $Python) {
    $PythonForImport = $Python
} elseif ($PyLauncher) {
    & $PyLauncher -3.11 -c "import sys; print(sys.executable)" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $PythonForImport = $PyLauncher
        $PythonForImportArgs = @("-3.11")
    }
} elseif ($SystemPython -and (Get-PythonMajorMinor $SystemPython) -eq "3.11") {
    $PythonForImport = $SystemPython
}

$PythonVersion = if ($PythonForImport) { Test-CommandVersion $PythonForImport ($PythonForImportArgs + @("--version")) } else { $null }
$PythonMajorMinor = if ($PythonForImport) { Get-PythonMajorMinor $PythonForImport $PythonForImportArgs } else { $null }
$Python311Available = ($PythonMajorMinor -eq "3.11")
$NodeVersion = if ($Node) { Test-CommandVersion $Node @("--version") } else { $null }
$NpmVersion = if ($Npm) { Test-CommandVersion $Npm @("--version") } else { $null }
$RgVersion = $null
if ($Rg) {
    $RgVersionOutput = Test-CommandVersion $Rg @("--version")
    if ($RgVersionOutput) {
        $RgVersion = ($RgVersionOutput -split "`n")[0]
    }
}

$Imports = [ordered]@{
    numpy = $false
    openvino = $false
    transformers = $false
    huggingface_hub = $false
}

if ($PythonForImport) {
    foreach ($Name in @($Imports.Keys)) {
        $Code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Name') else 1)"
        & $PythonForImport @PythonForImportArgs -c $Code 2>$null
        $Imports[$Name] = ($LASTEXITCODE -eq 0)
    }
}

$ModelFiles = @("openvino_model.xml", "openvino_model.bin", "tokenizer.json")
$MissingModelFiles = @($ModelFiles | Where-Object { !(Test-Path (Join-Path $ModelDir $_)) })
$IndexExists = (Test-Path (Join-Path $IndexDir "chunks.jsonl")) -and (Test-Path (Join-Path $IndexDir "embeddings.npy"))
$ManifestExists = Test-Path (Join-Path $IndexDir "manifest.json")
$ConfigText = if (Test-Path $ConfigPath) { Get-Content -Raw -Path $ConfigPath } else { "" }

$Status = $null
$StatusError = $null
$ReadyForOpenVinoStatus = (Test-Path $Python) -and $Python311Available -and ($MissingModelFiles.Count -eq 0) -and $Imports.openvino -and $Imports.transformers -and $Imports.numpy
if ($ReadyForOpenVinoStatus) {
    try {
        $StatusOutput = & $Python $Script --device $Device status --device-names
        $Status = $StatusOutput | ConvertFrom-Json
    } catch {
        $StatusError = $_.Exception.Message
    }
}

$NpuAvailable = ($Status -ne $null -and $Status.npu_available)
$Payload = [ordered]@{
    ok = ($Python311Available -and $Node -and $Npm -and $Rg -and (!$ReadyForOpenVinoStatus -or $NpuAvailable))
    root = $Root
    codex_home = $CodexHome
    python = [ordered]@{
        repo_venv = $Python
        repo_venv_exists = Test-Path $Python
        system_python = $SystemPython
        py_launcher = $PyLauncher
        active_for_checks = $PythonForImport
        active_args = $PythonForImportArgs
        version = $PythonVersion
        major_minor = $PythonMajorMinor
        python311_available = $Python311Available
        imports = $Imports
    }
    node = [ordered]@{
        path = $Node
        version = $NodeVersion
        npm = $Npm
        npm_version = $NpmVersion
    }
    rg = [ordered]@{
        path = $Rg
        version = $RgVersion
    }
    model = [ordered]@{
        dir = $ModelDir
        exists = ($MissingModelFiles.Count -eq 0)
        missing = $MissingModelFiles
    }
    index = [ordered]@{
        dir = $IndexDir
        exists = $IndexExists
        manifest_exists = $ManifestExists
    }
    codex = [ordered]@{
        config_path = $ConfigPath
        config_exists = Test-Path $ConfigPath
        mcp_configured = $ConfigText.Contains("[mcp_servers.codex-npu-context]")
        preload_enabled = $ConfigText.Contains('CODEX_NPU_CONTEXT_PRELOAD = "1"')
        skill_path = $SkillPath
        skill_exists = Test-Path $SkillPath
    }
    openvino_status = $Status
    openvino_status_error = $StatusError
    openvino_status_checked = $ReadyForOpenVinoStatus
    npu_available = $NpuAvailable
}

$Payload | ConvertTo-Json -Depth 8

if (!$Payload.ok) {
    exit 1
}
