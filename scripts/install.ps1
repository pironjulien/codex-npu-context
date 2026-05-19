param(
    [string]$ModelRepo = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
    [string]$Device = "NPU"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$ModelDir = Join-Path $Root "models\qwen3-embedding-0.6b-int8-ov"

function Invoke-NativeChecked {
    & $args[0] @($args[1..($args.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($args -join ' ')"
    }
}

function Get-PythonMajorMinor([string]$PythonCommand) {
    try {
        $Version = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        return ($Version -join "").Trim()
    } catch {
        return $null
    }
}

function Resolve-Python311 {
    $PyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($PyLauncher) {
        & $PyLauncher -3.11 -c "import sys; print(sys.executable)" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return $PyLauncher
        }
    }

    $SystemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($SystemPython -and (Get-PythonMajorMinor $SystemPython) -eq "3.11") {
        return $SystemPython
    }

    throw "Python 3.11 is required. Install Python 3.11 or make it available through the py launcher as py -3.11."
}

if (!(Test-Path $Python)) {
    $Python311 = Resolve-Python311
    if ((Split-Path -Leaf $Python311) -ieq "py.exe") {
        Invoke-NativeChecked $Python311 -3.11 -m venv $Venv
    } else {
        Invoke-NativeChecked $Python311 -m venv $Venv
    }
}

if ((Get-PythonMajorMinor $Python) -ne "3.11") {
    throw "The virtual environment must use Python 3.11. Delete .venv and re-run scripts\install.ps1 after installing Python 3.11."
}

Invoke-NativeChecked $Python -m pip install --upgrade pip
Invoke-NativeChecked $Python -m pip install -r (Join-Path $Root "requirements.txt")

$RequiredModelFiles = @(
    "openvino_model.xml",
    "openvino_model.bin",
    "tokenizer.json"
)
$ModelPresent = $true
foreach ($File in $RequiredModelFiles) {
    if (!(Test-Path (Join-Path $ModelDir $File))) {
        $ModelPresent = $false
        break
    }
}

if (!$ModelPresent) {
    $DownloadScript = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=$($ModelRepo | ConvertTo-Json),
    local_dir=$($ModelDir | ConvertTo-Json),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "openvino_model.xml",
        "openvino_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "openvino_config.json",
        "README.md",
    ],
)
"@
    $DownloadScriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ("codex-npu-context-download-{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        Set-Content -Path $DownloadScriptPath -Value $DownloadScript -Encoding utf8
        Invoke-NativeChecked $Python $DownloadScriptPath
    } finally {
        if (Test-Path $DownloadScriptPath) {
            Remove-Item -LiteralPath $DownloadScriptPath -Force
        }
    }
} else {
    Write-Host "Model already present: $ModelDir"
}
Invoke-NativeChecked $Python (Join-Path $Root "codex_npu_context.py") --device $Device status

Write-Host ""
Write-Host "Install complete."
Write-Host "Next: scripts\index-example.ps1 -Roots `$env:USERPROFILE\.codex\sessions"
