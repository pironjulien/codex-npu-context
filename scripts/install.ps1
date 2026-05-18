param(
    [string]$ModelRepo = "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
    [string]$Device = "NPU"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$ModelDir = Join-Path $Root "models\qwen3-embedding-0.6b-int8-ov"

if (!(Test-Path $Python)) {
    $SystemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (!$SystemPython) {
        $PyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
        if (!$PyLauncher) {
            throw "Python 3.11 is required. Install Python or add it to PATH."
        }
        & $PyLauncher -3.11 -m venv $Venv
    } else {
        & $SystemPython -m venv $Venv
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements.txt")

$DownloadScript = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$ModelRepo",
    local_dir=r"$ModelDir",
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

& $Python -c $DownloadScript
& $Python (Join-Path $Root "codex_npu_context.py") --device $Device status

Write-Host ""
Write-Host "Install complete."
Write-Host "Next: scripts\index-example.ps1 -Roots `$env:USERPROFILE\.codex\sessions"
