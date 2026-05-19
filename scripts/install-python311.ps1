param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Test-Python311 {
    $PyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($PyLauncher) {
        & $PyLauncher -3.11 -c "import sys; print(sys.version)" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
    }

    $SystemPython = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($SystemPython) {
        $Version = & $SystemPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if (($Version -join "").Trim() -eq "3.11") {
            return $true
        }
    }
    return $false
}

if ((Test-Python311) -and !$Force) {
    Write-Host "Python 3.11 is already available."
    exit 0
}

$Winget = (Get-Command winget -ErrorAction SilentlyContinue).Source
if (!$Winget) {
    throw "Python 3.11 is required and winget is not available. Install Python 3.11 manually, then re-run scripts\install.ps1."
}

& $Winget install `
    --id Python.Python.3.11 `
    --exact `
    --scope user `
    --silent `
    --accept-package-agreements `
    --accept-source-agreements

if (!(Test-Python311)) {
    throw "Python 3.11 installation did not become visible through py -3.11 or python. Restart the shell and re-run scripts\doctor.ps1."
}

Write-Host "Python 3.11 is available."
