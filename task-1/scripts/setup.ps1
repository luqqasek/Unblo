$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PIP_CACHE_DIR = Join-Path $PWD ".cache/pip"
$env:HF_HOME = Join-Path $PWD ".cache/huggingface"

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path ".venv/Scripts/python.exe")) {
    Invoke-Checked -Program "py" -Arguments @("-3.11", "-m", "venv", ".venv")
}
$Python = Join-Path $PWD ".venv/Scripts/python.exe"
Invoke-Checked -Program $Python -Arguments @("-m", "pip", "install", "-r", "requirements-windows.txt", "-e", ".[dev]")
Invoke-Checked -Program $Python -Arguments @("scripts/download_resources.py")
Invoke-Checked -Program $Python -Arguments @("download_models.py", "--models_dir", "checkpoints")
Invoke-Checked -Program $Python -Arguments @("-m", "pytest")
Invoke-Checked -Program $Python -Arguments @("scripts/verify_setup.py")
Write-Host "Setup complete. Use .venv\Scripts\python.exe as your Python interpreter."
