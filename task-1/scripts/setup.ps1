$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$env:PIP_CACHE_DIR = Join-Path $PWD ".cache/pip"
$env:HF_HOME = Join-Path $PWD ".cache/huggingface"
$env:PYTHONUTF8 = "1" # TrustMark's setup.py reads UTF-8 text using the process locale.

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path ".venv/Scripts/python.exe")) {
    $TaskPython = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $Candidate = & py -3.11 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0) { $TaskPython = $Candidate }
        } catch { $TaskPython = $null }
    }
    if ($TaskPython) {
        Invoke-Checked -Program $TaskPython -Arguments @("-m", "venv", ".venv")
    } else {
        # Bootstrap a project-local interpreter without changing system Python.
        $Uv = Join-Path $PWD ".cache/bootstrap/bin/uv.exe"
        if (-not (Test-Path $Uv)) {
            Invoke-Checked -Program "python" -Arguments @("-m", "pip", "install", "--target", ".cache/bootstrap", "uv")
        }
        Invoke-Checked -Program $Uv -Arguments @("python", "install", "3.11", "--install-dir", ".cache/python", "--no-bin")
        $TaskPython = Get-ChildItem ".cache/python/cpython-3.11*-windows-x86_64-none/python.exe" |
            Sort-Object FullName | Select-Object -Last 1 -ExpandProperty FullName
        if (-not $TaskPython) { throw "Project-local Python 3.11 installation not found" }
        Invoke-Checked -Program $Uv -Arguments @("venv", ".venv", "--python", $TaskPython, "--seed")
    }
}
$Python = Join-Path $PWD ".venv/Scripts/python.exe"
Invoke-Checked -Program $Python -Arguments @("-m", "pip", "install", "-r", "requirements-windows.txt", "-c", "requirements-windows.lock.txt", "-e", ".[dev]")
Invoke-Checked -Program $Python -Arguments @("scripts/download_resources.py")
Invoke-Checked -Program $Python -Arguments @("download_models.py", "--models_dir", "checkpoints")
Invoke-Checked -Program $Python -Arguments @("-m", "pytest")
Invoke-Checked -Program $Python -Arguments @("scripts/verify_setup.py")
Invoke-Checked -Program $Python -Arguments @("experiments/baseline/test_baseline.py")
Invoke-Checked -Program $Python -Arguments @("experiments/baseline/run.py", "--submission")
Write-Host "Setup complete. Use .venv\Scripts\python.exe as your Python interpreter."
