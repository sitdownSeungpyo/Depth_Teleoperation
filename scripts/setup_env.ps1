# Bootstrap a Python 3.11 virtualenv and install the project in editable mode.
# Run from the repo root: .\scripts\setup_env.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv)) {
    py -3.11 -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

Write-Output "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
