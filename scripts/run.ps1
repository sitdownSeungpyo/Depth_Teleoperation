# Run the imitation pipeline. Activates the venv if present, then forwards args to app.main.
# Examples:
#   .\scripts\run.ps1 --config .\config\default.yaml --tracker mock --publisher mock --replay .\tests\fixtures\arm_circle.jsonl
#   .\scripts\run.ps1 --config .\config\default.yaml --tracker realsense --publisher mock
$ErrorActionPreference = "Stop"

if (Test-Path .\.venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
}

python -m app.main @args
