# Install HaMeR (Berkeley CVPR 2024) hand-tracking backend.
# Run from the imitation_upper root with the project venv activated.
#
# Steps:
#   1. PyTorch + CUDA 12.1 wheels (~3 GB)
#   2. HaMeR git+pip install (clones + installs editable)
#   3. HaMeR pretrained weights via the repo's fetch script (~1.5 GB)
#   4. Print MANO download instructions (MUST be done manually, license-bound)
#
# Total disk: ~5 GB. Total time on a decent connection: ~10 min.
$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "venv not found at .\.venv — run scripts\setup_env.ps1 first"
    exit 1
}

$py = ".\.venv\Scripts\python.exe"
$pip = ".\.venv\Scripts\pip.exe"

Write-Output "==> Step 1/4: install PyTorch + CUDA 12.1 wheels"
& $pip install --upgrade pip
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Sanity: CUDA visible?
& $py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDA not detected. HaMeR will fall back to CPU (~500 ms/hand)."
}

Write-Output "==> Step 2/4: clone + install HaMeR"
$HamerDir = Join-Path $PSScriptRoot "..\third_party\hamer"
if (-not (Test-Path $HamerDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $HamerDir) | Out-Null
    git clone --recursive https://github.com/geopavlakos/hamer.git $HamerDir
} else {
    Write-Output "  HaMeR repo already at $HamerDir — pulling latest"
    git -C $HamerDir pull --rebase
}
Push-Location $HamerDir
try {
    & $pip install -e .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}

Write-Output "==> Step 3/4: fetch HaMeR pretrained weights"
Push-Location $HamerDir
try {
    if (Test-Path .\fetch_demo_data.sh) {
        # The shell script just curls a few files; we run with bash (Git Bash on Windows).
        bash ./fetch_demo_data.sh
        if ($LASTEXITCODE -ne 0) { Write-Warning "fetch_demo_data.sh exited $LASTEXITCODE — check manually" }
    } else {
        Write-Warning "fetch_demo_data.sh missing; refer to HaMeR README for weight URLs"
    }
} finally {
    Pop-Location
}

Write-Output ""
Write-Output "==> Step 4/4: MANO model files (MANUAL — license-bound)"
Write-Output "  1. Register at https://mano.is.tue.mpg.de/ (free, non-commercial)"
Write-Output "  2. Download MANO_v1_2.zip from 'Models & Code'"
Write-Output "  3. Extract MANO_RIGHT.pkl to:"
Write-Output "       $HamerDir\_DATA\data\mano\MANO_RIGHT.pkl"
Write-Output "  (HaMeR mirrors right hand for left, so only MANO_RIGHT.pkl is needed.)"
Write-Output ""
Write-Output "==> Done. To enable HaMeR, set in config\ubp.yaml:"
Write-Output "       tracker.realsense.hand_backend: hamer"
Write-Output ""
Write-Output "Verify with:  python -m app.debug_retarget --config .\config\ubp.yaml --duration 10"
