# Install HaMeR (Berkeley CVPR 2024) hand-tracking backend — Windows + Python 3.11.
#
# Run from imitation_upper root with the project venv activated.
# All steps are idempotent — re-running skips already-installed pieces.
#
# Why this script avoids `pip install -e .[all]` on hamer:
#   - detectron2 build needs torch importable in pip's isolated env (fails)
#   - chumpy setup.py uses `import pip` which is unavailable in isolated env
#   - We don't actually need detectron2 (HamerHandBackend gets bbox from
#     MediaPipe Pose wrist coords, skipping ViTDet entirely)
#   - We don't need pyrender at runtime (rendering disabled in our backend)
#
# Total disk: ~13 GB (5.7 GB tarball + extracted weights + torch wheels).
# Total time on decent connection: ~15 min.
$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "venv not found at .\.venv — run scripts\setup_env.ps1 first"
    exit 1
}

$ProjectRoot = (Resolve-Path .).Path
$py  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$HamerDir = Join-Path $ProjectRoot "third_party\hamer"
$WeightCkpt = Join-Path $HamerDir "_DATA\hamer_ckpts\checkpoints\hamer.ckpt"
$ManoPkl = Join-Path $HamerDir "_DATA\data\mano\MANO_RIGHT.pkl"

# ---------------------------------------------------------------------------
Write-Output "==> Step 1/6: PyTorch + CUDA 12.1 wheels"
& $py -c "import torch; assert torch.cuda.is_available()" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Output "  torch+cuda already present — skipping"
} else {
    & $pip install --upgrade pip setuptools wheel
    & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $py -c "import torch; print('CUDA OK:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE — HaMeR will be slow on CPU')"
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 2/6: HaMeR-required Python deps"
& $py -c "import pyrender, pytorch_lightning, skimage, yacs, gdown, timm" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Output "  base deps already installed — skipping"
} else {
    & $pip install gdown pyrender pytorch-lightning scikit-image yacs timm einops
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# chumpy: setup.py uses `import pip` which fails in isolated builds.
# --no-build-isolation forces use of our venv's pip, which works.
& $py -c "import chumpy" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Output "  installing chumpy --no-build-isolation"
    & $pip install chumpy --no-build-isolation
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# smplx pulls MANO; chumpy must be already satisfied.
& $py -c "import smplx" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $pip install smplx==0.1.28
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 3/6: clone + install HaMeR (skip detectron2)"
if (-not (Test-Path $HamerDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $HamerDir) | Out-Null
    git clone --recursive https://github.com/geopavlakos/hamer.git $HamerDir
} else {
    Write-Output "  HaMeR repo present at $HamerDir"
}

# Patch hamer/utils/__init__.py so renderer imports are optional (pyrender's
# OpenGL.EGL is Linux-only and breaks `from hamer.models import ...` on Windows).
$utilsInit = Join-Path $HamerDir "hamer\utils\__init__.py"
$utilsContent = Get-Content $utilsInit -Raw
if ($utilsContent -notmatch "try:\s*\r?\n\s*from \.renderer") {
    Write-Output "  patching $utilsInit (make renderer imports optional)"
    $patched = $utilsContent -replace `
        '(from \.renderer import Renderer\r?\nfrom \.mesh_renderer import MeshRenderer\r?\nfrom \.skeleton_renderer import SkeletonRenderer)', `
@'
# Renderer imports are optional — pyrender fails to load on Windows without
# EGL/OSMesa. We only need the inference path, so swallow ImportError.
try:
    from .renderer import Renderer
    from .mesh_renderer import MeshRenderer
    from .skeleton_renderer import SkeletonRenderer
except ImportError:
    Renderer = None
    MeshRenderer = None
    SkeletonRenderer = None
'@
    Set-Content -Path $utilsInit -Value $patched -NoNewline
}

# Install hamer with --no-deps so detectron2 is not pulled in. We already
# satisfied the deps we actually need above.
& $py -c "import hamer" 2>$null
if ($LASTEXITCODE -ne 0) {
    Push-Location $HamerDir
    try {
        & $pip install --no-deps -e .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 4/6: HaMeR pretrained weights (~5.7 GB)"
if (Test-Path $WeightCkpt) {
    Write-Output "  weights already at $WeightCkpt — skipping"
} else {
    Push-Location $HamerDir
    try {
        # UT Austin direct link is faster + more reliable than gdown.
        $WeightUrl = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
        Write-Output "  downloading $WeightUrl"
        Invoke-WebRequest -Uri $WeightUrl -OutFile "hamer_demo_data.tar.gz" -UseBasicParsing
        Write-Output "  extracting..."
        tar --warning=no-unknown-keyword --exclude=".*" -xf hamer_demo_data.tar.gz
        Remove-Item hamer_demo_data.tar.gz -Force
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 5/6: HaMeR import sanity check"
& $py -c "from hamer.models import load_hamer, DEFAULT_CHECKPOINT; print('OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "HaMeR import failed — see traceback above"
    exit 1
}

# ---------------------------------------------------------------------------
Write-Output ""
Write-Output "==> Step 6/6: MANO model files (MANUAL — license-bound)"
if (Test-Path $ManoPkl) {
    Write-Output "  MANO_RIGHT.pkl already at $ManoPkl"
    Write-Output ""
    Write-Output "==> Install complete. Activate HaMeR in config\ubp.yaml:"
    Write-Output "      tracker.realsense.hand_backend: hamer"
    Write-Output ""
    Write-Output "Test with:  python -m app.debug_retarget --config .\config\ubp.yaml --duration 10"
} else {
    Write-Output "  MISSING — download manually:"
    Write-Output "    1. Register at https://mano.is.tue.mpg.de/ (free, non-commercial)"
    Write-Output "    2. Download 'MANO_v1_2.zip' from 'Models & Code'"
    Write-Output "    3. Extract MANO_RIGHT.pkl to:"
    Write-Output "         $ManoPkl"
    Write-Output ""
    Write-Output "After placing MANO_RIGHT.pkl, set tracker.realsense.hand_backend: hamer in config\ubp.yaml."
}
