# Install HMR2.0 (4D-Humans, CVPR 2023) body-pose backend — Windows + Python 3.11.
#
# Run from imitation_upper root with the project venv activated.
# Idempotent — re-running skips already-installed pieces.
#
# Why this script avoids `pip install -e .[all]` on 4D-Humans:
#   - detectron2 build needs torch importable in pip's isolated env (fails)
#   - We don't need detectron2 (Hmr2BodyBackend uses MediaPipe Pose for the
#     body bbox, skipping ViTDet entirely)
#   - We don't need pyrender at runtime
#
# Reuses deps already installed by install_hamer.ps1 (torch+CUDA, chumpy,
# smplx, pytorch-lightning, timm, etc.). Total extra disk: ~700 MB checkpoint
# + cache. Time: ~5 min on a decent connection.
$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "venv not found at .\.venv — run scripts\setup_env.ps1 first"
    exit 1
}

$ProjectRoot = (Resolve-Path .).Path
$py  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"
$HmrDir = Join-Path $ProjectRoot "third_party\4D-Humans"
$SmplPkl = Join-Path $env:USERPROFILE ".cache\4DHumans\data\smpl\SMPL_NEUTRAL.pkl"

# ---------------------------------------------------------------------------
Write-Output "==> Step 1/5: shared deps (PyTorch + chumpy + smplx + ...)"
& $py -c "import torch, chumpy, smplx, pytorch_lightning, timm" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Output "  shared deps missing — run scripts\install_hamer.ps1 first (or install manually)."
    Write-Output "  Required: torch (cu121), chumpy --no-build-isolation, smplx==0.1.28,"
    Write-Output "            pytorch-lightning, timm, einops, gdown, pyrender, scikit-image, yacs"
    exit 1
} else {
    Write-Output "  shared deps OK"
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 2/5: clone + install 4D-Humans (skip detectron2)"
if (-not (Test-Path $HmrDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $HmrDir) | Out-Null
    git clone --recursive https://github.com/shubham-goel/4D-Humans.git $HmrDir
} else {
    Write-Output "  4D-Humans repo present at $HmrDir"
}

# Patch hmr2/utils/__init__.py — guard renderer imports (pyrender's OpenGL.EGL
# is Linux-only and breaks `from hmr2.models import ...` on Windows).
$utilsInit = Join-Path $HmrDir "hmr2\utils\__init__.py"
if (Test-Path $utilsInit) {
    $utilsContent = Get-Content $utilsInit -Raw
    if ($utilsContent -notmatch "try:\s*\r?\n\s*from \.renderer") {
        Write-Output "  patching $utilsInit (make renderer imports optional)"
        $patched = $utilsContent -replace `
            '(from \.renderer import Renderer.*?from \.skeleton_renderer import SkeletonRenderer)', `
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
}

# Install hmr2 package with --no-deps so detectron2 is skipped. Required deps
# are already satisfied above.
& $py -c "import hmr2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Push-Location $HmrDir
    try {
        & $pip install --no-deps -e .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 3/5: HMR2 checkpoint (~670 MB)"
# load_hmr2() auto-downloads on first call to ~/.cache/4DHumans/. We trigger
# the download here so the live test isn't delayed.
$CacheDir = Join-Path $env:USERPROFILE ".cache\4DHumans"
$CkptGlob = Join-Path $CacheDir "logs\train\multiruns\hmr2\0\checkpoints\*.ckpt"
if (Test-Path $CkptGlob) {
    Write-Output "  checkpoint already cached at $CacheDir"
} else {
    Write-Output "  triggering load_hmr2() to auto-download..."
    & $py -c @'
from hmr2.models import download_models, DEFAULT_CHECKPOINT
import os, pathlib
cache = pathlib.Path(os.path.expanduser("~/.cache/4DHumans"))
download_models(cache)
print("downloaded to", cache)
'@
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "auto-download failed; check network. Manual mirror: https://huggingface.co/spaces/brjathu/HMR2.0"
    }
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 4/5: HMR2 import sanity check"
& $py -c "from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT; print('hmr2 import OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "HMR2 import failed — see traceback above"
    exit 1
}

# ---------------------------------------------------------------------------
Write-Output ""
Write-Output "==> Step 5/5: SMPL model file (MANUAL — license-bound)"
if (Test-Path $SmplPkl) {
    Write-Output "  SMPL_NEUTRAL.pkl present at $SmplPkl"
    Write-Output ""
    Write-Output "==> Install complete. Activate HMR2 in config\ubp.yaml:"
    Write-Output "      tracker.realsense.body_backend: hmr2"
    Write-Output ""
    Write-Output "Test with:  python -m app.debug_retarget --config .\config\ubp.yaml --duration 10"
} else {
    Write-Output "  MISSING — download manually:"
    Write-Output "    1. Register at https://smpl.is.tue.mpg.de/ (free, non-commercial)"
    Write-Output "    2. Download 'Download SMPL for Python users' → SMPL_python_v.1.1.0.zip"
    Write-Output "       (or any SMPL v1.0/v1.1 package — we need the NEUTRAL model)"
    Write-Output "    3. Extract the neutral .pkl and rename/copy to:"
    Write-Output "         $SmplPkl"
    Write-Output "       Typical file inside the zip:"
    Write-Output "         models/basicmodel_neutral_lbs_10_207_0_v1.0.0.pkl"
    Write-Output "       Renaming to SMPL_NEUTRAL.pkl satisfies HMR2's loader."
    Write-Output ""
    Write-Output "After placing SMPL_NEUTRAL.pkl, set tracker.realsense.body_backend: hmr2 in config\ubp.yaml."
}
