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
# chumpy 0.70 (pulled in for SMPL .pkl loading) predates Python 3.11 + numpy 2:
#   - inspect.getargspec was removed in 3.11 (chumpy/ch.py still calls it)
#   - `from numpy import bool, int, float, ... ` aliases removed in numpy 2
# Patch chumpy/__init__.py idempotently so `import chumpy` (and SMPL load) works.
Write-Output "==> Step 0/5: patch chumpy for Python 3.11 + numpy 2"
$chumpyInit = Join-Path $ProjectRoot ".venv\Lib\site-packages\chumpy\__init__.py"
if (Test-Path $chumpyInit) {
    $c = Get-Content $chumpyInit -Raw
    if ($c -match 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf') {
        Write-Output "  patching $chumpyInit"
        $c = $c -replace 'from \.ch import \*', @'
import inspect as _inspect
if not hasattr(_inspect, "getargspec"):
    _inspect.getargspec = _inspect.getfullargspec
from .ch import *
'@
        $c = $c -replace 'from numpy import bool, int, float, complex, object, unicode, str, nan, inf', @'
from builtins import bool, int, float, complex, object, str
unicode = str
from numpy import nan, inf
'@
        Set-Content -Path $chumpyInit -Value $c -NoNewline
    } else {
        Write-Output "  chumpy already patched (or not present yet)"
    }
} else {
    Write-Output "  chumpy not installed yet — will be patched on a later run if needed"
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 1/5: shared deps (PyTorch + chumpy + smplx + ...)"
# NB: relax ErrorActionPreference around native-exe stderr redirects. In PS 5.1,
# `2>$null` on a python check that fails wraps stderr as a terminating
# NativeCommandError under -ErrorActionPreference Stop, aborting the script.
$ErrorActionPreference = 'SilentlyContinue'
& $py -c "import torch, chumpy, smplx, pytorch_lightning, timm" 2>$null
$sharedOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = 'Stop'
if (-not $sharedOk) {
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
            '(?s)(from \.renderer import Renderer.*?from \.skeleton_renderer import SkeletonRenderer)', `
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

# Patch hmr2/models/hmr2.py — SkeletonRenderer/MeshRenderer are None on Windows
# (made optional above). Guard their instantiation in __init__ so loading the
# model doesn't call None(...) (they're visualization-only, unused for inference).
$hmr2Py = Join-Path $HmrDir "hmr2\models\hmr2.py"
if (Test-Path $hmr2Py) {
    $hc = Get-Content $hmr2Py -Raw
    if ($hc -match 'if init_renderer:') {
        Write-Output "  patching $hmr2Py (guard renderer instantiation)"
        $hc = $hc -replace 'if init_renderer:', 'if init_renderer and SkeletonRenderer is not None and MeshRenderer is not None:'
        Set-Content -Path $hmr2Py -Value $hc -NoNewline
    }
}

# Install hmr2 package with --no-deps so detectron2 is skipped. Required deps
# are already satisfied above.
$ErrorActionPreference = 'SilentlyContinue'
& $py -c "import hmr2" 2>$null
$hmr2Installed = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = 'Stop'
if (-not $hmr2Installed) {
    Push-Location $HmrDir
    try {
        & $pip install --no-deps -e .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

# `pip install --no-deps` above skips HMR2 runtime deps not in the shared HaMeR
# set. Install the ones the inference path needs (pip no-ops if already present):
#   omegaconf / hydra-core / pyrootutils -> load model_config.yaml
#   webdataset                           -> hmr2.datasets.vitdet_dataset (ViTDetDataset)
#   dill                                 -> convert_pkl() for the SMPL .pkl (Step 5)
Write-Output "  installing HMR2 inference deps (omegaconf, hydra-core, pyrootutils, webdataset, dill)"
& $pip install omegaconf hydra-core pyrootutils webdataset dill

# ---------------------------------------------------------------------------
Write-Output "==> Step 3/5: HMR2 checkpoint (~670 MB)"
# load_hmr2() auto-downloads on first call to ~/.cache/4DHumans/. We trigger
# the download here so the live test isn't delayed.
$CacheDir = Join-Path $env:USERPROFILE ".cache\4DHumans"
$CkptGlob = Join-Path $CacheDir "logs\train\multiruns\hmr2\0\checkpoints\*.ckpt"
if (Test-Path $CkptGlob) {
    Write-Output "  checkpoint already cached at $CacheDir"
} else {
    Write-Output "  triggering download_models() to auto-download..."
    # Single -c line, no inner path string: download_models() defaults to
    # ~/.cache/4DHumans. (A here-string with quoted paths gets mangled when PS
    # passes it to python.exe -c.)
    & $py -c "from hmr2.models import download_models; download_models()"
    # download_models() saves hmr2_data.tar.gz but its auto-extract fails on
    # Windows (the archive is actually a plain tar mis-named .tar.gz), leaving
    # only the tarball. Extract it explicitly with bsdtar (ships with Win10+).
    $tar = Join-Path $CacheDir "hmr2_data.tar.gz"
    if ((-not (Test-Path $CkptGlob)) -and (Test-Path $tar)) {
        Write-Output "  extracting hmr2_data.tar.gz ..."
        tar -xf "$tar" -C "$CacheDir"
    }
    if (-not (Test-Path $CkptGlob)) {
        Write-Warning "checkpoint still missing; check network/extraction. Manual mirror: https://huggingface.co/spaces/brjathu/HMR2.0"
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
