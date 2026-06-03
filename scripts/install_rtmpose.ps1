# Install the RTMPose body-pose backend deps — Windows + Python 3.11 + NVIDIA GPU.
#
# Run from imitation_upper root with the project venv activated.
# Idempotent — re-running skips already-installed pieces.
#
# Installs:
#   - rtmlib            (high-level RTMPose Body wrapper; pulls its own onnx model on first use)
#   - onnxruntime-gpu   (CUDA execution provider)
#
# rtmlib's Body downloads + caches its ONNX models under ~/.cache on first call.
# For an offline operating PC, run this once online so the cache is warm.
#
# Total extra disk: ~300 MB (onnxruntime-gpu + cached RTMPose/det ONNX models).
$ErrorActionPreference = "Stop"

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Error "venv not found at .\.venv — run scripts\setup_env.ps1 first"
    exit 1
}

$ProjectRoot = (Resolve-Path .).Path
$py  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$pip = Join-Path $ProjectRoot ".venv\Scripts\pip.exe"

# ---------------------------------------------------------------------------
# Order matters: rtmlib depends on CPU `onnxruntime`. If onnxruntime-gpu is
# installed first, pip pulls CPU onnxruntime afterwards into the SAME package
# dir and shadows the GPU build (CUDAExecutionProvider disappears). So install
# rtmlib first, then force the GPU build to win.
Write-Output "==> Step 1/3: rtmlib (pulls CPU onnxruntime as a dependency)"
& $py -c "import rtmlib" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $pip install rtmlib
} else {
    Write-Output "  rtmlib already installed"
}

# ---------------------------------------------------------------------------
Write-Output "==> Step 2/3: replace CPU onnxruntime with onnxruntime-gpu"
& $pip uninstall -y onnxruntime 2>$null
& $pip install --force-reinstall --no-deps onnxruntime-gpu

# ---------------------------------------------------------------------------
Write-Output "==> Step 3/3: verify CUDAExecutionProvider is available"
& $py -c "import onnxruntime as ort; ps = ort.get_available_providers(); print('providers:', ps); assert 'CUDAExecutionProvider' in ps, 'CUDAExecutionProvider missing — check CUDA/cuDNN install'"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "CUDAExecutionProvider not available — RTMPose will fall back to CPU (slow). Check your CUDA + cuDNN runtime."
} else {
    Write-Output "  CUDA provider OK"
}

Write-Output ""
Write-Output "Done. Enable with config/ubp.yaml: tracker.realsense.body_backend: rtmpose"
Write-Output "First live run downloads the RTMPose ONNX models to ~/.cache (one-time)."
