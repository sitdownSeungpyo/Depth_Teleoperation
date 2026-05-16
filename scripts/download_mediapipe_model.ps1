# Download the MediaPipe Pose Landmarker (lite) model asset.
# Saves to .\models\pose_landmarker_lite.task — the path config/default.yaml expects.
$ErrorActionPreference = "Stop"

# 정확도 우선 (heavy) > full > lite (속도 우선)
# 첫 인자로 모델 변종 선택: lite / full / heavy. 기본 full (정확도 + 속도 균형).
$Variant = if ($args.Count -gt 0) { $args[0] } else { "full" }

$Url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$Variant/float16/latest/pose_landmarker_$Variant.task"
$OutDir = Join-Path $PSScriptRoot "..\models"
$OutPath = Join-Path $OutDir "pose_landmarker_$Variant.task"

if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

if (Test-Path $OutPath) {
    Write-Output "Model already at $OutPath ($(((Get-Item $OutPath).Length / 1MB).ToString('F1')) MB) — skipping download."
    exit 0
}

Write-Output "Downloading $Url ..."
Invoke-WebRequest -Uri $Url -OutFile $OutPath -UseBasicParsing
Write-Output "Saved to $OutPath ($(((Get-Item $OutPath).Length / 1MB).ToString('F1')) MB)"
