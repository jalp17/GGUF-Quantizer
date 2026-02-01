param (
    [switch]$ClearCache = $false
)

# GGUF-Quantizer Setup Script for Windows
# This script clones llama.cpp and applies custom image quantization patches.

$RepoUrl = "https://github.com/ggerganov/llama.cpp.git"
$TargetDir = "llama.cpp"
# LTS Base Commit (v1.2.1 LTS - Master Sync 2026-02-01)
$LtsHash = "2634ed207a17db1a54bd8df0555bd8499a6ab691"

Write-Host "=== GGUF-Quantizer Setup (Windows) ===" -ForegroundColor Cyan

if ($ClearCache) {
    Write-Host "Cleaning ccache..." -ForegroundColor Yellow
    ccache -C
}

if (-not (Test-Path $TargetDir)) {
    Write-Host "Cloning llama.cpp (LTS Version)..."
    git clone $RepoUrl $TargetDir
    Set-Location $TargetDir
    git checkout $LtsHash
    Set-Location ..
}
else {
    Write-Host "Directory $TargetDir already exists. Ensuring it's on LTS version..."
    Set-Location $TargetDir
    git checkout $LtsHash
    Set-Location ..
}

Set-Location $TargetDir

Write-Host "Applying patches..."
$Patches = Get-ChildItem -Path "..\patches\*.patch" | Sort-Object Name

foreach ($patch in $Patches) {
    if ($patch.Length -eq 0) {
        Write-Host "Skipping empty patch: $($patch.Name)" -ForegroundColor Gray
        continue
    }
    Write-Host "Applying $($patch.Name)..."
    git apply --verbose $patch.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error applying $($patch.Name). Please check for conflicts." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "=== Success! ===" -ForegroundColor Green
Write-Host "Now you can build llama.cpp using your preferred method (cmake or make)."
Write-Host "Example:"
Write-Host "  cmake -B build"
Write-Host "  cmake --build build --config Release"
