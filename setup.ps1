# GGUF-Quantizer Setup Script for Windows
# This script clones llama.cpp and applies custom image quantization patches.

$RepoUrl = "https://github.com/ggerganov/llama.cpp.git"
$TargetDir = "llama.cpp"

Write-Host "=== GGUF-Quantizer Setup (Windows) ===" -ForegroundColor Cyan

if (-not (Test-Path $TargetDir)) {
    Write-Host "Cloning llama.cpp..."
    git clone --depth 1 $RepoUrl $TargetDir
} else {
    Write-Host "Directory $TargetDir already exists. Skipping clone."
}

Set-Location $TargetDir

Write-Host "Applying patches..."
$Patches = Get-ChildItem -Path "..\patches\*.patch" | Sort-Object Name

foreach ($patch in $Patches) {
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
