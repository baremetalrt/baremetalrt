# BareMetalRT Installer Builder
# Usage: powershell -ExecutionPolicy Bypass -File build-installer.ps1
#
# Prerequisites:
#   - Python 3.13 with PyInstaller: pip install pyinstaller Pillow
#   - Inno Setup 6: winget install JRSoftware.InnoSetup

param(
    [switch]$SkipAssets,
    [switch]$SkipBuild,
    [string]$SignCert = "",
    [string]$SignPassword = ""
)

$ErrorActionPreference = "Stop"
$INSTALLER_DIR = $PSScriptRoot
$ROOT = Split-Path -Parent $INSTALLER_DIR
$VERSION = (Get-Content "$ROOT\VERSION" -Raw).Trim()
$OUTPUT_NAME = "BareMetalRT-$VERSION-Setup"

# Find Inno Setup
$ISCC = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $ISCC)) {
    $ISCC = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
}
if (-not (Test-Path $ISCC)) {
    $ISCC = "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
}

Write-Host ""
Write-Host "  BareMetalRT Installer Builder" -ForegroundColor Cyan
Write-Host "  Version: $VERSION" -ForegroundColor Gray
Write-Host ""

# ── Step 1: Generate assets ────────────────────────────
if (-not $SkipAssets) {
    Write-Host "[1/3] Generating installer assets..." -ForegroundColor Yellow
    Push-Location $INSTALLER_DIR
    try {
        & py -3.13 generate-assets.py
        if ($LASTEXITCODE -ne 0) { throw "Asset generation failed" }
    } finally { Pop-Location }
    Write-Host "  Assets ready" -ForegroundColor Green
} else {
    Write-Host "[1/3] Skipping asset generation" -ForegroundColor Gray
}

# ── Step 2: PyInstaller build ──────────────────────────
if (-not $SkipBuild) {
    Write-Host "[2/3] Building executable with PyInstaller..." -ForegroundColor Yellow
    Push-Location $ROOT
    try {
        if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
        if (Test-Path "build") { Remove-Item -Recurse -Force "build" }

        & py -3.13 -m PyInstaller baremetalrt.spec --noconfirm
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

        if (-not (Test-Path "dist\baremetalrt.exe")) {
            throw "Expected output dist\baremetalrt.exe not found"
        }
        $size = [math]::Round((Get-Item "dist\baremetalrt.exe").Length / 1MB, 1)
        Write-Host "  Built: baremetalrt.exe ($size MB)" -ForegroundColor Green
    } finally { Pop-Location }
} else {
    Write-Host "[2/3] Skipping PyInstaller build" -ForegroundColor Gray
}

# ── Step 3: Inno Setup ────────────────────────────────
Write-Host "[3/3] Building installer with Inno Setup..." -ForegroundColor Yellow

if (-not (Test-Path $ISCC)) {
    Write-Host "  Inno Setup not found. Install with:" -ForegroundColor Red
    Write-Host "    winget install JRSoftware.InnoSetup" -ForegroundColor Yellow
    throw "Inno Setup 6 not installed"
}

& $ISCC "$INSTALLER_DIR\baremetalrt.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }

$msiPath = "$ROOT\dist\$OUTPUT_NAME.exe"

# ── Optional: Code signing ────────────────────────────
if ($SignCert -and (Test-Path $msiPath)) {
    Write-Host "  Signing installer..." -ForegroundColor Yellow
    $signArgs = @("sign", "/f", $SignCert, "/t", "http://timestamp.digicert.com", "/d", "BareMetalRT", $msiPath)
    if ($SignPassword) {
        $signArgs = @("sign", "/f", $SignCert, "/p", $SignPassword, "/t", "http://timestamp.digicert.com", "/d", "BareMetalRT", $msiPath)
    }
    & signtool @signArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Warning: Signing failed (installer still usable)" -ForegroundColor Yellow
    }
}

# ── Done ──────────────────────────────────────────────
$finalSize = [math]::Round((Get-Item $msiPath).Length / 1MB, 1)
Write-Host ""
Write-Host "  ========================================" -ForegroundColor Green
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  ========================================" -ForegroundColor Green
Write-Host "  Output: dist\$OUTPUT_NAME.exe ($finalSize MB)" -ForegroundColor Cyan
Write-Host ""
