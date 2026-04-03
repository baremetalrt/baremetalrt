# BareMetalRT Daemon Installer
# Run as Administrator: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"
$INSTALL_DIR = "$env:ProgramFiles\BareMetalRT"
$DATA_DIR = "$env:APPDATA\BareMetalRT"
$REPO_URL = "https://github.com/baremetalrt/baremetalrt"

Write-Host ""
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host "  |         B A R E M E T A L R T        |" -ForegroundColor Cyan
Write-Host "  |    GPU Inference Daemon Installer     |" -ForegroundColor Cyan
Write-Host "  +======================================+" -ForegroundColor Cyan
Write-Host ""

# --- Check Admin ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] Please run as Administrator" -ForegroundColor Red
    Write-Host "    Right-click PowerShell -> Run as Administrator" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Check NVIDIA GPU ---
Write-Host "[1/6] Checking NVIDIA GPU..." -ForegroundColor Yellow
$gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -match "NVIDIA" }
if (-not $gpu) {
    Write-Host "[!] No NVIDIA GPU detected. BareMetalRT requires an NVIDIA GPU." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  Found: $($gpu.Name)" -ForegroundColor Green

# --- Check Python 3.13 ---
Write-Host "[2/6] Checking Python 3.13..." -ForegroundColor Yellow
$python = $null
try {
    $ver = & py -3.13 --version 2>&1
    if ($ver -match "3.13") {
        $python = "py -3.13"
        Write-Host "  Found: $ver" -ForegroundColor Green
    }
} catch {}

if (-not $python) {
    Write-Host "  Python 3.13 not found." -ForegroundColor Yellow
    Write-Host "  Opening python.org download page..." -ForegroundColor Cyan
    Start-Process "https://www.python.org/downloads/"
    Write-Host ""
    Write-Host "  Please install Python 3.13, then re-run this installer." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Check CUDA Toolkit ---
Write-Host "[3/6] Checking CUDA Toolkit..." -ForegroundColor Yellow
$cudaPath = $null
foreach ($v in @("12.8", "12.6", "12.4")) {
    $p = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v$v"
    if (Test-Path "$p\bin\nvcc.exe") {
        $cudaPath = $p
        Write-Host "  Found: CUDA $v at $p" -ForegroundColor Green
        break
    }
}
if (-not $cudaPath) {
    Write-Host "  CUDA Toolkit 12.4+ not found." -ForegroundColor Yellow
    Write-Host "  Opening NVIDIA CUDA download page..." -ForegroundColor Cyan
    Start-Process "https://developer.nvidia.com/cuda-downloads"
    Write-Host ""
    Write-Host "  Please install CUDA Toolkit 12.4+, then re-run this installer." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Check TensorRT ---
Write-Host "[4/6] Checking TensorRT..." -ForegroundColor Yellow
$trtPath = $null
foreach ($v in @("10.15.1.29", "10.15.1", "10.16.0.72")) {
    $p = "C:\TensorRT\TensorRT-$v"
    if (Test-Path "$p\bin\nvinfer_10.dll") {
        $trtPath = $p
        Write-Host "  Found: TensorRT at $p" -ForegroundColor Green
        break
    }
}
if (-not $trtPath) {
    Write-Host "  TensorRT not found." -ForegroundColor Yellow
    Write-Host "  Opening NVIDIA TensorRT download page..." -ForegroundColor Cyan
    Start-Process "https://developer.nvidia.com/tensorrt/download"
    Write-Host ""
    Write-Host "  Please install TensorRT to C:\TensorRT\, then re-run this installer." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Install Python packages ---
Write-Host "[5/6] Installing Python packages..." -ForegroundColor Yellow
$pipPkgs = @("torch --extra-index-url https://download.pytorch.org/whl/cu124", "tensorrt", "transformers", "pynvml", "httpx", "fastapi", "uvicorn", "websockets", "huggingface_hub", "sentencepiece", "protobuf")
foreach ($pkg in $pipPkgs) {
    Write-Host "  Installing $($pkg.Split(' ')[0])..." -ForegroundColor Gray
    & py -3.13 -m pip install -q $pkg.Split(' ') 2>&1 | Out-Null
}
Write-Host "  Python packages installed" -ForegroundColor Green

# --- Clone/Update repo ---
Write-Host "[6/6] Installing BareMetalRT daemon..." -ForegroundColor Yellow
if (Test-Path "$INSTALL_DIR\.git") {
    Write-Host "  Updating existing installation..." -ForegroundColor Gray
    Push-Location $INSTALL_DIR
    & git pull origin main 2>&1 | Out-Null
    Pop-Location
} else {
    Write-Host "  Cloning repository..." -ForegroundColor Gray
    & git clone $REPO_URL $INSTALL_DIR 2>&1 | Out-Null
}

# --- Create data directory ---
if (-not (Test-Path $DATA_DIR)) {
    New-Item -ItemType Directory -Path $DATA_DIR | Out-Null
}

# --- Create start script ---
$startScript = @"
@echo off
cd /d "$INSTALL_DIR\daemon"
py -3.13 daemon.py %*
"@
$startScript | Out-File -FilePath "$INSTALL_DIR\start-daemon.bat" -Encoding ASCII

# --- Add firewall rules ---
Write-Host ""
Write-Host "Adding firewall rules..." -ForegroundColor Gray
netsh advfirewall firewall delete rule name="BareMetalRT" 2>&1 | Out-Null
netsh advfirewall firewall add rule name="BareMetalRT" dir=in action=allow protocol=TCP localport=8080 2>&1 | Out-Null

# --- Done ---
Write-Host ""
Write-Host "  ========================================" -ForegroundColor Green
Write-Host "  BareMetalRT installed successfully!" -ForegroundColor Green
Write-Host "  ========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Install directory: $INSTALL_DIR" -ForegroundColor Cyan
Write-Host "  Config directory:  $DATA_DIR" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open https://baremetalrt.ai/app and sign in" -ForegroundColor White
Write-Host "  2. Go to Account Settings > API Keys > Generate" -ForegroundColor White
Write-Host "  3. Save your API key to $DATA_DIR\config.json:" -ForegroundColor White
Write-Host '     {"api_key": "bmrt_your_key", "orchestrator": "https://baremetalrt.ai"}' -ForegroundColor Gray
Write-Host "  4. Run: $INSTALL_DIR\start-daemon.bat" -ForegroundColor White
Write-Host ""
Write-Host "  Or start from the system tray:" -ForegroundColor Yellow
Write-Host "  py -3.13 $INSTALL_DIR\daemon\baremetalrt.py" -ForegroundColor Gray
Write-Host ""
Read-Host "Press Enter to finish"
