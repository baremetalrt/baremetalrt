# BareMetalRT Daemon Installer (Beta)

One-script installer for the BareMetalRT GPU inference daemon on Windows.

## Requirements

- **Windows 10/11**
- **NVIDIA GPU** (RTX 2000 series or newer recommended)
- **Git** installed and in PATH
- **Internet connection** for downloads

The installer will check for and guide you through installing:
- Python 3.13
- CUDA Toolkit 12.4+
- TensorRT 10.15+

## Install

Run PowerShell **as Administrator**:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

## After Install

1. Open https://baremetalrt.ai/app and sign in
2. Go to Account Settings → API Keys → Generate New Key
3. Copy your key and save to `%APPDATA%\BareMetalRT\config.json`:
```json
{
    "api_key": "bmrt_your_key_here",
    "orchestrator": "https://baremetalrt.ai"
}
```
4. Run `start-daemon.bat` or launch from system tray

## Status: Beta

This is an early release. Expect rough edges. Report issues at https://github.com/baremetalrt/baremetalrt/issues
