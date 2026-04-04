# Quick Start

Get BareMetalRT running in under 5 minutes.

## 1. Install

Download the installer from [GitHub Releases](https://github.com/baremetalrt/baremetalrt/releases/latest) and run it.

The installer will check for:
- **CUDA Toolkit 12.4+** — [Download](https://developer.nvidia.com/cuda-downloads)
- **TensorRT 10.15+** — [Download](https://developer.nvidia.com/tensorrt/download)

## 2. Create an Account

Go to [baremetalrt.ai/app](https://baremetalrt.ai/app) and sign up.

## 3. Link Your GPU

After installing, the daemon starts automatically. Open the web app — it auto-discovers the daemon running on your machine and connects it to your account.

## 4. Chat

Use the web interface at [baremetalrt.ai/app](https://baremetalrt.ai/app), or connect via the API:

```bash
curl https://baremetalrt.ai/v1/chat/completions \
  -H "Authorization: Bearer bmrt_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Generate API keys in [Account Settings](https://baremetalrt.ai/account).

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 |
| GPU | NVIDIA RTX 2000 series | RTX 3060+ (12GB VRAM) |
| CUDA | 12.4 | 12.6+ |
| TensorRT | 10.15 | Latest |
| RAM | 8 GB | 16 GB+ |

## Troubleshooting

- **Daemon not detected:** Make sure the BareMetalRT service is running (check System Tray).
- **GPU not showing:** Verify CUDA and TensorRT are installed — run `nvidia-smi` in a terminal.
- **SmartScreen warning:** The installer is not yet code-signed. Click "More info" then "Run anyway".
