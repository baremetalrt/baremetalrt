# BareMetalRT

[![PyPI](https://img.shields.io/pypi/v/baremetalrt)](https://pypi.org/project/baremetalrt/)
[![Release](https://img.shields.io/github/v/release/baremetalrt/baremetalrt?include_prereleases)](https://github.com/baremetalrt/baremetalrt/releases/latest)
[![License](https://img.shields.io/badge/license-proprietary-blue)](LICENSE)

**The world's first global GPU-native edge compute mesh.**

Intelligence shouldn't be owned by the hyperscalers alone. BareMetalRT turns the 200+ million NVIDIA GPUs running Windows into a distributed compute mesh — using NVIDIA's own TensorRT-LLM CUDA kernels, the same engine that powers cloud inference APIs. Built for the edge, not the cloud.

**[Download Installer](https://github.com/baremetalrt/baremetalrt/releases/latest)** | **[Live Demo](https://baremetalrt.ai/demo)** | **[Documentation](https://baremetalrt.ai/docs)** | **[PyPI](https://pypi.org/project/baremetalrt/)**

## How It Works

1. **Install** the BareMetalRT daemon on each Windows machine with an NVIDIA GPU
2. **Connect** your GPU to your account at [baremetalrt.ai](https://baremetalrt.ai)
3. **Run inference** — the system automatically shards models across your available GPUs

```
┌──────────────────────────────────┐
│  baremetalrt.ai                  │
│  Auth, routing, OpenAI-compat API│
└──────────────┬───────────────────┘
               │ WebSocket
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Node A │ │Node B │ │Node C │
│3090   │ │4060   │ │3060   │
│24GB   │ │8GB    │ │12GB   │
│Daemon │ │Daemon │ │Daemon │
└───────┘ └───────┘ └───────┘
```

## System Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU (RTX 2000+ recommended)
- CUDA Toolkit 12.4+
- TensorRT 10.15+

## Quick Start

### 1. Download and Install

Download the latest installer from [GitHub Releases](https://github.com/baremetalrt/baremetalrt/releases/latest) and run it. The installer will check for NVIDIA prerequisites and guide you through setup.

### 2. Connect Your GPU

Sign in at [baremetalrt.ai/app](https://baremetalrt.ai/app) and link your GPU using the one-click claim flow.

### 3. Chat

Use the web interface at [baremetalrt.ai/app](https://baremetalrt.ai/app) or connect any OpenAI-compatible client:

```bash
curl https://baremetalrt.ai/v1/chat/completions \
  -H "Authorization: Bearer bmrt_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## CLI

```bash
pip install baremetalrt
bmrt status
bmrt models
bmrt run mistral-7b
```

## What's in This Repo

This is the **public product repo** — the server, web UI, installer, and documentation.

```
baremetalrt/
├── server/        # FastAPI server (auth, chat relay, node management)
├── web/           # Product web app (chat UI, account, downloads)
├── site/          # Landing page and demo
├── installer/     # Windows installer (Inno Setup)
├── cli/           # bmrt CLI (pip install baremetalrt)
├── docs/          # Documentation
└── paper/         # Technical paper
```

The inference engine, transport layer, and daemon are in a separate private repository.

## API

BareMetalRT exposes an OpenAI-compatible API. See [API docs](https://baremetalrt.ai/docs) or the [API reference](docs/API.md).

## Current Status

**v0.5.1-beta** — [Changelog](CHANGELOG.md)

- Single-GPU and TP=2 multi-GPU inference on Windows
- Mistral 7B at 12.5 tok/s across heterogeneous GPUs over TCP
- Web chat UI with streaming
- Windows installer with one-click GPU claiming
- OpenAI-compatible API

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

## License

BareMetalRT is proprietary software. See [LICENSE](LICENSE) for terms.
