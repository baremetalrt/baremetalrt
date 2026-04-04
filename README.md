# BareMetalRT

[![PyPI](https://img.shields.io/pypi/v/baremetalrt)](https://pypi.org/project/baremetalrt/)
[![Release](https://img.shields.io/github/v/release/baremetalrt/baremetalrt?include_prereleases)](https://github.com/baremetalrt/baremetalrt/releases/latest)
[![License](https://img.shields.io/badge/license-proprietary-blue)](LICENSE)

**The world's first global GPU-native edge compute mesh.**

Intelligence shouldn't be owned by the hyperscalers alone. BareMetalRT turns the 200+ million NVIDIA GPUs running Windows into a distributed compute mesh — using NVIDIA's own TensorRT-LLM CUDA kernels, the same engine that powers cloud inference APIs. Built for the edge, not the cloud.

**[Download Installer](https://github.com/baremetalrt/baremetalrt/releases/latest)** | **[Live Demo](https://baremetalrt.ai/demo)** | **[Documentation](https://baremetalrt.ai/docs)** | **[PyPI](https://pypi.org/project/baremetalrt/)** | **[Technical Paper](paper/main.pdf)**

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
└───┬───┘ └───┬───┘ └───┬───┘
    └── TCP AllReduce ──┘
```

## Why BareMetalRT

| | Cloud (OpenAI, etc.) | Local (Ollama, LM Studio) | Distributed (Petals, Exo) | **BareMetalRT** |
|---|---|---|---|---|
| **Kernels** | Optimized (proprietary) | Generic CUDA | Generic CUDA | **TensorRT-LLM (1,500+ optimized .cu)** |
| **Multi-GPU** | NVLink (datacenter only) | Single GPU only | Pipeline parallelism | **Tensor parallelism over TCP** |
| **Heterogeneous GPUs** | No | N/A | No | **Yes — different VRAM, different SMs** |
| **Windows native** | N/A | Yes | Partial | **Yes** |
| **Cost** | Per-token pricing | Free (your hardware) | Free (your hardware) | **Free (your hardware)** |
| **Privacy** | Data leaves your machine | Fully local | Weights distributed | **Weights distributed, execution local** |

**The key difference:** every other consumer GPU project uses pipeline parallelism, which leaves GPUs idle 50% of the time. BareMetalRT is the first to achieve tensor parallelism across heterogeneous consumer GPUs — both GPUs compute on every layer, every token.

## Benchmarks

Tested with Mistral 7B Instruct (14 GB FP16) across an RTX 4070 Super (12 GB) and an RTX 4060 Laptop (8 GB) — **a model too large for either GPU alone**.

| Configuration | Latency | Throughput | Notes |
|---|---|---|---|
| llama.cpp — 4070S single GPU | 3.4 ms/tok | 295 tok/s | Q8 quantized, fits on one card |
| BareMetalRT TP=2 — WiFi | 277 ms/tok | 3.6 tok/s | TinyLlama 1.1B, 316ms ping |
| BareMetalRT TP=2 — Ethernet | 276 ms/tok | 3.6 tok/s | TinyLlama 1.1B, 1ms ping |
| **BareMetalRT TP=2 — Mistral 7B** | **80 ms/tok** | **12.5 tok/s** | **KV cache + overlapped AllReduce** |

> **Key finding:** A 300x improvement in network speed (WiFi → ethernet) yielded zero throughput improvement. GPU synchronization overhead — not network latency — is the dominant bottleneck. The network is not the problem.

12.5 tok/s streams faster than a human reads. The throughput is practical for interactive use, and the correctness result — identical output from mismatched GPUs over a commodity network — is what matters.

## System Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU (RTX 2000+ recommended)
- CUDA Toolkit 12.4+
- TensorRT 10.15+

## Quick Start

### 1. Download and Install

Download the latest installer from [GitHub Releases](https://github.com/baremetalrt/baremetalrt/releases/latest) and run it. The installer will check for NVIDIA prerequisites and guide you through setup.

### 2. Connect Your GPU

Sign in at [baremetalrt.ai/app](https://baremetalrt.ai/app) — the web app automatically detects the daemon running on your machine and links your GPU to your account.

### 3. Chat

Use the web interface at [baremetalrt.ai/app](https://baremetalrt.ai/app) or connect any OpenAI-compatible client:

```bash
curl https://baremetalrt.ai/v1/chat/completions \
  -H "Authorization: Bearer bmrt_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral-7b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Works with any OpenAI-compatible client — Python `openai` SDK, Continue, Cursor, or `curl`.

## CLI

```bash
pip install baremetalrt
bmrt status
bmrt models
bmrt run mistral-7b
```

## Technical Details

- **FP32 precision correctness** — custom CUDA kernel performs AllReduce on-GPU in FP32, achieving the theoretical floor of IEEE 754 arithmetic. 2,500x more accurate than FP16. Identical to NCCL on NVLink.
- **Asymmetric-tolerant transport** — GPUs with different VRAM and compute capabilities participate in the same AllReduce without barrier stalls. The slower GPU sets the pace; the faster GPU waits on a non-blocking receive.
- **TensorRT plugin integration** — custom `IPluginV2DynamicExt` plugins intercept every AllReduce/AllGather call at execution time, replacing NCCL with our TCP transport without modifying TRT-LLM's model definitions.
- **Overlapped AllReduce** — TCP recv runs in a background thread during GPU sync wait. When sync_wait > recv_time, the network transfer adds zero time to the critical path.
- **Double-buffered pinned memory** — four page-locked host buffers alternate between consecutive AllReduce calls, preventing data races between in-flight transfers.
- **TensorRT-LLM on Windows** — full native port of NVIDIA's inference engine (Conan profiles, FMHA kernels, nanobind bindings, MSVC/CUDA interop). No WSL, no Docker.

See [Architecture](docs/ARCHITECTURE.md) for the full system design, or read the [technical paper](paper/main.pdf).

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
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── QUICKSTART.md
│   └── MISSION.md
└── paper/         # Technical paper (arXiv-ready LaTeX)
```

The inference engine, transport layer, and daemon are in a separate private repository.

## API

BareMetalRT exposes an OpenAI-compatible API. See [API docs](https://baremetalrt.ai/docs) or the [API reference](docs/API.md).

## Current Status

**v0.5.1-beta** — [Changelog](CHANGELOG.md)

- Single-GPU and TP=2 multi-GPU inference on Windows
- Mistral 7B at 12.5 tok/s across heterogeneous GPUs over TCP
- Web chat UI with streaming
- Windows installer with automatic GPU claiming
- OpenAI-compatible API
- User accounts with Google OAuth + API key auth
- Chat history encrypted on-device — never stored on our servers

## Roadmap

- **TP=4+** — scale beyond two GPUs using ring AllReduce (transport implemented, untested beyond TP=2)
- **Mixture-of-Experts** — replace dense AllReduce with sparse expert routing. 2 of 8 experts per token = 75% of the mesh available for concurrent serving. See [Architecture § Future](docs/ARCHITECTURE.md#future-mixture-of-experts).
- **Distributed KV cache** — page KV entries to peer GPU VRAM across the mesh, enabling 128K+ context on consumer hardware
- **Asymmetric weight splitting** — proportional column assignment based on per-GPU VRAM
- **Continuous batching** — serve multiple users per forward pass

## Security

Found a vulnerability? See [SECURITY.md](SECURITY.md) for our responsible disclosure policy.

## License

BareMetalRT is proprietary software. See [LICENSE](LICENSE) for terms.
