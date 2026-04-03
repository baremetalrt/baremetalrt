# BareMetalRT — Mission

## The Language Model Delivery Network

**Every NVIDIA GPU is an inference node. Every gaming rig, laptop, and workstation becomes part of a global compute layer. No datacenter required.**

BareMetalRT is a distributed inference engine built on TensorRT-LLM's CUDA kernels — the fastest open-source inference runtime on NVIDIA hardware. We replaced the Linux-only communication stack with a TCP transport layer that runs natively on Windows, turning consumer GPUs into a coordinated inference cluster.

### The Problem

AI inference is centralized by design. Cloud providers charge per-token, own the hardware, and sit between every user and every model. Latency is bounded by geography. Cost is bounded by margin. Privacy is bounded by trust.

Meanwhile, hundreds of millions of NVIDIA GPUs sit idle — in gaming rigs, on desks, in dorm rooms. The world's largest distributed supercomputer already exists. It just doesn't know it yet.

### The Vision

**LMDN — the Language Model Delivery Network.**

Agents and applications route inference to nearby hardware with models hot-loaded in VRAM. Execution is local. Response is instant. The cloud is bypassed entirely.

A gamer in Jakarta. A laptop in a coworking space. A doctor in Nairobi. All become part of the compute layer.

- **Bare metal fast** — TensorRT-LLM CUDA kernels, no Python overhead, full VRAM control
- **Distributed by default** — models split across GPUs over LAN or WAN, smart layer sharding based on available hardware
- **Hardware-agnostic** — RTX 3060 to 4090, heterogeneous clusters, consumer hardware
- **Light enough to coexist** — runs alongside games, creative tools, or background agents
- **BYOM** — bring your own model, any open-weight architecture, 4-bit quantized

If done right, LMDN delivers a theoretical floor for cost and latency in AI compute — orders of magnitude cheaper and faster than centralized infrastructure.

### Why Now

NVIDIA open-sourced TensorRT-LLM under Apache 2.0 — then abandoned Windows support to protect their datacenter business. We took their own engine and pointed it at the platform they left behind.

Open-weight models now match or exceed proprietary models at the 12B-70B scale. Mistral, Qwen, and Gemma run at production quality on consumer hardware. The model gap has closed. The infrastructure gap is the last barrier.

### The Technical Edge

- **TensorRT-LLM CUDA kernels** — 1,500+ optimized .cu files for attention, GEMM, quantization. Nothing is faster on NVIDIA silicon.
- **TCP transport replacing NCCL** — distributed inference over standard networking, no NVLink required, Windows-native
- **Coordinator** — intelligent layer sharding across heterogeneous GPUs, fault tolerance, KV cache distribution
- **Protocol-level encryption** — nodes serve models without exposing weights

### What We're Building

Phase 1: Single-node TRT-LLM on Windows — prove the speed.
Phase 2: TCP transport — two machines, one model, distributed inference.
Phase 3: Coordinator — automatic sharding, health monitoring, OpenAI-compatible API.
Phase 4: LMDN — the network. Consumer GPUs worldwide, inference at the edge, the cloud melts.
