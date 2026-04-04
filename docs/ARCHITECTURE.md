# Architecture

BareMetalRT is a distributed inference engine that splits large language models across consumer NVIDIA GPUs over standard TCP networks. This document explains how the system works.

## Overview

```
┌─────────────────────────────────────────────┐
│              baremetalrt.ai                  │
│  Auth · Routing · OpenAI-compatible API      │
│  WebSocket relay to daemon nodes             │
└──────────────────┬──────────────────────────┘
                   │ WebSocket
        ┌──────────┼──────────┐
        │          │          │
   ┌────▼────┐ ┌───▼───┐ ┌───▼───┐
   │ Node A  │ │Node B │ │Node C │
   │ 4070S   │ │4060L  │ │3060   │
   │ 12 GB   │ │ 8 GB  │ │12 GB  │
   │ Daemon  │ │Daemon │ │Daemon │
   └────┬────┘ └───┬───┘ └───┬───┘
        │          │          │
        └──── TCP AllReduce ──┘
          (tensor parallelism)
```

The system has three layers:

1. **Server** — FastAPI backend at baremetalrt.ai handling auth, chat relay, and node management
2. **Daemon** — runs on each GPU machine, manages the TensorRT-LLM engine, connects to the server via WebSocket
3. **Transport** — custom TCP layer replacing NVIDIA's NCCL, enabling AllReduce across heterogeneous GPUs over commodity networks

## Why Tensor Parallelism (Not Pipeline)

Every prior distributed inference project on consumer hardware (Petals, Exo) chose **pipeline parallelism** — splitting the model into sequential stages. Pipeline parallelism is simple but wasteful: each GPU sits idle while the other computes, losing ~50% of available throughput.

BareMetalRT uses **tensor parallelism** — splitting every layer horizontally across all GPUs. Both GPUs compute simultaneously on every layer, synchronized by an AllReduce at each layer boundary. This keeps all GPUs active all the time.

The tradeoff: tensor parallelism requires 22-64 network round-trips per token (one per AllReduce point) versus 1-2 for pipeline parallelism. The conventional wisdom was that this makes TP infeasible over commodity networks. Our benchmarks show this is wrong — **GPU synchronization overhead, not network latency, is the dominant cost**. A 300x improvement in network speed (WiFi to ethernet) produced zero throughput improvement.

## Transport Layer

The transport replaces NCCL with a custom TCP implementation built on WinSock2. It solves two problems that blocked tensor parallelism on consumer hardware:

### 1. FP32 Precision Correctness

When partial results from different GPUs are summed during AllReduce, floating-point rounding introduces error. In FP16, this error is 2,500x larger than in FP32 and compounds through layers — we observed outright NaN at layer 23 in early FP16 experiments.

Our design runs the **entire computation path in FP32**: weights are stored in FP16 for memory efficiency, but every matmul, residual connection, normalization, and AllReduce executes in FP32. A custom CUDA kernel performs the reduction on-GPU — the math never touches the CPU. This achieves the **theoretical floor** of IEEE 754 single-precision arithmetic, identical to NCCL on NVLink.

### 2. Asymmetric Compute Tolerance

NCCL assumes all GPUs complete each layer at roughly the same time. When a 4070 Super finishes in 2ms and a 4060 Laptop takes 4ms, NCCL's synchronization breaks down.

Our transport doesn't assume symmetric timing. Each rank computes at its own speed, signals readiness, and the exchange happens when both sides are done. The slower GPU sets the pace, but the faster GPU doesn't stall the protocol — it waits on a non-blocking receive.

### Data Flow (per AllReduce call)

```
Rank 0 (4070 Super)              Rank 1 (4060 Laptop)
─────────────────────             ─────────────────────
① GPU computes partial            ① GPU computes partial
② D2H → pinned buffer (0.1ms)    ② D2H → pinned buffer (0.1ms)
③ TCP send ──────────────────→    ③ TCP send ──────────────────→
④ TCP recv ←──────────────────    ④ TCP recv ←──────────────────
⑤ GPU reduce in FP32 (0.3ms)     ⑤ GPU reduce in FP32 (0.3ms)
```

Send and receive happen simultaneously on dual TCP sockets (one for each direction). A background thread starts the recv before GPU compute finishes, hiding network latency behind the GPU synchronization wait.

### Pinned Memory

The transport pre-allocates page-locked (pinned) host buffers at init using `cudaHostAlloc`. Four pinned buffers in a double-buffered configuration prevent the current call's recv from overwriting the previous call's in-flight send data. A separate GPU scratch buffer handles in-place reduction.

## TensorRT Plugin Integration

TensorRT-LLM hardcodes all collective operations to NCCL. There is no plugin interface for swapping backends. We implemented custom TensorRT `IPluginV2DynamicExt` plugins in C++ — one for AllReduce, one for AllGather — that register under the same operation names TRT-LLM emits during engine compilation. The plugins intercept every collective call at execution time without modifying TRT-LLM's model definitions, graph construction, or weight loading.

## TensorRT-LLM on Windows

NVIDIA discontinued Windows support for TensorRT-LLM. We restored it (v0.12.0), patching four layers:

- **Package management** — Conan dependency profiles rewritten for Windows toolchains
- **FMHA kernels** — fused multi-head attention ported from GCC intrinsics to MSVC
- **Python bindings** — nanobind CMake targets patched for Windows DLL generation
- **MSVC/CUDA interop** — ABI and linking fixes across the toolchain boundary

The result: native Windows execution of NVIDIA's 1,500+ hand-tuned CUDA kernels, no WSL or Docker required.

## GPU Claiming (Device Linking)

When a user signs into baremetalrt.ai/app, the web app probes `localhost:8080` and `localhost:9000` for a running daemon. If found, a three-step handshake links the GPU to the user's account:

1. Web app fetches a claim token from the local daemon
2. Web app sends the claim token to the server, which validates and creates the association
3. Server pushes an API key back to the daemon

This is automatic — no user action beyond signing in.

## Server Architecture

The FastAPI server handles:

- **Auth** — Google OAuth + JWT sessions, API key management (`bmrt_` prefixed)
- **Chat relay** — WebSocket bridge between web clients and daemon nodes
- **Node management** — daemon registration, health monitoring, session matching
- **OpenAI-compatible API** — `POST /v1/chat/completions` with SSE streaming

Backed by Postgres on a DigitalOcean droplet, fronted by nginx.

## Optimization History

Three stages of optimization on Mistral 7B (32 layers, 14 GB, TP=2):

| Stage | Latency | Throughput | What Changed |
|-------|---------|------------|--------------|
| Synchronous AllReduce | 293 ms/tok | 3.4 tok/s | All stages sequential, recv blocks critical path |
| Overlapped AllReduce | 185 ms/tok | 5.4 tok/s | Recv runs in background thread during GPU sync wait |
| + KV cache | **80 ms/tok** | **12.5 tok/s** | Payload shrinks from 49 KB to 8 KB per call |

The remaining bottleneck is the GPU synchronization wait itself (~0.7 ms/call) — the irreducible cost of the model's arithmetic. Larger models improve the ratio: more compute per layer means more sync wait to hide communication behind.

## Future: Mixture-of-Experts

Dense transformers are the worst case for distributed inference — every layer requires AllReduce across all ranks. MoE architectures (like Mixtral) activate only 2 of 8 experts per token, replacing dense AllReduce with sparse point-to-point routing:

- **Expert parallelism across the internet** — route tokens to the right expert node (one TCP hop, latency-tolerant)
- **Tensor parallelism within experts** — AllReduce over LAN for experts too large for one GPU
- **75% of nodes idle per token** — available to serve other users concurrently
- **Fault tolerance** — replicate experts across nodes, re-route on failure

Data centers are overprovisioned for MoE: a $3M GB200 NVL72 rack has 75% of its 13.8 TB VRAM idle per token. A mesh of consumer GPUs inverts the cost structure — idle VRAM costs nearly nothing.
