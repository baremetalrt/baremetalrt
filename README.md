# BareMetalRT

**Distributed LLM inference across consumer GPUs on Windows.**

BareMetalRT takes NVIDIA's open-source TensorRT-LLM — the fastest inference engine on the planet — and brings it to the 200+ million NVIDIA GPUs running Windows that NVIDIA abandoned. Distributed across your gaming rigs, no data center required.

## What This Is

A fork of [TensorRT-LLM v1.2.0](https://github.com/NVIDIA/TensorRT-LLM) (Apache 2.0) with:

- **TCP transport layer** replacing NCCL (Linux-only) for distributed inference over Windows LAN
- **Distributed coordinator** that orchestrates inference across heterogeneous consumer GPUs
- **Windows-native build** targeting MSVC and consumer NVIDIA GPUs (RTX 30xx/40xx/50xx)
- **Chat UI** for interacting with your distributed cluster

## Architecture

```
┌─────────────────────────────────┐
│  Coordinator                    │
│  Node registry, layer sharding, │
│  health monitoring, OpenAI API  │
└──────────────┬──────────────────┘
               │ TCP
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Node A │ │Node B │ │Node C │
│3090   │ │4060   │ │3060   │
│24GB   │ │8GB    │ │12GB   │
│TRT-LLM│ │TRT-LLM│ │TRT-LLM│
│Engine  │ │Engine  │ │Engine  │
└───────┘ └───────┘ └───────┘
```

## Project Structure

```
baremetalrt/
├── engine/                 # C++ inference engine
│   ├── tensorrt-llm/       # TRT-LLM v1.2.0 submodule (NVIDIA, Apache 2.0)
│   ├── transport/          # TCP transport replacing NCCL
│   ├── windows/            # Windows build configuration
│   └── CMakeLists.txt      # Build system
├── coordinator/            # Distributed orchestration (Python)
│   └── api/                # OpenAI-compatible API server
├── chat-ui/                # Next.js chat interface
├── scripts/                # Build and utility scripts
└── docs/                   # Documentation
```

## Target Models

- **Mistral Nemo 12B** — Development/testing (single GPU)
- **Mistral Small 3.1 24B** — Pipeline parallel across 2-3 GPUs
- **Mixtral 8x22B** — Expert parallel, the showcase demo

All at 4-bit quantization on consumer hardware.

## Status

**v2 — In Development**

- [x] Chat UI (Next.js, multi-provider)
- [x] Single-node TensorRT-LLM inference (v1 legacy)
- [ ] TRT-LLM v1.2.0 Windows build
- [ ] TCP transport layer
- [ ] Distributed coordinator
- [ ] Multi-node inference demo

## Roadmap

### Phase 1: Single-Node Windows Build
Get TRT-LLM v1.2.0 compiling and running on Windows with MSVC.

- [ ] Build TRT-LLM C++ core with `ENABLE_MULTI_DEVICE=OFF` (strips NCCL/MPI)
- [ ] Resolve Windows-specific build issues (long paths, static linking, MSVC compat)
- [ ] Load and run Mistral Nemo 12B (GGUF/safetensors) on a single RTX GPU
- [ ] Benchmark: tokens/sec vs llama.cpp on same hardware
- [ ] Restore OpenAI-compatible API endpoint for single-node

**Milestone:** Single-GPU inference on Windows at TRT-LLM speed.

### Phase 2: TCP Transport Layer
Replace NCCL with TCP-based communication for Windows LAN.

- [ ] Implement `ITransport` interface (send, recv, allReduce, allGather, reduceScatter)
- [ ] Pinned memory staging: GPU → cudaMemcpyAsync → pinned host → TCP → peer
- [ ] CUDA IPC path for same-machine multi-GPU (bypass TCP)
- [ ] Ring all-reduce implementation over TCP
- [ ] Benchmark: transport overhead characterization (LAN latency, bandwidth)
- [ ] Wire transport into TRT-LLM plugin interface (match ncclPlugin/ signatures)

**Milestone:** Two GPUs on separate Windows machines running inference together.

### Phase 3: Distributed Coordinator
Build the orchestration layer — this is the product.

- [ ] Node agent: auto-registers with coordinator, reports GPU specs (VRAM, SM version, bandwidth)
- [ ] Layer assignment algorithm: optimal model sharding given heterogeneous GPU inventory
- [ ] Engine distribution: push TRT-LLM engine slices to nodes
- [ ] Health monitoring + heartbeat + fault detection
- [ ] KV cache budget manager (tracks aggregate context capacity across nodes)
- [ ] OpenAI-compatible API with streaming support
- [ ] Chat UI integration with coordinator

**Milestone:** 3-node cluster running Mistral Small 3.1 24B with automatic sharding.

### Phase 4: The Showcase
Demonstrate what no one else can do on consumer hardware.

- [ ] Mixtral 8x22B expert-parallel across 4+ GPUs
- [ ] 100K+ context window distributed across node memory
- [ ] Automated engine builder: `baremetalrt deploy mistral-small-3.1 --nodes node1,node2,node3`
- [ ] Performance dashboard: per-node utilization, throughput, latency breakdown
- [ ] One-click installer for Windows

**Milestone:** Mixtral 8x22B running across gaming rigs over LAN. The demo that sells itself.

## License

BareMetalRT is proprietary software. See [LICENSE](LICENSE) for terms.

The TensorRT-LLM submodule (`engine/tensorrt-llm/`) is licensed separately under NVIDIA's Apache License 2.0.
