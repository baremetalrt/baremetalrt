# BareMetalRT Bittorrent style Distributed MVP

## Overview
This document provides a high-level overview, technical decisions, and system architecture for a Bittorrent-style inference job board/marketplace.

## MVP Goals

- Windows client with distributed inference. Run Petals style servers on Windows (single and multi-node distributed inference)
- Maximize code reuse from upstream Petals
- Minimize dependencies and keep the solution lightweight
- Patch or stub only what is needed for Windows compatibility
- Multi backend support (BareMetalRT, Hugging Face Transformers, PyTorch)
- Web UI for basic inference (single modal) requests
- Web UI for distributed/single node inference job board/marketplace
- Run distributed inference on Llama 70B

## Development Principles

- **Keep all code minimal and ultra-lightweight.**
- **Avoid unnecessary dependencies**—only add what is absolutely required for MVP functionality.
- **Patch or stub only what is needed for Windows compatibility.**
- **Maximize code reuse** from upstream Petals and existing open-source projects.
- **Document every patch, stub, or technical deviation** from upstream or standard approaches.
- **Prioritize Windows compatibility** in all design and implementation decisions.
- **Do not over-engineer**; focus on a working, maintainable MVP above all else.

## System Architecture & Tech Stack

This project is composed of modular components:
- **Inference Backends:** Petals (distributed, PyTorch), proprietary custome backend BareMetalRT (native C++)
- **Peer Discovery & Job Routing:** Hivemind, p2pd (currently incompatible with Windows due to DHT sockets); ZeroMQ (Actively researching)
- **API Layer:** FastAPI (Python REST, OpenAI-compatible endpoints via `openai_api`), planned gRPC
- **Web UI:** Chatbot UI (React/Next.js) for prompt submission and response display (OpenAI-compatible, single-user, no RAG/multi-user for MVP)
    - **Note:** Supabase integration is not required for the MVP. All Supabase-related code is commented out in the frontend. No Supabase keys or configuration are needed for local or production use unless explicitly re-enabled in future updates.
- **Frontend Dependency:** Node.js (v24.1 or later) is required to build and run the Chatbot UI frontend.

## Inference Backends

### Hugging Face Transformers (PyTorch)
- Fully functional and validated for Llama 2 7B (12GB GPU), Mistral 7B, GPT-2, etc.
- PyTorch/Transformers backend, with quantized (8-bit) and float16 support.
- Local inference via FastAPI server.

### Petals (Distributed, PyTorch)
- Distributed inference backend, PyTorch-based.
- Relies on peer/job routing for multi-node coordination.
- Patched for Windows support (compiles/installs, distributed inference not functional).
- All modifications and Windows patches are tracked for compliance and reproducibility.

### BareMetalRT (Native C++/TensorRT-LLM)
- Ultra-lightweight C++17 CLI, targets TensorRT-LLM 0.10.0 `.engine` files and custom CUDA/C++ plugins.
- No Python/container dependencies. ONNX fallback and vLLM runtime planned.
- Llama 2/3 support blocked pending plugin development. pure TensorRT path Not actively developed for latest Llama models yet. TensoRT-LLM-0.10.0 path in active development.
- Intended for maximum performance and minimal overhead on Windows.

## Peer Discovery & Job Routing

- **Purpose:** Node discovery, job assignment, coordination.
- **Current Implementation:** Hivemind, p2pd (blocked due to DHT sockets compatibility for windows). Requires rewrite for Windows.
- **Active Research:** ZeroMQ for lightweight, cross-platform compatibility.

## API & Integration

- **FastAPI (Python):** RESTful API for both Petals and BareMetalRT backends (developer-friendly, rapid integration). Now exposes `/v1/completions` and other endpoints compatible with OpenAI API (TensorRT-LLM style) via `openai_api`.
- **Planned:** gRPC-based protocol for distributed job routing; ZeroMQ as a lightweight alternative.
- **Web UI:** Planned for prompt submission and job monitoring.

## Supported Models & Inference Paths

### Single-Node / Local Inference (PyTorch/Transformers)
- **Purpose:** Run Llama 2 7B/13B, Mistral 7B, GPT-2, etc. on a single machine (Windows, CUDA GPU recommended).
- **Backend:** HuggingFace Transformers (PyTorch), optionally quantized (float16/8bit)
- **Status:** Complete for Llama 2 7B Chat (12GB GPU)
- **Usage:** Ideal for demos, local/private inference, and developer validation.

### Distributed Inference (Petals/Hivemind)
- **Purpose:** Run very large models (e.g., Llama 70B) across many machines, pooling VRAM/resources.
- **Backend:** Petals (PyTorch) + Hivemind DHT (peer-to-peer layer assignment)
- **Status:** Deferred (Windows DHT compatibility blockers)
- **Usage:** Enables scaling beyond single-GPU/host limits, supports public/private networks.

---

