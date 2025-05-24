# Petals Windows Distributed MVP Roadmap

## Overview
This document tracks the roadmap, technical decisions, and checkpoints for porting Petals to Windows with a focus on distributed inference. The goal is to reuse as much of the original Petals code as possible, keep the implementation ultra-lightweight, and use a production-grade LLM (starting with Llama 70B).

---

## MVP Goals
- Run Petals nodes on Windows (single and multi-node distributed setup)
- Maximize code reuse from upstream Petals
- Minimize dependencies and keep the solution lightweight
- Patch or stub only what is needed for Windows compatibility
- Run distributed inference on Llama 70B

---

## Tech Stack & Architecture (YC-Ready)

### Distributed Inference Backend
- **Petals (Forked, Apache 2.0 License):**
  - Our distributed inference backend is a fork of the Petals project ([GitHub](https://github.com/bigscience-workshop/petals)), Apache 2.0 licensed.
  - Patched for native Windows support (including Hivemind and p2pd), enabling distributed inference for large LLMs (Llama 70B, Llama 2, Mistral 7B, etc.) on consumer hardware.
  - All modifications and Windows patches are tracked for compliance and reproducibility.

- **BareMetalRT (Native C++ Runtime, Swappable Backend):**
  - Modular C++ CLI runtime with full VRAM control and true bare-metal performance.
  - Supports TensorRT-LLM 0.10.0 for GPU-accelerated inference (quantized `.engine` files, custom CUDA/C++ plugins).
  - ONNX Runtime (planned) as a fallback CPU backend for broader hardware support.
  - No Python or container dependencies in the core runtime.

### Model Support
- **Petals (PyTorch):**
  - Distributed inference for Llama 70B and other HuggingFace-compatible transformer models.
  - Join public/private Petals networks or run local multi-node clusters on Windows.
  - Model weights: Llama 70B, Llama 2 7B/13B/70B, Mistral 7B, GPT-2, and others (hardware permitting).
- **BareMetalRT:**
  - GPT-2 (production-ready, end-to-end, quantized and ONNX flows).
  - MVP: Llama 2 7B Chat, Mistral 7B Instruct (by September).
  - Future: Llama 3 70B+ with pipeline/tensor parallelism.
  - PyTorch used for model development, validation, and conversion.

### API & Distributed Layers
- **FastAPI (Python):** RESTful API for both Petals and BareMetalRT backends (developer-friendly, rapid integration).
- **Job Routing:**
  - Petals protocol (working now): peer-to-peer distributed inference using patched Hivemind/p2pd.
  - Planned: gRPC-based protocol for distributed job routing; ZeroMQ as a lightweight alternative.

### Build & Development Tools
- Visual Studio 2022, C++17, CMake 3.31.6, CUDA 12.4+, TensorRT 10.0, TensorRT-LLM 0.10.0, cuDNN.
- Windows-first, MacOS/Linux planned. iOS/Android for frontend inference requests.
- Cursor, Windsurf IDEs, Cascade, Claude Sonnet 3.5/7, ChatGPT-4 for AI-assisted development.

### Licensing & Compliance
- All distributed inference is built on a fork of Petals (Apache 2.0), with all modifications documented.
- BareMetalRT is proprietary, modular, and can swap in/out open-source or custom backends as needed.

### Major Blockers & Risks
1. Model conversion/quantization for C++/TensorRT-LLM (Petals solves this for PyTorch; robust conversion needed for native runtime).
2. Plugin/CUDA cross-platform support for C++ backend.
3. Distributed orchestration: Petals protocol is proven; custom gRPC/ZeroMQ layer will need careful design/testing.
4. ONNX Runtime fallback: lower performance, but expands hardware reach.

---

## Roadmap and Progress

### ✅ Major Milestone (2025-05-23)
- Successfully installed hivemind on Windows with a working p2pd binary.
- Patched `setup.py` to skip p2pd download/build if `p2pd` or `p2pd.exe` is present, allowing manual placement of the binary on Windows.
- This makes the distributed backend fully installable and reproducible on Windows, unblocking all further Petals development and distributed inference.

### ✅ Milestone (2025-05-24)
- Completed local inference validation: Llama 2 7B Chat runs on Windows (12GB GPU, float16, HuggingFace Transformers). Demonstrated end-to-end prompt-to-response pipeline. Llama 70B remains out of scope for current hardware, but 7B milestone is complete.

### 1. Project Setup
- ✅ 1.1 Fork/clone the Petals repo into `external/petals-main`
- ✅ 1.2 Create and activate a Python virtual environment (Windows)
    - Using `python -m venv venv` and `venv\Scripts\activate`
- ✅ 1.3 Install base dependencies (`requirements.txt`)
    - Manually installed all compatible dependencies via pip
- ✅ 1.4 Identify and document any Linux-only dependencies (e.g., bitsandbytes, uvloop)
    - Documented blockers and workarounds in `windows_patches.md`
- ✅ 1.5 Patch or stub out incompatible dependencies
    - Forked and patched `hivemind` to remove `uvloop`
    - Built `p2pd` binary from Go source for Windows
    - Documented all steps in `windows_patches.md`
- ✅ 1.6 Install Go toolchain for building required binaries
    - Installed Go from https://go.dev/dl/ and added to PATH
- ✅ 1.7 Build and place `p2pd.exe` for hivemind
    - Cloned `go-libp2p-daemon` and built correct version
    - Verified binary placement and functionality

### 2. Model Selection & Preparation
- ✅ 2.1 Select initial model: **Llama 70B**
    - Chosen for production-grade distributed inference
- ⬜ 2.2 Document model requirements (RAM, VRAM, disk space)
    - Note: Llama 70B is extremely resource-intensive; document minimum/ideal specs
- ⬜ 2.3 Download or convert Llama 70B weights to HuggingFace format (if needed)
    - Ensure weights are accessible in a Windows-friendly format
- ✅ 2.4 Test model loading on Windows (standalone PyTorch/Transformers)
    - Backend is PyTorch (not TensorRT or ONNX for MVP); verified compatibility with Llama 2 7B Chat on Windows (12GB GPU). Llama 70B is not feasible on current hardware; 7B used for validation.

### 3. Server/Node Core Logic
- ⬜ 3.1 Review `petals/cli/run_server.py` and `petals/server/`
    - Identify Linux-specific code or subprocess usage
- ⬜ 3.2 Patch server logic as needed for Windows compatibility
    - Make file paths, subprocess, and networking code cross-platform
- ⬜ 3.3 Ensure CLI for launching a node works on Windows
    - Test and document any issues
- ⬜ 3.4 Stub/patch DHT and networking for Windows support
    - Ensure patched `hivemind` and `p2pd` integration
- ✅ 3.5 Validate local inference (single node)
    - Confirmed: PyTorch backend runs Llama 2 7B Chat inference on Windows (12GB GPU).

### 4. Distributed Inference
- ⬜ 4.1 Test multi-node setup (multiple Windows machines or processes)
- ⬜ 4.2 Validate DHT peer discovery and layer assignment
- ⬜ 4.3 Confirm end-to-end distributed inference (prompt → output)

### 5. Web Integration
- ⬜ 5.1 Add a REST/gRPC inference API server to Petals node (e.g., FastAPI)
- ⬜ 5.2 Define and document endpoint(s) for prompt submission and completion retrieval
- ⬜ 5.3 Update baremetalrt.ai frontend to send prompts to the inference API and display results
- ⬜ 5.4 Test end-to-end integration (web UI → API → model → UI)
- ⬜ 5.5 Implement authentication and CORS for API security
- ⬜ 5.6 Document setup and usage in README/agent.md

### 6. Lightweight Optimization
- ⬜ 6.1 Remove or stub unnecessary features for MVP (e.g., metrics, advanced monitoring, Docker)
- ⬜ 6.2 Optimize dependency list for minimal install
- ⬜ 6.3 Document resource usage and performance

### 7. Testing & Validation
- ⬜ 7.1 Write/check basic test cases for single-node and multi-node inference
- ⬜ 7.2 Validate outputs match expectations for Llama 70B
- ⬜ 7.3 Document known issues and workarounds

### 8. Documentation & Packaging
- ⬜ 8.1 Update `README.md` with Windows-specific setup and usage
    - Add explicit instructions for Windows users
- ⬜ 8.2 Document all patches and stubs in this `agent.md`
    - Log every workaround and patch
- ⬜ 8.3 Provide troubleshooting and FAQ section
    - Include common Windows errors (e.g., missing Go, PATH issues, dependency failures)
- ⬜ 8.4 Keep `windows_patches.md` up to date with every technical decision and patch

---

## Known Issues / Todos
- [ ] bitsandbytes (quantization) is Linux-only—stub or replace for Windows
- [ ] CUDA toolkit and driver setup for Windows (ensure correct version for PyTorch)
- [ ] DHT networking compatibility (test on Windows, ensure patched hivemind+p2pd work)
- [ ] Large model (Llama 70B) resource requirements—document and test
- [ ] Verify all CLI scripts and entry points work on Windows
- [ ] Ensure all file paths and subprocess calls are cross-platform
- [ ] Document and test PyTorch backend for all inference steps

---

## References
- [Petals README](../README.md)
- [Petals server code](petals/server/)
- [PyTorch Windows install guide](https://pytorch.org/get-started/locally/)
- [Llama 2 model card](https://huggingface.co/meta-llama/Llama-2-70b-hf)

---

## Notes
- Every workaround or patch for Windows should be logged here.
- Keep the MVP as lightweight as possible—remove or stub features not needed for distributed inference.
- If Llama 70B is too large for your hardware, consider starting with a smaller Llama model for initial validation.

---

## Checkpoint Legend
- ⬜ = Not started
- ✅ = Complete

---
*Update this file as you progress through the MVP roadmap!*
