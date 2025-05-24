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

## Roadmap & Checkpoints

### 1. Project Setup
- ✅ 1.1 Fork/clone the Petals repo into `external/petals-main`
- ⬜ 1.2 Create and activate a Python virtual environment (Windows)
- ⬜ 1.3 Install base dependencies (`requirements.txt`)
- ⬜ 1.4 Identify and document any Linux-only dependencies (e.g., bitsandbytes)
- ⬜ 1.5 Patch or stub out incompatible dependencies

### 2. Model Selection & Preparation
- ✅ 2.1 Select initial model: **Llama 70B**
- ⬜ 2.2 Document model requirements (RAM, VRAM, disk space)
- ⬜ 2.3 Download or convert Llama 70B weights to HuggingFace format (if needed)
- ⬜ 2.4 Test model loading on Windows (standalone PyTorch/Transformers)

### 3. Server/Node Core Logic
- ⬜ 3.1 Review `petals/cli/run_server.py` and `petals/server/`
- ⬜ 3.2 Patch server logic as needed for Windows compatibility
- ⬜ 3.3 Ensure CLI for launching a node works on Windows
- ⬜ 3.4 Stub/patch DHT and networking for Windows support
- ⬜ 3.5 Validate local inference (single node)

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
- ⬜ 8.2 Document all patches and stubs in this `agent.md`
- ⬜ 8.3 Provide troubleshooting and FAQ section

---

## Known Issues / Todos
- [ ] bitsandbytes (quantization) is Linux-only—stub or replace for Windows
- [ ] CUDA toolkit and driver setup for Windows
- [ ] DHT networking compatibility (test on Windows)
- [ ] Large model (Llama 70B) resource requirements—document and test

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
