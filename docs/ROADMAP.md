# Roadmap & Progress

## 🏁 Major Milestones

- ✅ Petals/Hivemind compiles and installs on Windows (distributed inference not functional)
- ✅ BareMetalRT CLI validated for local inference
- ✅ FastAPI REST API operational for local inference
- ✅ Model loading and quantized inference validated (Llama 2 7B, Mistral 7B, GPT-2)
- ⏸️ Distributed inference (Petals/Hivemind) deferred pending DHT/network compatibility

## 1. Local Inference

- 1.1 ✅ Validate Hugging Face Transformers (PyTorch) backend for local inference (Llama 2 7B, Mistral 7B, GPT-2)
- 1.2 ✅ Validate BareMetalRT CLI for local inference (TensorRT-LLM backend)
- 1.3 ⬜ Add quantized/fp16 model support for both backends
- 1.4 ⬜ Document hardware requirements and performance for each backend
- 1.5 ✅ Integrate local inference with FastAPI REST API (OpenAI-compatible, robustly tested)

## 2. Node Architecture & Logic

- 2.1 ✅ Modularize node/server architecture for backend selection (Petals, BareMetalRT, future backends)
- 2.2.1 ✅ Refactor node logic for single-node/local inference
- 2.2.2 ⏸️ Refactor node logic for distributed inference (deferred)
- 2.3 ⬜ Integrate peer discovery and job routing layer (Hivemind, p2pd, ZeroMQ research)
- 2.4 ⬜ Document and test node startup, backend selection, and routing logic
- 2.5 ⬜ Ensure all patches/stubs for Windows compatibility are tracked and reproducible

## 3. Distributed Inference (Deferred)

- 3.1 ⏸️ Test multi-node setup (multiple Windows machines or processes) *(Deferred: Windows DHT/distributed support is currently on hold due to compatibility blockers)*
- 3.2 ⏸️ Validate DHT peer discovery and layer assignment
- 3.3 ⏸️ Confirm end-to-end distributed inference (prompt → output)

## 4. Web Integration

- 4.1 ✅ Add a REST inference API server (FastAPI) for single-node/local inference
- 4.2 ✅ Define and document OpenAI-compatible endpoint(s) (`/v1/completions`) for prompt submission and completion retrieval (TensorRT-LLM style)
- 4.3 ⬜ Integrate Chatbot UI frontend with the OpenAI-compatible inference API (`openai_api`) (MVP: single-user, no RAG, no multi-user)
- 4.4 ⬜ Test end-to-end integration (web UI → API → model → UI)
- 4.5 ⬜ Implement authentication and CORS for API security
- 4.6 ⬜ Document setup and usage in README/API.md


## 5. Lightweight Optimization

- 5.1 ⬜ Remove or stub unnecessary features for MVP (e.g., metrics, advanced monitoring, Docker)
- 5.2 ⬜ Optimize dependency list for minimal install
- 5.3 ⬜ Document resource usage and performance

## 6. Testing & Validation

- 6.1 ⬜ Write/check basic test cases for single-node and multi-node inference
- 6.2 ⬜ Validate outputs match expectations for Llama 70B
- 6.3 ⬜ Document known issues and workarounds

## 7. Documentation & Packaging

- 7.1 ⬜ Update `README.md` with Windows-specific setup and usage
- 7.2 ⬜ Document all patches and stubs in this `agent.md`
- 7.3 ⬜ Provide troubleshooting and FAQ section
- 7.4 ⬜ Keep `windows_patches.md` up to date with every technical decision and patch
