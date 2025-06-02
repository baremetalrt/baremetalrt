#!/bin/bash
# Activate the TensorRT-LLM Python virtual environment
source ~/trtllm-venv/bin/activate

# Path to your HuggingFace model directory (AutoDeploy will handle engine loading)
MODEL_DIR="/home/brian/baremetalrt/models/Llama-3.1-8b-hf"

# Start the official TensorRT-LLM OpenAI-compatible API server (on port 8001)
python -m tensorrt_llm.llmapi.server \
  --model $MODEL_DIR \
  --backend trtllm \
  --port 8000
