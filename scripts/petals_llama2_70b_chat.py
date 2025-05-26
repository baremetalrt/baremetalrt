#!/usr/bin/env python
"""
Petals Llama 2 70B Chat Inference Script
Connects to the Petals distributed mesh and generates text using Llama 2 70B Chat (full precision).

Usage:
    python scripts/petals_llama2_70b_chat.py "Your prompt here"

Requirements:
    pip install petals transformers
"""
import sys
from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import os
import sys

MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"

# Try to read token from llama2_7b_chat_8int.py
hf_token = None
try:
    from importlib.util import spec_from_file_location, module_from_spec
    llama_path = os.path.join(os.path.dirname(__file__), "llama2_7b_chat_8int.py")
    spec = spec_from_file_location("llama2_7b_chat_8int", llama_path)
    llama_mod = module_from_spec(spec)
    spec.loader.exec_module(llama_mod)
    hf_token = getattr(llama_mod, "hf_token", None)
except Exception:
    pass

# Fallback to environment variable
if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    print("Error: HuggingFace token not found. Please set HF_TOKEN env var or define hf_token in llama2_7b_chat_8int.py.")
    sys.exit(1)

if len(sys.argv) > 1 and sys.argv[1] == "--check":
    try:
        print(f"Checking Petals mesh connectivity for {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
        model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token)
        # Try a dummy forward pass (no actual generation)
        _ = model.config
        print("ONLINE")
        sys.exit(0)
    except Exception as e:
        print(f"OFFLINE: {e}")
        sys.exit(1)
else:
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "Hello, world!"

    print(f"Connecting to Petals mesh for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoDistributedModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== Response ===\n")
    print(response)
