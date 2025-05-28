#!/usr/bin/env python
"""
Llama-3 8B Chat Inference Script (4-bit)
Connects to a local quantized Llama-3 8B Chat model and generates text.

Usage:
    python scripts/llama3_8b_chat_4bit.py "Your prompt here"

Requirements:
    pip install transformers accelerate bitsandbytes
    (Make sure you have a 4-bit quantized checkpoint, e.g. TheBloke/Meta-Llama-3-8B-Instruct-GPTQ)
"""
import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "TheBloke/Meta-Llama-3-8B-Instruct-GPTQ"  # Change to your preferred Q4 checkpoint

# Try to read token from llama2_7b_chat_8int.py for consistency
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

if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
    print(f"""
Llama-3 8B Chat Inference Script (4-bit)
Usage:
    python {sys.argv[0]} [PROMPT]
    python {sys.argv[0]} --help
PROMPT: The prompt to send to the model (default: 'Hello, world!')
""")
    sys.exit(0)

if not hf_token:
    print("Error: HuggingFace token not found. Please set HF_TOKEN env var or define hf_token in llama2_7b_chat_8int.py.")
    sys.exit(1)

import time

def main():
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "Hello, world!"

    print(f"Loading Llama-3 8B Chat (4-bit) model: {MODEL_NAME} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, use_fast=True)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_token
    )
    t1 = time.time()
    print(f"Model loaded in {t1-t0:.2f}s")

    print(f"\nPrompt: {prompt}\nGenerating...")
    t2 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    t3 = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generation time: {t3-t2:.2f}s")

    print("\n=== Response ===\n")
    print(response)

if __name__ == "__main__":
    main()
