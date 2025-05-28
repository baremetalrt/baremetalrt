# Requires: pip install bitsandbytes transformers accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token = "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"  # Your HuggingFace token

import sys

def main():
    print("Loading tokenizer and model in 8-bit mode. This may take a few minutes on first run...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
        token=hf_token
    )

    # ---- Quantization and device verification ----
    print("\nVerifying quantized layer devices...")
    bnb_layers = [m for m in model.modules() if m.__class__.__name__.startswith("Linear8bit") or m.__class__.__name__.startswith("Linear4bit")]
    if bnb_layers:
        print(f"Found {len(bnb_layers)} bitsandbytes quantized linear layers.")
        print(f"First quantized layer device: {bnb_layers[0].weight.device}")
        devices = set([l.weight.device for l in bnb_layers])
        print(f"All quantized layer devices: {devices}")
    else:
        print("No bitsandbytes quantized layers found! (Are you sure quantization is enabled?)")
    print(f"Model main device: {model.device}")
    # --------------------------------------------

    prompt = "What's the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    print("\n--- Model Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import atexit
import json
import os

STATUS_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'model_status.json')
MODEL_ID = "llama2_7b_chat_8int"

def set_status(status):
    try:
        with open(STATUS_PATH, 'r') as f:
            data = json.load(f)
    except Exception:
        data = {}
    data[MODEL_ID] = status
    with open(STATUS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def mark_offline():
    set_status("offline")

set_status("online")
atexit.register(mark_offline)

if __name__ == "__main__":
    main()
