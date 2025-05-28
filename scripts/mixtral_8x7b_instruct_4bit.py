# Requires: pip install bitsandbytes transformers accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
hf_token = "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"  # Your HuggingFace token

import sys

def main():
    print("Loading Mixtral 8x7B Instruct in 4-bit mode. This may take a few minutes on first run...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_token
    )
    prompt = "You are a helpful assistant. What are the main advantages of Mixture of Experts models like Mixtral 8x7B?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    print("\n--- Model Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import atexit
import json
import os

STATUS_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'model_status.json')
MODEL_ID = "mixtral_8x7b_instruct_4bit"

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
