# Requires: pip install bitsandbytes transformers accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
hf_token = "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"  # Your HuggingFace token

import sys

def main():
    print("Loading Mistral 7B Instruct in 8-bit mode. This may take a few minutes on first run...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        token=hf_token
    )

    prompt = "What are the main differences between Llama 2 and Mistral 7B?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    print("\n--- Model Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import atexit
import json
import os

STATUS_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'model_status.json')
MODEL_ID = "mistral_7b_instruct_8bit"

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
