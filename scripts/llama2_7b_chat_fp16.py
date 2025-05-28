# Most impressive local inference demo for Llama-2 7B Chat on a 12GB VRAM GPU
# Requirements: pip install torch transformers accelerate
# (Optional but recommended: run with CUDA, i.e., on an NVIDIA GPU)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token = "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"  # Your HuggingFace token


def main():
    print("Loading Llama-2 7B Chat in float16 on GPU (if available)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        token=hf_token
    )
    print(f"Model loaded on device: {model.device}")

    prompt = "You are a helpful AI assistant.\nUser: What's the capital of France?\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,  # More natural output
            temperature=0.7,
            top_p=0.95
        )
    print("\n--- Model Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

import atexit
import json
import os

STATUS_PATH = os.path.join(os.path.dirname(__file__), '..', 'api', 'model_status.json')
MODEL_ID = "llama2_7b_chat_fp16"

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
