# Requires: pip install bitsandbytes transformers accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token = "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"  # Your HuggingFace token

import sys

def main():
    print("Loading tokenizer and model in 8-bit mode. This may take a few minutes on first run...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,  # Enable 8-bit quantization
        token=hf_token
    )

    prompt = "What's the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=64)
    print("\n--- Model Output ---\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    if '--check' in sys.argv:
        print('ONLINE')
        sys.exit(0)
    main()
