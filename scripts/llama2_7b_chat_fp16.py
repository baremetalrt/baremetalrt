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

if __name__ == "__main__":
    main()
