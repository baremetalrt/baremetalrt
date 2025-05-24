# (Moved from project root)
# Petals distributed inference experiment script for Llama-2 70B Chat
# If you want to run this, ensure you have Petals and all dependencies installed.

from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-70b-chat-hf"
token = "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"

def main():
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    prompt = "What's the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
