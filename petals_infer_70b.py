import sys
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-70b-chat-hf"
# Add your Hugging Face access token here
token = "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"

def main():
    model = AutoDistributedModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    # Prompt for demo: Unix sockets vs TCP sockets
    prompt = "What's the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

