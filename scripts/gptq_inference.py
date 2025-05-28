import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
hf_token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
