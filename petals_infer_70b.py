from petals import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-2-70b-chat-hf"
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "What are the main differences between Windows and Linux file locking mechanisms?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
