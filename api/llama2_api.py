# FastAPI server for local Llama-2 7B Chat inference
# Requirements: pip install fastapi uvicorn torch transformers
# Run with: uvicorn api.llama2_api:app --host 0.0.0.0 --port 8000

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token = os.environ.get("HF_TOKEN", "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")

# Load model and tokenizer at startup
print("Loading Llama-2 7B Chat model (float16, GPU if available)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=hf_token
)
print(f"Model loaded on device: {model.device}")

app = FastAPI(title="Llama-2 7B Chat API", description="OpenAI-style local inference endpoint.")

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/v1/completions")
def chat(request: ChatRequest):
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p
            )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"completion": completion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
