# FastAPI server for OpenAI-compatible Llama 2 inference
# Requirements: pip install fastapi uvicorn torch transformers
# Run with: uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
# Endpoint and schema compatible with OpenAI API (TensorRT-LLM style)
# See: https://platform.openai.com/docs/api-reference/completions/create

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

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

app = FastAPI(title="OpenAI-Compatible LLM API", description="OpenAI-style local inference endpoint.")

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    """OpenAI-compatible completion endpoint."""
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p
            )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": completion,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
