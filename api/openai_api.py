# FastAPI server for OpenAI-compatible Llama 2 inference
# Requirements: pip install fastapi uvicorn torch transformers
# Run with: uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
# Endpoint and schema compatible with OpenAI API (TensorRT-LLM style)
# See: https://platform.openai.com/docs/api-reference/completions/create

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import time

model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_token = os.environ.get("HF_TOKEN", "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")

# Load model and tokenizer at startup
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required, but was not detected. Please ensure you have a compatible NVIDIA GPU and the correct drivers/PyTorch installed.")
print("Loading Llama-2 7B Chat model (8-bit quantized, GPU required)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=hf_token
)
print(f"Model loaded on device: {model.device}")

app = FastAPI(title="OpenAI-Compatible LLM API", description="OpenAI-style local inference endpoint.")

@app.get("/health")
def health():
    """Health check endpoint."""
    device = str(model.device)
    return {"status": "ok", "model_loaded": True, "device": device}

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for all origins (less secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    """OpenAI-compatible completion endpoint."""
    import time as _time
    try:
        t0 = _time.time()
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        t1 = _time.time()
        print(f"[TIMING] Tokenization: {t1 - t0:.3f} seconds")
        with torch.no_grad():
            t2 = _time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p
            )
            t3 = _time.time()
            print(f"[TIMING] Model.generate: {t3 - t2:.3f} seconds")
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        t4 = _time.time()
        print(f"[TIMING] Decoding: {t4 - t3:.3f} seconds")
        print(f"[TIMING] Total: {t4 - t0:.3f} seconds")
        # Remove the prompt from the start of the completion if present
        prompt_text = request.prompt.strip()
        completion_text = completion.strip()
        if completion_text.startswith(prompt_text):
            answer = completion_text[len(prompt_text):].lstrip("\n ")
        else:
            answer = completion_text
        response = {
            "id": f"cmpl-{int(time.time()*1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "text": answer,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
