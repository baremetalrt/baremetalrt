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

import gc
from threading import Lock
# try:
#     from petals import AutoDistributedModelForCausalLM
#     PETALS_AVAILABLE = True
# except ImportError:
#     PETALS_AVAILABLE = False

model = None
model_name = None
tokenizer = None
# petals_model = None
# petals_tokenizer = None
current_model_id = None
model_lock = Lock()
hf_token = os.environ.get("HF_TOKEN", "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")

app = FastAPI(title="OpenAI-Compatible LLM API", description="OpenAI-style local inference endpoint.")

from . import model_manager
app.include_router(model_manager.router, prefix="/api")

def unload_model():
    global model, tokenizer, petals_model, petals_tokenizer
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        model = None
        tokenizer = None
    # if petals_model is not None:
    #     del petals_model
    #     del petals_tokenizer
    #     gc.collect()
    #     petals_model = None
    #     petals_tokenizer = None

def update_model_status(model_id, status):
    import os, json
    base_dir = os.path.dirname(__file__)
    status_path = os.path.abspath(os.path.join(base_dir, "model_status.json"))
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            status_dict = json.load(f)
    else:
        status_dict = {}
    # Set all models offline, then set the current one online
    for k in status_dict:
        status_dict[k] = "offline"
    status_dict[model_id] = status
    with open(status_path, "w") as f:
        json.dump(status_dict, f, indent=2)

def load_model(model_id):
    global model, model_name, tokenizer, current_model_id
    # petals_model, petals_tokenizer
    unload_model()
    if model_id == "llama2_7b_chat_8int":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
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
        petals_model = None
        petals_tokenizer = None
    elif model_id == "llama3_8b_chat_4bit":
        model_name = "astronomer/Llama-3-8B-Instruct-GPTQ-4-Bit"
        print("Loading Llama-3 8B Chat model (4-bit quantized, GPU required)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=hf_token
        )
        print(f"Model loaded on device: {model.device}")
        petals_model = None
        petals_tokenizer = None
    elif model_id == "mistral_7b_instruct_8bit":
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print("Loading Mistral 7B Instruct model (8-bit quantized, GPU required)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            token=hf_token
        )
        print(f"Model loaded on device: {model.device}")
        petals_model = None
        petals_tokenizer = None
    elif model_id == "llama2_13b_chat_4bit":
        model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
        print("Loading Llama-2 13B Chat model (4-bit GPTQ, GPU required)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            token=hf_token
        )
        print(f"Model loaded on device: {model.device}")
        petals_model = None
        petals_tokenizer = None
    elif model_id == "llama2_70b_chat_petals":
        if not PETALS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Petals is not installed. Please install the 'petals' package to use the 70B model.")
        petals_model_name = "meta-llama/Llama-2-70b-chat-hf"
        print("Connecting to Petals mesh for Llama 2 70B Chat...")
        petals_tokenizer = AutoTokenizer.from_pretrained(petals_model_name, token=hf_token)
        petals_model = AutoDistributedModelForCausalLM.from_pretrained(petals_model_name, token=hf_token)
        print("Connected to Petals mesh.")
        model = None
        tokenizer = None
        model_name = petals_model_name
    elif model_id == "deepseek_llm_7b_chat_4bit":
        model_name = "TheBloke/deepseek-llm-7b-chat-GPTQ"
        print("Loading Deepseek LLM 7B Chat model (4-bit GPTQ, GPU required)...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token
        )
        print(f"Model loaded on device: {model.device}")
        petals_model = None
        petals_tokenizer = None
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    current_model_id = model_id
    update_model_status(model_id, "online")

# Load Llama 2 7B Chat (8-bit quantized) model by default at startup
load_model("llama2_7b_chat_8int")

@app.post("/api/switch_model")
def switch_model(request: dict):
    """Switch the active model for inference."""
    model_id = request.get("model_id")
    with model_lock:
        try:
            load_model(model_id)
            return {"status": "ok", "active_model": model_id}
        except Exception as e:
            unload_model()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """Health check endpoint."""
    with model_lock:
        if current_model_id == "llama2_7b_chat_8int" and model is not None:
            device = str(model.device)
            return {"status": "ok", "model_loaded": True, "device": device, "active_model": current_model_id}
        # elif current_model_id == "llama2_70b_chat_petals" and petals_model is not None:
        #     return {"status": "ok", "model_loaded": True, "device": "petals_mesh", "active_model": current_model_id}
        else:
            return {"status": "error", "model_loaded": False, "active_model": current_model_id}


from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://picture-pockets-herald-toys.trycloudflare.com",
        "https://baremetalrt.ai"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    """OpenAI-compatible completion endpoint."""
    import time as _time
    with model_lock:
        allowed_models = ["llama2_7b_chat_8int", "deepseek_llm_7b_chat_4bit"]
        if model is not None and current_model_id in allowed_models:
            try:
                t0 = _time.time()
                inputs = tokenizer(request.prompt, return_tensors="pt")
                # Move tensors to the correct device (for quantized models, do NOT use .to() on model)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                t1 = _time.time()
                print(f"[TIMING] Tokenization: {t1 - t0:.3f} seconds")
                with torch.no_grad():
                    t2 = _time.time()
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=True
                    )
                    t3 = _time.time()
                    print(f"[TIMING] Generation: {t3 - t2:.3f} seconds")
                    # Only decode new tokens (not the prompt)
                    prompt_len = inputs["input_ids"].shape[-1]
                    generated_tokens = outputs[0][prompt_len:]
                    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    t4 = _time.time()
                    print(f"[TIMING] Decoding: {t4 - t3:.3f} seconds")
                response = {
                    "id": f"cmpl-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(_time.time()),
                    "model": current_model_id,
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
        elif current_model_id == "llama2_70b_chat_petals" and petals_model is not None:
            if not PETALS_AVAILABLE:
                raise HTTPException(status_code=503, detail="Petals is not installed. Please install the 'petals' package to use the 70B model.")
            try:
                t0 = _time.time()
                inputs = petals_tokenizer(request.prompt, return_tensors="pt")
                t1 = _time.time()
                print(f"[TIMING] Tokenization: {t1 - t0:.3f} seconds (Petals)")
                outputs = petals_model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                t3 = _time.time()
                print(f"[TIMING] Model.generate: {t3 - t1:.3f} seconds (Petals)")
                completion = petals_tokenizer.decode(outputs[0], skip_special_tokens=True)
                t4 = _time.time()
                print(f"[TIMING] Decoding: {t4 - t3:.3f} seconds (Petals)")
                print(f"[TIMING] Total: {t4 - t0:.3f} seconds (Petals)")
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
        else:
            raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")

