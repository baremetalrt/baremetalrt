# FastAPI server for OpenAI-compatible Llama 3 TRT-LLM inference
# Requirements: pip install fastapi uvicorn torch transformers tensorrt-llm
# Run with: uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
# Endpoint and schema compatible with OpenAI API (TensorRT-LLM style)
# See: https://platform.openai.com/docs/api-reference/completions/create

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
import os as _os
import os
import time
import threading
import gc
from threading import Lock
from tensorrt_llm import SamplingParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global model state (Llama 3 TRT-LLM only)
model = None
model_name = None
tokenizer = None
current_model_id = None
model_lock = Lock()
model_ready = False
hf_token = os.environ.get("HF_TOKEN", "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")

app = FastAPI(title="OpenAI-Compatible LLM API", description="OpenAI-style local inference endpoint.")

from . import model_manager
app.include_router(model_manager.router, prefix="/api")

def unload_model():
    global model, tokenizer
    if model is not None:
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        model = None
        tokenizer = None
    # Set status to offline when unloading
    try:
        update_model_status(current_model_id, "offline")
    except Exception:
        pass

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

def warmup_model(model, tokenizer, warmup_prompts, eos_token_id):
    logger.info("Warming up model with multiple prompts...")
    logger.debug(f"Tokenizer object at warmup: {tokenizer}")
    logger.debug(f"Model object at warmup: {model}")
    for prompt, tokens in warmup_prompts:
        warmup_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=tokens,
            end_id=eos_token_id
        )
        try:
            inputs = tokenizer(prompt)
            input_ids = inputs["input_ids"]  # This is a list of ints
            _ = model.generate([input_ids], warmup_params)
        except Exception as e:
            logger.error(f"Exception during warmup for prompt '{prompt}': {e}")
            import traceback
            traceback.print_exc()
    logger.info("Model warmup complete.")

def load_model(model_id):
    global model_ready
    model_ready = False
    global model, model_name, tokenizer, current_model_id
    # Set status to warming_up as soon as loading starts
    update_model_status(model_id, "warming_up")
    unload_model()
    try:
        match model_id:
            case "llama3.1_8b_trtllm_instruct_int4_streaming":
                logger.info("Loading Llama-3.1 8B Instruct (TensorRT-LLM, INT4, STREAMING) engine (instruction-tuned, GPU required, NEW ENGINE DIR)...")
                try:
                    from tensorrt_llm import LLM
                    from transformers import PreTrainedTokenizerFast
                    ENGINE_DIR = "/mnt/c/Github/baremetalrt/models/Llama-3.1-8B-trtllm-engine-streaming"
                    model = LLM(model=ENGINE_DIR)
                    import tensorrt_llm
                    logger.info(f"Loaded model: {model_name}")
                    logger.debug(f"Model type: {type(model)}")
                    logger.debug(f"Model attributes: {dir(model)}")
                    logger.info(f"TRT-LLM version: {tensorrt_llm.__version__}")
                    logger.info(f"Model has generate_stream: {hasattr(model, 'generate_stream')}")
                    model_name = "llama3.1_8b_trtllm_instruct_int4_streaming"
                    logger.info("TensorRT-LLM Instruct INT4 STREAMING engine loaded successfully.")
                    tokenizer_path = os.path.join(ENGINE_DIR, "tokenizer.json")
                    warmup_prompts = [
                        ("What is the capital of France? Explain in detail.", 256),
                        ("Write a Python function to compute Fibonacci numbers.", 128),
                        ("Summarize the theory of relativity in 100 words.", 200),
                        ("Explain the difference between supervised and unsupervised learning.", 128),
                        ("Generate a short story about a robot and a cat.", 150)
                    ]
                    if os.path.exists(tokenizer_path):
                        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                        logger.info("Loaded tokenizer from engine directory.")
                    else:
                        tokenizer = None
                        logger.warning("No tokenizer found in engine directory.")
                    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer and hasattr(tokenizer, 'convert_tokens_to_ids') else 2
                    warmup_model(model, tokenizer, warmup_prompts, eos_token_id)
                except Exception as e:
                    logger.error(f"Failed to load TRT-LLM Instruct INT4 STREAMING engine: {e}")
                    model = None
                    tokenizer = None
                    model_ready = False
                else:
                    if model is not None and tokenizer is not None:
                        model_ready = True
                    else:
                        logger.error("TRT-LLM Instruct INT4 STREAMING engine or tokenizer failed to load correctly!")
                        model_ready = False
                if model_ready:
                    logger.info("Model loaded on device: cuda:0 (TensorRT-LLM Instruct INT4 STREAMING)")
                else:
                    logger.warning("Model NOT loaded: TRT-LLM Instruct INT4 STREAMING is unavailable or failed to load.")
            case "llama3.1_8b_trtllm_instruct_int4":
                logger.info("Loading Llama-3.1 8B Instruct (TensorRT-LLM, INT4) engine (instruction-tuned, GPU required)...")
                try:
                    from tensorrt_llm import LLM
                    from transformers import PreTrainedTokenizerFast
                    ENGINE_DIR = "/mnt/c/Github/baremetalrt/models/Llama-3.1-8B-trtllm-engine"
                    model = LLM(model=ENGINE_DIR)
                    import tensorrt_llm
                    logger.info(f"Loaded model: {model_name}")
                    logger.info(f"TRT-LLM version: {tensorrt_llm.__version__}")
                    logger.info(f"Model has generate_stream: {hasattr(model, 'generate_stream')}")
                    model_name = "llama3.1_8b_trtllm_instruct_int4"
                    logger.info("TensorRT-LLM Instruct INT4 engine loaded successfully.")
                    tokenizer_path = os.path.join(ENGINE_DIR, "tokenizer.json")
                    warmup_prompts = [
                        ("What is the capital of France? Explain in detail.", 256),
                        ("Write a Python function to compute Fibonacci numbers.", 128),
                        ("Summarize the theory of relativity in 100 words.", 200),
                        ("Explain the difference between supervised and unsupervised learning.", 128),
                        ("Generate a short story about a robot and a cat.", 150)
                    ]
                    if not os.path.exists(tokenizer_path):
                        raise RuntimeError(f"Tokenizer file not found at {tokenizer_path}. Backend cannot start without tokenizer.")
                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                    logger.info("Loaded tokenizer from engine directory.")
                    # Call setup to enable stop sequence support
                    if hasattr(model, 'setup'):
                        logger.debug("Calling model.setup(tokenizer=tokenizer) to enable stop sequence support...")
                        model.setup(tokenizer=tokenizer)
                    else:
                        logger.warning("Model object does not have a 'setup' method. Stop sequence support may not work.")
                    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if hasattr(tokenizer, 'convert_tokens_to_ids') else 2
                    warmup_model(model, tokenizer, warmup_prompts, eos_token_id)
                except Exception as e:
                    logger.error(f"Failed to load TRT-LLM Instruct INT4 engine: {e}")
                    model = None
                    tokenizer = None
                    model_ready = False
                else:
                    if model is not None and tokenizer is not None:
                        model_ready = True
                    else:
                        logger.error("TRT-LLM Instruct INT4 engine or tokenizer failed to load correctly!")
                        model_ready = False
                if model_ready:
                    logger.info("Model loaded on device: cuda:0 (TensorRT-LLM Instruct INT4)")
            case _:
                raise ValueError(f"Unknown model_id: {model_id}")
        current_model_id = model_id
        update_model_status(model_id, "online")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        print(f"[ERROR] Failed to load model: {e}")
        update_model_status(model_id, "offline")
        model_ready = False
        raise

def get_online_model_id():
    import json
    status_path = _os.path.join(_os.path.dirname(__file__), "model_status.json")
    with open(status_path, "r") as f:
        status_dict = json.load(f)
    for model_id, status in status_dict.items():
        if status == "online":
            return model_id
    raise ValueError("No model marked as online in model_status.json")

def background_load_model():
    try:
        model_id = get_online_model_id()
        print(f"[INFO] Model marked as online in status file: {model_id}")
    except Exception as e:
        print(f"[WARN] No model marked as online in model_status.json ({e}), setting default to 8INT Instruct.")
        model_id = "llama3.1_8b_trtllm_instruct"
        update_model_status(model_id, "online")
    load_model(model_id)

@app.on_event("startup")
def startup_event():
    threading.Thread(target=background_load_model, daemon=True).start()

@app.get("/health")
def health():
    """Health check endpoint."""
    with model_lock:
        if current_model_id in ["llama3.1_8b_trtllm_instruct_int4", "llama3.1_8b_trtllm_instruct_int4_streaming"] and model is not None:
            device = str(model.device) if hasattr(model, 'device') else "cuda"
            return {"status": "ok", "model_loaded": True, "device": device, "active_model": current_model_id}
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
    max_tokens: int = 1024  # Increased default for more verbose completions
    temperature: float = 0.7
    top_p: float = 0.95

@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is warming up, please try again in a few seconds.")
    import time as _time
    import re
    with model_lock:
        match current_model_id:
            case "llama3.1_8b_trtllm_instruct_int4" if model is not None:
                try:
                    t0 = _time.time()
                    # Llama 3 native chat format with special tokens
                    preprompt = (
                        "<|begin_of_text|><|start_header_id|>system\n"
                        "You are a helpful assistant. Answer the user's question as fully and accurately as possible. "
                        "Do not ask follow-up questions or continue the conversation after your answer. "
                        "When you have finished answering, stop.\n"
                        "<|end_header_id|>\n"
                        "<|start_header_id|>user\n"
                    )
                    prompt = (
                        preprompt
                        + request.prompt.strip() + "\n"
                        + "<|end_header_id|>\n"
                        + "<|start_header_id|>assistant"
                    )
                    input_ids = tokenizer(prompt)["input_ids"]
                    max_seq_len = 2048  # Actual engine/model context window (see error). TODO: Auto-detect from model if possible.
                    prompt_strip = prompt.strip()
                    # Dynamic max_tokens logic for 4k context window
                    user_max = getattr(request, 'max_tokens', None)
                    if len(prompt_strip) <= 40:
                        # Short prompt: default 256, cap 1024, never exceed context window
                        if user_max is None:
                            max_tokens = min(256, 1024, max_seq_len - len(input_ids))
                        else:
                            max_tokens = min(user_max, 1024, max_seq_len - len(input_ids))
                    else:
                        # Long prompt: default 512, cap 8192, never exceed context window
                        if user_max is None:
                            max_tokens = min(512, 8192, max_seq_len - len(input_ids))
                        else:
                            max_tokens = min(user_max, 8192, max_seq_len - len(input_ids))
                    # Use <|eot_id|> as EOS and pass stop sequences as strings
                    eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if hasattr(tokenizer, 'convert_tokens_to_ids') else 2
                    stop_strings = ["<|end_header_id|>", "<|eot_id|>", "<|start_header_id|>assistant"]
                    print("[DEBUG] Stop strings:", stop_strings)
                    # TRT-LLM 0.19.0 does not support stop sequences, only EOS token
                    sampling_params = SamplingParams(
                        temperature=request.temperature,
                        top_p=request.top_p,
                        max_tokens=max_tokens,
                        end_id=eos_token_id
                    )
                    # stop_strings debug left for future upgrades
                    # Call setup immediately before inference to ensure stop sequence support
                    if hasattr(model, 'setup'):
                        print(f"[DEBUG] Calling model.setup(tokenizer=tokenizer) just before inference. Model id: {id(model)}, Tokenizer id: {id(tokenizer)}")
                        model.setup(tokenizer=tokenizer)
                    else:
                        print("[WARN] Model object does not have a 'setup' method. Stop sequence support may not work.")
                    outputs = model.generate([input_ids], sampling_params)
                    print(f"[DEBUG] Prompt: {prompt}")
                    print(f"[DEBUG] Input IDs: {input_ids}")
                    print(f"[DEBUG] Raw outputs: {outputs}")
                    if outputs and hasattr(outputs[0], 'outputs'):
                        print(f"[DEBUG] Number of output candidates: {len(outputs[0].outputs)}")
                        for i, seq in enumerate(outputs[0].outputs):
                            print(f"[DEBUG] Candidate {i}: {seq.text!r}")
                    def crop_output_for_short_prompt(prompt, output_text):
                        prompt_strip = prompt.strip()
                        output_strip = output_text.strip()
                        # Remove prompt/question if repeated at the start
                        if output_strip.startswith(prompt_strip):
                            output_strip = output_strip[len(prompt_strip):].lstrip("\n ")
                        # For short prompts, crop at next question (avoid Q&A chains)
                        if len(prompt_strip) <= 40 and output_strip:
                            qmark_idx = output_strip.find('?\n')
                            question_word = re.search(r'\n\s*(What|Who|Where|When|Why|How)[^\n]*\?', output_strip, re.IGNORECASE)
                            if qmark_idx > 0:
                                crop_idx = qmark_idx + 1
                                output_strip = output_strip[:crop_idx].strip()
                            elif question_word:
                                crop_idx = question_word.start()
                                output_strip = output_strip[:crop_idx].strip()
                            else:
                                newline_idx = output_strip.find('\n')
                                period_idx = output_strip.find('. ')
                                crop_points = [i for i in [newline_idx, period_idx+1 if period_idx!=-1 else -1] if i > 0]
                                if crop_points:
                                    crop_idx = min(crop_points)
                                    output_strip = output_strip[:crop_idx].strip()
                        return output_strip
                    if outputs and hasattr(outputs[0], 'outputs') and outputs[0].outputs:
                        full_answer = "".join([seq.text for seq in outputs[0].outputs])
                        answer = crop_output_for_short_prompt(prompt, full_answer)
                    else:
                        answer = ""
                    t1 = _time.time()
                    print(f"[TIMING] INT4 Generation: {t1 - t0:.3f} seconds")
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
                    raise HTTPException(status_code=500, detail=f"TRT-LLM INT4 inference failed: {e}")
            case _:
                raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")

# Streaming endpoint for TRT-LLM INT4
@app.post("/v1/completions/stream")
def stream_completion(request: CompletionRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is warming up, please try again in a few seconds.")
    import time as _time
    with model_lock:
        match current_model_id:
            case "llama3.1_8b_trtllm_instruct_int4_streaming" if model is not None:
                # Llama 3/2 Instruct-style preprompt
                preprompt = "[INST] <<SYS>>\nYou are a helpful, concise assistant. Answer only the user's question and stop.\n<</SYS>>\n\n"
                prompt = preprompt + request.prompt.strip() + " [/INST]"
                eos_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer and hasattr(tokenizer, 'convert_tokens_to_ids') else 2
                input_ids = tokenizer(prompt)["input_ids"] if tokenizer is not None else None
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    end_id=eos_token_id
                )
                if hasattr(model, "generate_stream") and callable(getattr(model, "generate_stream", None)):
                    def token_stream():
                        for token in model.generate_stream([input_ids], sampling_params):
                            token_text = tokenizer.decode([token]) if tokenizer is not None else str(token)
                            yield token_text
                    return StreamingResponse(token_stream(), media_type="text/plain")
                else:
                    logger.warning("Streaming not available for this engine/model. Falling back to non-streaming response.")
                    output = model.generate([input_ids], sampling_params)
                    if output and hasattr(output[0], 'outputs') and output[0].outputs:
                        output_text = "".join([seq.text for seq in output[0].outputs])
                    else:
                        output_text = ""
                    return {"choices": [{"text": output_text}]}
            case _:
                raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")
