# FastAPI server for OpenAI-compatible Llama 2 inference
# Requirements: pip install fastapi uvicorn torch transformers
# Run with: uvicorn api.openai_api:app --host 0.0.0.0 --port 8000
# Endpoint and schema compatible with OpenAI API (TensorRT-LLM style)
# See: https://platform.openai.com/docs/api-reference/completions/create

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os as _os
import os
import time
import threading
import gc
from threading import Lock
from tensorrt_llm import SamplingParams  # Ensure SamplingParams is always available
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
model_ready = False
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
    # Set status to offline when unloading
    try:
        update_model_status(current_model_id, "offline")
    except Exception:
        pass
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
    global model_ready
    model_ready = False
    global model, model_name, tokenizer, current_model_id
    # Set status to warming_up as soon as loading starts
    update_model_status(model_id, "warming_up")
    # petals_model, petals_tokenizer
    unload_model()
    try:
        if model_id == "llama3.1_8b_trtllm_instruct_int4_streaming":
            print("Loading Llama-3.1 8B Instruct (TensorRT-LLM, INT4, STREAMING) engine (instruction-tuned, GPU required, NEW ENGINE DIR)...")
            try:
                from tensorrt_llm import LLM
                from transformers import PreTrainedTokenizerFast
                # Updated engine directory for new streaming model
                ENGINE_DIR = "/mnt/c/Github/baremetalrt/models/Llama-3.1-8B-trtllm-engine-streaming"
                model = LLM(model=ENGINE_DIR)  # Use engine dir for streaming model as with non-streaming
                import tensorrt_llm
                print(f"[INFO] Loaded model: {model_name}")
                print(f"[DEBUG] Model type: {type(model)}")
                print(f"[DEBUG] Model attributes: {dir(model)}")
                print(f"[INFO] TRT-LLM version: {tensorrt_llm.__version__}")
                print(f"[INFO] Model has generate_stream: {hasattr(model, 'generate_stream')}")
                model_name = "llama3.1_8b_trtllm_instruct_int4_streaming"
                print("TensorRT-LLM Instruct INT4 STREAMING engine loaded successfully.")
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
                    print("Loaded tokenizer from engine directory.")
                else:
                    tokenizer = None
                    print("[WARN] No tokenizer found in engine directory.")
                print("Warming up TRT-LLM Instruct INT4 STREAMING engine with multiple prompts...")
                for prompt, tokens in warmup_prompts:
                    warmup_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=tokens,
                        end_id=2
                    )
                    _ = model.generate([prompt], warmup_params)
                print("[INFO] TRT-LLM Instruct INT4 STREAMING engine full warmup complete.")
            except Exception as e:
                print(f"[ERROR] Failed to load TRT-LLM Instruct INT4 STREAMING engine: {e}")
                model = None
                tokenizer = None
                model_ready = False
            else:
                if model is not None and tokenizer is not None:
                    model_ready = True
                else:
                    print("[ERROR] TRT-LLM Instruct INT4 STREAMING engine or tokenizer failed to load correctly!")
                    model_ready = False
            petals_model = None
            petals_tokenizer = None
            if model_ready:
                print("Model loaded on device: cuda:0 (TensorRT-LLM Instruct INT4 STREAMING)")
            else:
                print("[WARN] Model NOT loaded: TRT-LLM Instruct INT4 STREAMING is unavailable or failed to load.")
        elif model_id == "llama3.1_8b_trtllm_instruct_int4":
            print("Loading Llama-3.1 8B Instruct (TensorRT-LLM, INT4) engine (instruction-tuned, GPU required)...")
            try:
                from tensorrt_llm import LLM
                from transformers import PreTrainedTokenizerFast
                ENGINE_DIR = "/mnt/c/Github/baremetalrt/models/Llama-3.1-8B-trtllm-engine"
                model = LLM(model=ENGINE_DIR)
                import tensorrt_llm
                print(f"[INFO] Loaded model: {model_name}")
                print(f"[INFO] TRT-LLM version: {tensorrt_llm.__version__}")
                print(f"[INFO] Model has generate_stream: {hasattr(model, 'generate_stream')}")
                model_name = "llama3.1_8b_trtllm_instruct_int4"
                print("TensorRT-LLM Instruct INT4 engine loaded successfully.")
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
                print("Loaded tokenizer from engine directory.")
                # Call setup to enable stop sequence support
                if hasattr(model, 'setup'):
                    print("[DEBUG] Calling model.setup(tokenizer=tokenizer) to enable stop sequence support...")
                    model.setup(tokenizer=tokenizer)
                else:
                    print("[WARN] Model object does not have a 'setup' method. Stop sequence support may not work.")
                print("Warming up TRT-LLM Instruct INT4 engine with multiple prompts...")
                print(f"[DEBUG] Tokenizer object at warmup: {tokenizer}")
                print(f"[DEBUG] Model object at warmup: {model}")
                for prompt, tokens in warmup_prompts:
                    warmup_params = SamplingParams(
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=tokens,
                        end_id=2
                    )
                    try:
                        inputs = tokenizer(prompt)
                        input_ids = inputs["input_ids"]  # This is a list of ints
                        _ = model.generate([input_ids], warmup_params)
                    except Exception as e:
                        print(f"[ERROR] Exception during warmup for prompt '{prompt}': {e}")
                        import traceback
                        traceback.print_exc()
                print("[INFO] TRT-LLM Instruct INT4 engine full warmup complete.")
            except Exception as e:
                print(f"[ERROR] Failed to load TRT-LLM Instruct INT4 engine: {e}")
                model = None
                tokenizer = None
                model_ready = False
            else:
                if model is not None and tokenizer is not None:
                    model_ready = True
                else:
                    print("[ERROR] TRT-LLM Instruct INT4 engine or tokenizer failed to load correctly!")
                    model_ready = False
            petals_model = None
            petals_tokenizer = None
            print("Model loaded on device: cuda:0 (TensorRT-LLM Instruct INT4)")
        elif model_id == "llama2_7b_chat_8int":


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

        elif model_id == "llama3.1_8b_trtllm_8int":

            print("Loading Llama-3.1 8B (TensorRT-LLM, INT8) engine (ultra-fast, INT8 quantized, GPU required)...")
            try:
                from tensorrt_llm import LLM
                from transformers import PreTrainedTokenizerFast
                ENGINE_DIR = "/mnt/c/Github/baremetalrt/models/Llama-3.1-8B-trtllm-engine"
                model = LLM(model=ENGINE_DIR)
                model_name = "llama3.1_8b_trtllm_8int"
                print("TensorRT-LLM INT8 engine loaded successfully.")
                model_ready = True
                # Try to load tokenizer from tokenizer.json if present
                import os as _os
                tokenizer_path = _os.path.join(ENGINE_DIR, "tokenizer.json")
                if _os.path.exists(tokenizer_path):
                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
                    print("Loaded tokenizer.json with")
                    print("Warning: tokenizer.json not found, proceeding without tokenizer.")
                # Improved warmup: multiple realistic prompts with higher max_tokens
                try:
                    warmup_prompts = [
                        ("Hello!", 8),
                        ("What is the capital of France? Explain in detail.", 256),
                        ("Write a Python function to compute Fibonacci numbers.", 128),
                        ("Summarize the theory of relativity in 100 words.", 200),
                        ("Explain the difference between supervised and unsupervised learning.", 128),
                        ("Generate a short story about a robot and a cat.", 150),
                        ("List the first 20 prime numbers.", 32),
                        ("Translate the following English text to French: 'The quick brown fox jumps over the lazy dog.'", 32),
                        ("Write a poem about the ocean.", 64),
                        ("Explain how to implement a binary search algorithm in Python.", 128),
                        ("Describe the process of photosynthesis.", 96)
                    ]
                    print("Warming up TRT-LLM engine with multiple prompts...")
                    for prompt, tokens in warmup_prompts:
                        warmup_params = SamplingParams(
                            temperature=0.7,
                            top_p=0.95,
                            max_tokens=tokens,
                            end_id=2
                        )
                        _ = model.generate([prompt], warmup_params)
                    print("[INFO] TRT-LLM engine full warmup complete.")
                except Exception as e:
                    print(f"[WARN] TRT-LLM warmup failed: {e}")
                model_ready = True
            except Exception as e:
                print(f"[ERROR] Failed to load TRT-LLM engine: {e}")
                model = None
                tokenizer = None
            petals_model = None
            petals_tokenizer = None
            print("Model loaded on device: cuda:0 (TensorRT-LLM)")
        else:
            raise ValueError(f"Unknown model_id: {model_id}")
        current_model_id = model_id
        update_model_status(model_id, "online")
    except Exception as e:
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
        if current_model_id == "llama3.1_8b_trtllm_instruct_int4" and model is not None:
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
        else:
            raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")

# Streaming endpoint for TRT-LLM INT4
@app.post("/v1/completions/stream")
def stream_completion(request: CompletionRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is warming up, please try again in a few seconds.")
    import time as _time
    with model_lock:
        if current_model_id == "llama3.1_8b_trtllm_instruct_int4_streaming" and model is not None:
            # Llama 3/2 Instruct-style preprompt
            preprompt = "[INST] <<SYS>>\nYou are a helpful, concise assistant. Answer only the user's question and stop.\n<</SYS>>\n\n"
            prompt = preprompt + request.prompt.strip() + " [/INST]"
            input_ids = tokenizer(prompt)["input_ids"] if tokenizer is not None else None
            sampling_params = SamplingParams(
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                end_id=2
            )
            if hasattr(model, "generate_stream") and callable(getattr(model, "generate_stream", None)):
                def token_stream():
                    for token in model.generate_stream([input_ids], sampling_params):
                        token_text = tokenizer.decode([token]) if tokenizer is not None else str(token)
                        yield token_text
                return StreamingResponse(token_stream(), media_type="text/plain")
            else:
                print("[WARN] Streaming not available for this engine/model. Falling back to non-streaming response.")
                output = model.generate([input_ids], sampling_params)
                if output and hasattr(output[0], 'outputs') and output[0].outputs:
                    output_text = "".join([seq.text for seq in output[0].outputs])
                else:
                    output_text = ""
                return {"choices": [{"text": output_text}]}

        elif current_model_id == "llama3.1_8b_trtllm_instruct_int4" and model is not None:
            # Explicitly reject streaming for non-streaming model
            raise HTTPException(status_code=501, detail="Streaming not available for this model/engine. Please use the non-streaming /v1/completions endpoint.")
        else:
            raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is warming up, please try again in a few seconds.")
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
        elif current_model_id == "llama3.1_8b_trtllm_instruct" and model is not None:
            # TRT-LLM inference path
            try:
                t0 = _time.time()
                prompt = request.prompt
                # Use tokenizer if available for input_ids, else just pass prompt
                if tokenizer is not None:
                    input_ids = tokenizer.encode(prompt, return_tensors=None)
                else:
                    input_ids = None
                # Use default sampling params or map from request
                                # Ensure max_tokens does not exceed model context window
                input_ids = tokenizer(prompt)["input_ids"]
                prompt_length = len(input_ids)
                max_seq_len = 2048
                max_tokens = min(request.max_tokens, max_seq_len - prompt_length)
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    end_id=2  # EOS token for Llama models (adjust if needed)
                )
                # TRT-LLM expects a list of prompts
                outputs = model.generate([prompt], sampling_params)
                # Concatenate all outputs for the first prompt (if multiple candidates are returned)
                if outputs and outputs[0].outputs:
                    output_text = "".join([seq.text for seq in outputs[0].outputs])
                else:
                    output_text = ""
                t1 = _time.time()
                print(f"[TIMING] TRT-LLM Generation: {t1 - t0:.3f} seconds")
                response = {
                    "id": f"cmpl-{int(time.time()*1000)}",
                    "object": "text_completion",
                    "created": int(_time.time()),
                    "model": current_model_id,
                    "choices": [
                        {
                            "text": output_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop"
                        }
                    ]
                }
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"TRT-LLM inference failed: {e}")
        elif current_model_id == "llama3.1_8b_trtllm_instruct_int4" and model is not None:
            # TRT-LLM INT4 inference path
            try:
                t0 = _time.time()
                # Use OpenAI-style chat history if provided
                prompt = None
                if hasattr(request, 'messages') and request.messages:
                    # Prepend system prompt and format messages
                    prompt = "System: You are a helpful assistant.\n"
                    for msg in request.messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            prompt += f"User: {content}\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n"
                    prompt += "Assistant:"
                else:
                    prompt = request.prompt
                # Explicitly tokenize prompt to list of ints
                input_ids = tokenizer(prompt)["input_ids"]
                input_batch = [input_ids]  # Batch of 1
                from tensorrt_llm.llmapi import SamplingParams
                # Set EOS token for end_id (prefer tokenizer.eos_token_id, fallback to 2)
                eos_token_id = getattr(tokenizer, 'eos_token_id', None)
                if eos_token_id is None:
                    # Try to get EOS by decoding the special token or fallback to 2
                    try:
                        eos_token = getattr(tokenizer, 'eos_token', None)
                        if eos_token is not None:
                            eos_token_id = tokenizer.encode(eos_token, add_special_tokens=False)[0]
                        else:
                            eos_token_id = 2
                    except Exception:
                        eos_token_id = 2
                print(f"[DEBUG] Using end_id for INT4 SamplingParams: {eos_token_id}")
                # Use max_tokens (not max_new_tokens) and set a high default if not provided
                # Set max_tokens to maximum allowed by context window
                max_seq_len = 2048  # Ensure this matches your INT4 engine's context window
                prompt_strip = prompt.strip()
                if len(prompt_strip) <= 40:
                    # For short prompts, cap max_tokens to 128 or less
                    max_tokens = getattr(request, 'max_tokens', None)
                    if max_tokens is None:
                        max_tokens = min(128, max_seq_len - len(input_ids))
                    else:
                        max_tokens = min(max_tokens, 128, max_seq_len - len(input_ids))
                else:
                    max_tokens = getattr(request, 'max_tokens', None)
                    if max_tokens is None:
                        max_tokens = max_seq_len - len(input_ids)
                    else:
                        max_tokens = min(max_tokens, max_seq_len - len(input_ids))
                # Smart default stop: prevent Q&A chains, allow multi-line answers
                # Use only EOS token as stop sequence (most robust for TRT-LLM INT4)
                stop = getattr(request, 'stop', None)
                if stop is None:
                    stop = [tokenizer.eos_token] if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token else None
                # Do not call setup; pass stop directly as EOS
                sampling_params = SamplingParams(
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=max_tokens,
                    end_id=eos_token_id,
                    stop=stop
                )
                outputs = model.generate(input_batch, sampling_params)
                print(f"[DEBUG] Prompt: {prompt}")
                print(f"[DEBUG] Input IDs: {input_ids}")
                print(f"[DEBUG] Raw outputs: {outputs}")
                if outputs and hasattr(outputs[0], 'outputs'):
                    print(f"[DEBUG] Number of output candidates: {len(outputs[0].outputs)}")
                    for i, seq in enumerate(outputs[0].outputs):
                        print(f"[DEBUG] Candidate {i}: {seq.text!r}")
                # outputs is a list of RequestOutput objects; each has .outputs (list of candidates), each with .text
                import re
                def crop_output_for_short_prompt(prompt, output_text):
                    import re
                    prompt_strip = prompt.strip()
                    output_strip = output_text.strip()
                    # Remove prompt/question if repeated at the start
                    if output_strip.startswith(prompt_strip):
                        output_strip = output_strip[len(prompt_strip):].lstrip("\n ")
                    # For short prompts, crop at next question (avoid Q&A chains)
                    if len(prompt_strip) <= 40 and output_strip:
                        # Find next question pattern: either '?\n', or a line starting with question word
                        qmark_idx = output_strip.find('?\n')
                        # Regex for lines starting with question words (What, Who, Where, When, Why, How, etc.)
                        question_word = re.search(r'\n\s*(What|Who|Where|When|Why|How)[^\n]*\?', output_strip, re.IGNORECASE)
                        if qmark_idx > 0:
                            crop_idx = qmark_idx + 1  # include the '?'
                            output_strip = output_strip[:crop_idx].strip()
                        elif question_word:
                            crop_idx = question_word.start()
                            output_strip = output_strip[:crop_idx].strip()
                        else:
                            # Fallback: first newline or period
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
        else:
            raise HTTPException(status_code=503, detail="No model loaded or model is still loading. Please switch to a model and try again.")

