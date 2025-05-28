from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# List available models
@router.get("/models")
async def get_models():
    import json
    import os
    base_dir = os.path.dirname(__file__)
    status_path = os.path.abspath(os.path.join(base_dir, "model_status.json"))
    models = [
        {
            "id": "llama2_7b_chat_8int",
            "name": "Llama 2 7B Chat INT8"
        },
        {
            "id": "llama2_13b_chat_4bit",
            "name": "Llama 2 13B Chat INT4"
        },
        {
            "id": "llama2_7b_chat_fp16",
            "name": "Llama 2 7B Chat FP16"
        },
        {
            "id": "llama3_8b_chat_4bit",
            "name": "Llama-3 8B Chat INT4"
        },
        {
            "id": "deepseek_llm_7b_chat_4bit",
            "name": "Deepseek LLM 7B Chat INT4"
        },
        {
            "id": "mistral_7b_instruct_8bit",
            "name": "Mistral 7B Instruct INT8"
        },
        {
            "id": "mixtral_8x7b_instruct_4bit",
            "name": "Mixtral 8x7B Instruct INT4"
        },
        {
            "id": "llama2_70b_chat_petals",
            "name": "Llama 2 70B Chat (petals)"
        }
    ]
    # Load status from JSON
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            status_dict = json.load(f)
    else:
        status_dict = {}
    result = []
    for model in models:
        status = status_dict.get(model["id"], "offline")
        result.append({
            "id": model["id"],
            "name": model["name"],
            "status": status
        })
    return result

# For now, we hardcode the available models. Later, this can be extended to scan scripts or config.
import os

import subprocess
PETALS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "petals_llama2_70b_chat.py")

def get_petals_status():
    if not os.path.exists(PETALS_SCRIPT):
        return False, "Not available on this node. Script missing."
    try:
        import sys
        result = subprocess.run([
            sys.executable, PETALS_SCRIPT, "--check"
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "ONLINE" in result.stdout:
            return True, "Available. Mesh connection OK."
        else:
            msg = result.stdout.strip() or result.stderr.strip() or "Unknown error."
            return False, f"Offline: {msg}"
    except Exception as e:
        return False, f"Offline: {e}"

