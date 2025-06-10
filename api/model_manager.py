from fastapi import APIRouter
from fastapi.responses import JSONResponse
import os
import json
try:
    from api.openai_api import model_ready
except ImportError:
    model_ready = True  # fallback if import fails (for testing)
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
            "id": "llama3.1_8b_trtllm_instruct_int4",
            "name": "Llama 3.1 8B Instruct (INT4)"
        },
        {
            "id": "llama3.1_8b_trtllm_instruct_int4_streaming",
            "name": "Llama 3.1 8B Instruct (INT4, Streaming)"
        },
        {
            "id": "llama2_7b_chat_8int",
            "name": "Llama 2 7B Chat (INT8)"
        },
        {
            "id": "deepseek_7b",
            "name": "Deepseek LLM 7B"
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

@router.get("/health")
def api_health_check():
    if model_ready:
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "warming_up"})

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

