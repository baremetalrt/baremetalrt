from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# For now, we hardcode the available models. Later, this can be extended to scan scripts or config.
import os

import subprocess
PETALS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "petals_llama2_70b_chat.py")

def get_petals_status():
    if not os.path.exists(PETALS_SCRIPT):
        return False, "Not available on this node. Script missing."
    try:
        result = subprocess.run([
            "python", PETALS_SCRIPT, "--check"
        ], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "ONLINE" in result.stdout:
            return True, "Available. Mesh connection OK."
        else:
            msg = result.stdout.strip() or result.stderr.strip() or "Unknown error."
            return False, f"Offline: {msg}"
    except Exception as e:
        return False, f"Offline: {e}"

class ModelInfo(BaseModel):
    id: str
    name: str
    status: str
    description: str

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """List available models and their status."""
    petals_online, petals_desc = get_petals_status()
    models = [
        {
            "id": "llama2_7b_chat_8int",
            "name": "Llama 2 7B Chat (8-bit, INT)",
            "status": "online",
            "description": "Quantized Llama 2 7B chat model (8-bit)"
        },
        {
            "id": "llama2_70b_chat_petals",
            "name": "Llama 2 70B Chat (Petals)",
            "status": "online" if petals_online else "offline",
            "description": f"Distributed Llama 2 70B via Petals mesh (requires internet). {petals_desc}"
        }
    ]
    return models

class ModelInfo(BaseModel):
    id: str
    name: str
    status: str
    description: str

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """List available models and their status."""
    return AVAILABLE_MODELS
