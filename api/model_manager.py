from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# For now, we hardcode the available models. Later, this can be extended to scan scripts or config.
import os

PETALS_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "petals_llama2_70b_chat.py")
petals_online = os.path.exists(PETALS_SCRIPT)

AVAILABLE_MODELS = [
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
        "description": "Distributed Llama 2 70B via Petals mesh (requires internet). " + ("Available." if petals_online else "Not available on this node.")
    }
]

class ModelInfo(BaseModel):
    id: str
    name: str
    status: str
    description: str

@router.get("/models", response_model=List[ModelInfo])
def list_models():
    """List available models and their status."""
    return AVAILABLE_MODELS
