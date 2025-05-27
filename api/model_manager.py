from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

# List available models
@router.get("/models")
async def get_models():
    import sys
    import subprocess
    base_dir = os.path.dirname(__file__)
    scripts_dir = os.path.abspath(os.path.join(base_dir, "..", "scripts"))
    models = [
        {
            "id": "llama2_7b_chat_8int",
            "name": "Llama 2 7B Chat 8INT (4070 Super)",
            "script": os.path.join(scripts_dir, "llama2_7b_chat_8int.py")
        },
        {
            "id": "llama2_70b_chat_petals",
            "name": "Llama 2 70B Chat (Petals mesh)",
            "script": os.path.join(scripts_dir, "petals_llama2_70b_chat.py")
        },
        {
            "id": "mistral_7b_instruct_8bit",
            "name": "Mistral 7B Instruct 8INT (4070 Super)",
            "script": os.path.join(scripts_dir, "mistral_7b_instruct_8bit.py")
        }
    ]
    result = []
    for model in models:
        script_path = model["script"]
        if not os.path.exists(script_path):
            status = "offline"
        else:
            try:
                proc = subprocess.run(
                    [sys.executable, script_path, "--check"],
                    capture_output=True, text=True, timeout=5
                )
                if proc.returncode == 0 and "ONLINE" in proc.stdout:
                    status = "online"
                else:
                    status = "offline"
            except Exception:
                status = "offline"
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

