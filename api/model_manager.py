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

