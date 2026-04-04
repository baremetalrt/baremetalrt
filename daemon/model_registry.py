"""Model catalog and local state for BareMetalRT daemon."""

import json
import os
from pathlib import Path

import sys
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Supported models — curated list of models known to work with TRT-LLM
CATALOG = [
    {
        "id": "phi-3-mini",
        "family": "Phi",
        "name": "Phi-3 Mini 3.8B Instruct",
        "hf_repo": "microsoft/Phi-3-mini-4k-instruct",
        "params_b": 3.8,
        "vram_fp16_mb": 8000,
        "context_length": 4096,
        "num_layers": 32,
        "description": "Microsoft's compact powerhouse. Trained on 3.3T tokens of heavily filtered web data and synthetic data. Outperforms models twice its size on reasoning benchmarks. Best balance of quality and speed for 8GB+ GPUs.",
        "license": "MIT",
    },
    {
        "id": "mistral-7b-instruct",
        "family": "Mistral",
        "name": "Mistral 7B Instruct v0.1",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.1",
        "params_b": 7.0,
        "vram_fp16_mb": 14500,
        "context_length": 8192,
        "num_layers": 32,
        "description": "Mistral AI's flagship 7B model with sliding window attention and grouped-query attention. Top-tier instruction following, coding, and reasoning. Needs 16GB+ VRAM at FP16 or quantization for smaller GPUs.",
        "license": "Apache 2.0",
    },
    {
        "id": "qwen2-7b-instruct",
        "family": "Qwen",
        "name": "Qwen2 7B Instruct",
        "hf_repo": "Qwen/Qwen2-7B-Instruct",
        "params_b": 7.0,
        "vram_fp16_mb": 14500,
        "context_length": 8192,
        "num_layers": 32,
        "description": "Alibaba's multilingual model supporting 29 languages including English, Chinese, French, Spanish, and Arabic. Strong on coding, math, and multilingual tasks. Needs 16GB+ VRAM at FP16.",
        "license": "Apache 2.0",
    },
    {
        "id": "tinyllama-1.1b",
        "family": "Llama",
        "name": "TinyLlama 1.1B Chat",
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "params_b": 1.1,
        "vram_fp16_mb": 2400,
        "context_length": 2048,
        "num_layers": 22,
        "description": "Compact 1.1B model trained on 3T tokens. Useful for testing and low-VRAM GPUs (4GB+). Limited instruction-following ability compared to larger models.",
        "license": "Apache 2.0",
    },
]


def _state_file() -> Path:
    return PROJECT_ROOT / "models" / "registry.json"


def _load_state() -> dict:
    f = _state_file()
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {"models": {}}


def _save_state(state: dict):
    f = _state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(state, indent=2))


def _dir_has_model_files(path: str | None) -> bool:
    """Check if a directory actually contains model files on disk."""
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    # Must have at least config.json or some .safetensors / .bin weights
    has_config = (p / "config.json").exists()
    has_weights = any(p.glob("*.safetensors")) or any(p.glob("*.bin"))
    return has_config or has_weights


def list_models(vram_mb: int = 0) -> list[dict]:
    """List all models with download/engine status and VRAM fit info."""
    state = _load_state()
    dirty = False
    result = []
    for m in CATALOG:
        model_state = state.get("models", {}).get(m["id"], {})
        downloaded = model_state.get("downloaded", False)
        hf_dir = model_state.get("hf_dir")
        engine_built = model_state.get("engine_built", False)
        engine_dir = model_state.get("engine_dir")

        # Validate downloaded flag against actual files on disk
        if downloaded and not _dir_has_model_files(hf_dir):
            downloaded = False
            hf_dir = None
            model_state["downloaded"] = False
            model_state.pop("hf_dir", None)
            dirty = True

        # Validate engine_built flag against actual engine dir
        if engine_built and (not engine_dir or not Path(engine_dir).is_dir()):
            engine_built = False
            engine_dir = None
            model_state["engine_built"] = False
            model_state.pop("engine_dir", None)
            dirty = True

        entry = {
            **m,
            "fits": vram_mb >= m["vram_fp16_mb"] if vram_mb else None,
            "downloaded": downloaded,
            "hf_dir": hf_dir,
            "engine_built": engine_built,
            "engine_dir": engine_dir,
        }
        result.append(entry)

    if dirty:
        _save_state(state)

    return result


def get_model(model_id: str) -> dict | None:
    """Get a single model entry."""
    for m in list_models():
        if m["id"] == model_id:
            return m
    return None


def mark_downloaded(model_id: str, hf_dir: str):
    state = _load_state()
    if "models" not in state:
        state["models"] = {}
    if model_id not in state["models"]:
        state["models"][model_id] = {}
    state["models"][model_id]["downloaded"] = True
    state["models"][model_id]["hf_dir"] = hf_dir
    _save_state(state)


def mark_engine_built(model_id: str, engine_dir: str):
    state = _load_state()
    if "models" not in state:
        state["models"] = {}
    if model_id not in state["models"]:
        state["models"][model_id] = {}
    state["models"][model_id]["engine_built"] = True
    state["models"][model_id]["engine_dir"] = engine_dir
    _save_state(state)


def recommend_model(vram_mb: int) -> dict | None:
    """Recommend the best model that fits the GPU."""
    models = list_models(vram_mb)
    fitting = [m for m in models if m["fits"]]
    if not fitting:
        return models[0] if models else None  # fallback to smallest
    # Pick the largest that fits
    return max(fitting, key=lambda m: m["params_b"])
