import os
import subprocess
import sys
from huggingface_hub import snapshot_download, login

# CONFIGURABLE PARAMETERS
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Change as needed
LOCAL_MODEL_DIR = "./Llama-2-7b-chat-hf"
TRTLLM_EXPORT_SCRIPT = "examples/llama/export.py"  # Path to export.py in your TensorRT-LLM repo
TRTLLM_BUILD_SCRIPT = "examples/llama/build.py"    # Path to build.py in your TensorRT-LLM repo
TRTLLM_MODEL_DIR = "./trtllm-model"
TRTLLM_ENGINE_DIR = "./trtllm-engine"

# Set your Hugging Face token if required (Llama-2 is gated)
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    # Fallback: hardcode your token here for convenience
    HF_TOKEN = "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"  # <-- Replace with your token if needed
    if not HF_TOKEN:
        print("[ERROR] Please set your Hugging Face token in the HUGGINGFACE_HUB_TOKEN environment variable or hardcode it in the script.")
        sys.exit(1)

# Authenticate with Hugging Face
login(token=HF_TOKEN)

# 1. Download model from Hugging Face if not already present
def ensure_model_downloaded(model_id, local_dir):
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"[INFO] Downloading model {model_id} to {local_dir} ...")
        snapshot_download(repo_id=model_id, local_dir=local_dir, resume_download=True)
    else:
        print(f"[INFO] Model directory {local_dir} already exists and is not empty.")

# 2. Export to TensorRT-LLM format
def export_to_trtllm(model_dir, output_dir):
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print(f"[INFO] Exporting model to TensorRT-LLM format...")
        cmd = [sys.executable, TRTLLM_EXPORT_SCRIPT, "--model_dir", model_dir, "--output_dir", output_dir]
        print("[CMD]", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"[INFO] TRT-LLM export directory {output_dir} already exists and is not empty.")

# 3. Build the TensorRT engine
def build_engine(model_dir, output_dir):
    if not os.path.exists(output_dir) or not os.listdir(output_dir):
        print(f"[INFO] Building TensorRT engine...")
        cmd = [sys.executable, TRTLLM_BUILD_SCRIPT, "--model_dir", model_dir, "--output_dir", output_dir, "--dtype", "float16", "--max_input_len", "4096", "--max_output_len", "1024"]
        print("[CMD]", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"[INFO] Engine directory {output_dir} already exists and is not empty.")

if __name__ == "__main__":
    ensure_model_downloaded(MODEL_ID, LOCAL_MODEL_DIR)
    export_to_trtllm(LOCAL_MODEL_DIR, TRTLLM_MODEL_DIR)
    build_engine(TRTLLM_MODEL_DIR, TRTLLM_ENGINE_DIR)
    print("[SUCCESS] Model downloaded, exported, and engine built!")
