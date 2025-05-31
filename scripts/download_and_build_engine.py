import os
import subprocess
import sys
from huggingface_hub import snapshot_download, login

# CONFIGURABLE PARAMETERS
MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # Change as needed
LOCAL_MODEL_DIR = "./Llama-2-7b-chat-hf"
# Path to new Llama conversion script in TensorRT-LLM
TRTLLM_CONVERT_SCRIPT = "/home/brian/baremetalrt/TensorRT-LLM/examples/models/core/llama/convert_checkpoint.py"
TRTLLM_MODEL_DIR = "./trtllm-model"  # Output dir for converted model (adjust as needed)

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

# 2. Convert Hugging Face model to TensorRT-LLM format using new script
def convert_to_trtllm(model_dir, output_dir):
    print(f"[INFO] Converting model using TensorRT-LLM convert_checkpoint.py...")
    cmd = [sys.executable, TRTLLM_CONVERT_SCRIPT, \
           "--input_dir", model_dir, "--output_dir", output_dir]
    print("[CMD]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print("[ERROR] Conversion failed. Printing help for convert_checkpoint.py:")
        help_cmd = [sys.executable, TRTLLM_CONVERT_SCRIPT, "--help"]
        subprocess.call(help_cmd)
        raise e

if __name__ == "__main__":
    ensure_model_downloaded(MODEL_ID, LOCAL_MODEL_DIR)
    convert_to_trtllm(LOCAL_MODEL_DIR, TRTLLM_MODEL_DIR)
    print("[SUCCESS] Model downloaded and converted!")
