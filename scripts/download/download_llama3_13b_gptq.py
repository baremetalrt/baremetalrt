import os
from huggingface_hub import snapshot_download, login
import sys

MODEL_ID = "meta-llama/Meta-Llama-3-13B-Instruct"  # Official instruct/chat model for Llama-3 13B
LOCAL_MODEL_DIR = "./Meta-Llama-3-13B-Instruct"

# Set your Hugging Face token (hardcoded fallback)
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = "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"  # <-- Replace with your token if needed
    if not HF_TOKEN:
        print("[ERROR] Please set your Hugging Face token in the HUGGINGFACE_HUB_TOKEN environment variable or hardcode it in the script.")
        sys.exit(1)

login(token=HF_TOKEN)

def ensure_model_downloaded(model_id, local_dir):
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        print(f"[INFO] Downloading model {model_id} to {local_dir} ...")
        snapshot_download(repo_id=model_id, local_dir=local_dir, resume_download=True)
    else:
        print(f"[INFO] Model directory {local_dir} already exists and is not empty.")

if __name__ == "__main__":
    ensure_model_downloaded(MODEL_ID, LOCAL_MODEL_DIR)
    print("[SUCCESS] Llama-3 13B GPTQ model downloaded (or already present)!")
