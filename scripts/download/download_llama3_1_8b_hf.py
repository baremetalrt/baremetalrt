import os
from huggingface_hub import snapshot_download, login
import sys

REPO_ID = "meta-llama/Llama-3.1-8B"
LOCAL_DIR = "/home/brian/baremetalrt/models/Llama-3.1-8b-hf"

# Set your Hugging Face token (env var or hardcoded fallback)
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = "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"  # <-- Replace with your token if needed
    if not HF_TOKEN:
        print("[ERROR] Please set your Hugging Face token in the HUGGINGFACE_HUB_TOKEN environment variable or hardcode it in the script.")
        sys.exit(1)

login(token=HF_TOKEN)

if __name__ == "__main__":
    print(f"[INFO] Downloading {REPO_ID} to {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
        token=HF_TOKEN
    )
    print(f"[SUCCESS] Model downloaded to {LOCAL_DIR}")
