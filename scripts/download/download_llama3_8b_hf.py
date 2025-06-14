import os
from huggingface_hub import snapshot_download

# Set your Hugging Face token here or use the environment variable HUGGINGFACE_TOKEN
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# The official Llama 3 8B repo
REPO_ID = "meta-llama/Meta-Llama-3-8B"
# Where to download the model
LOCAL_DIR = "/home/brian/baremetalrt/models/Llama-3-8b-hf"

if __name__ == "__main__":
    print(f"[INFO] Downloading {REPO_ID} to {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN
    )
    print(f"[SUCCESS] Model downloaded to {LOCAL_DIR}")
