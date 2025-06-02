import os
import subprocess
import sys

# Adjust these paths as needed for your environment
TRTLLM_CONVERT_SCRIPT = "/mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py"
LOCAL_MODEL_DIR = "../external/models/Llama-3.1-8B"  # Input dir for original model (relative to scripts/)
TRTLLM_MODEL_DIR = "../external/models/trtllm-Llama-3.1-8B"  # Output dir for converted model

def convert_to_trtllm(model_dir, output_dir):
    print(f"[INFO] Converting model using TensorRT-LLM convert_checkpoint.py...")
    cmd = [sys.executable, TRTLLM_CONVERT_SCRIPT, \
           "--model_dir", model_dir, "--output_dir", output_dir]
    print("[CMD]", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print("[ERROR] Conversion failed. Printing help for convert_checkpoint.py:")
        help_cmd = [sys.executable, TRTLLM_CONVERT_SCRIPT, "--help"]
        subprocess.call(help_cmd)
        raise e

if __name__ == "__main__":
    convert_to_trtllm(LOCAL_MODEL_DIR, TRTLLM_MODEL_DIR)
    print("[SUCCESS] Model converted to TensorRT-LLM format!")
