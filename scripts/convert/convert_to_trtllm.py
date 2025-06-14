import os
import subprocess
import sys

# Adjust these paths as needed for your environment
TRTLLM_CONVERT_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py"))
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Llama-3.1-8B-Instruct"))  # Absolute path for input dir
TRTLLM_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Llama-3.1-8B-converted"))  # Absolute path for output dir

def convert_to_trtllm(model_dir, output_dir):
    print(f"[INFO] Converting model using TensorRT-LLM convert_checkpoint.py...")
    cmd = [
        sys.executable, TRTLLM_CONVERT_SCRIPT,
        "--model_dir", model_dir,
        "--output_dir", output_dir,
        "--use_weight_only",
        "--weight_only_precision", "int4",
        "--n_positions", "8192",
        "--load_model_on_cpu"
    ]
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
    # Copy tokenizer files for inference
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = os.path.join(LOCAL_MODEL_DIR, fname)
        dst = os.path.join(TRTLLM_MODEL_DIR, fname)
        if os.path.exists(src):
            print(f"[INFO] Copying {fname} to engine dir...")
            import shutil
            shutil.copyfile(src, dst)
        else:
            print(f"[WARN] {fname} not found in Hugging Face model dir.")
