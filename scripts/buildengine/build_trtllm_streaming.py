import os
import subprocess
import sys

# Adjust these paths as needed for your environment
LOCAL_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Llama-3.1-8B-converted"))  # Absolute path for converted checkpoint dir
TRTLLM_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/Llama-3.1-8B-trtllm-streaming"))  # Absolute path for engine output dir

# Streaming conversion uses int4 weight-only and int8 kv cache

def convert_to_trtllm_streaming(model_dir, output_dir):
    print(f"[INFO] Building TRT-LLM engine for streaming using new builder API (tensorrt_llm.commands.build)...")
    # Use only CLI flags for quantization and streaming (no build config JSON)
    # Use --model_config to specify quantization and streaming (see trtllm_model_config.json)
    model_config_path = os.path.join(output_dir, "trtllm_model_config.json")
    cmd = [sys.executable, "-m", "tensorrt_llm.commands.build",
           "--checkpoint_dir", model_dir,
           "--output_dir", output_dir,
           "--model_config", model_config_path,
           "--kv_cache_type", "paged",
           "--streamingllm", "enable"]
    # NOTE: Only one streaming engine directory should be used. Remove or archive any duplicate engine dirs to avoid confusion.
    print("[CMD]", " ".join(cmd))
    # Force the correct config: copy TRT-LLM config to checkpoint dir as config.json
    import shutil
    shutil.copyfile(
        os.path.join(TRTLLM_MODEL_DIR, "trtllm_model_config.json"),
        os.path.join(LOCAL_MODEL_DIR, "config.json")
    )
    # Run the TRT-LLM builder command
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        print("[ERROR] Engine build failed. Printing help for builder:")
        help_cmd = [sys.executable, "-m", "tensorrt_llm.commands.build", "--help"]
        subprocess.call(help_cmd)
        raise e

if __name__ == "__main__":
    convert_to_trtllm_streaming(LOCAL_MODEL_DIR, TRTLLM_MODEL_DIR)
    print("[SUCCESS] Streaming model built to TensorRT-LLM format!")
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
