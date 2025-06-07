import os
import sys
import subprocess
import argparse

# Default paths (edit as needed)
TRTLLM_CONVERT_SCRIPT = "/mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py"
ENGINE_EXPORT_SCRIPT = "/mnt/c/Github/baremetalrt/scripts/build_trtllm_engine.py"  # Updated to local build script

# Canonical model and engine/tokenizer directories
CANONICAL_MODEL_DIR = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-Instruct"
CANONICAL_ENGINE_DIR = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine"

# Ultra-conservative resource settings for consumer GPUs (guaranteed OOM-safe; increase if you have >24GB VRAM)
SAFE_MAX_BATCH_SIZE = 4
SAFE_MAX_NUM_SEQUENCES = 1
SAFE_MAX_INPUT_LEN = 2048
SAFE_MAX_SEQ_LEN = 2048  # You can increase these if you confirm no OOM

# Helper to run shell commands
def run(cmd, cwd=None):
    print(f"[RUN] {' '.join(str(x) for x in cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(str(x) for x in cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Auto-convert, export, and deploy a HuggingFace Llama model to TensorRT-LLM (SAFE GPU SETTINGS).")
    parser.add_argument("--model_dir", default=CANONICAL_MODEL_DIR, help="Path to HuggingFace model directory (default: canonical instruct model)")
    parser.add_argument("--output_dir", default=CANONICAL_ENGINE_DIR, help="Path to output TensorRT-LLM directory (default: canonical trtllm engine)")
    parser.add_argument("--quantization", default="int8", choices=["int8", "int4", "fp16"], help="Quantization type")
    parser.add_argument("--autodeploy", action="store_true", help="Enable experimental autodeploy if supported")
    parser.add_argument("--engine_export", action="store_true", help="Run engine export after conversion (if build.py exists)")
    args = parser.parse_args()

    # Step 1: Convert model
    convert_cmd = [sys.executable, TRTLLM_CONVERT_SCRIPT,
                   "--model_dir", args.model_dir,
                   "--output_dir", args.output_dir,
                   "--use_weight_only",
                   "--weight_only_precision", args.quantization]
    if args.quantization == "int8":
        convert_cmd.append("--int8_kv_cache")
    # DO NOT pass --autodeploy to convert_checkpoint.py (not supported)
    convert_cmd.append("--load_model_on_cpu")
    print(f"[INFO] Running model conversion with quantization={args.quantization}")
    try:
        run(convert_cmd)
    except Exception as e:
        print(f"[ERROR] Model conversion failed: {e}")
        sys.exit(1)

    # Step 2: (Optional) Export engine with SAFE resource settings
    if args.engine_export and os.path.exists(ENGINE_EXPORT_SCRIPT):
        # Clean engine dir except tokenizer.json
        print(f"[INFO] Cleaning engine directory: {args.output_dir}")
        for f in os.listdir(args.output_dir):
            if f != "tokenizer.json":
                try:
                    os.remove(os.path.join(args.output_dir, f))
                except Exception as e:
                    print(f"[WARN] Could not remove {f}: {e}")
        export_cmd = [sys.executable, ENGINE_EXPORT_SCRIPT,
                      "--model_dir", args.output_dir,
                      "--output_dir", args.output_dir,
                      "--max_batch_size", str(SAFE_MAX_BATCH_SIZE),
                      "--max_input_len", str(SAFE_MAX_INPUT_LEN),
                      "--max_seq_len", str(SAFE_MAX_SEQ_LEN)]
        print(f"[INFO] Building engine with max_batch_size={SAFE_MAX_BATCH_SIZE}, max_num_sequences={SAFE_MAX_NUM_SEQUENCES}, max_input_len={SAFE_MAX_INPUT_LEN}, max_seq_len={SAFE_MAX_SEQ_LEN}")
        if args.autodeploy:
            export_cmd.append("--autodeploy")
            print("[WARN] --autodeploy is only supported for the build step. Custom batch/seq settings will override build.py defaults.")
        try:
            run(export_cmd)
        except Exception as e:
            print(f"[ERROR] Engine build failed: {e}")
            sys.exit(2)
        # List engine dir contents
        print(f"[INFO] Engine directory contents after build:")
        for f in os.listdir(args.output_dir):
            print(f"  - {f}")
    # Always copy tokenizer.json from model_dir to output_dir after build
    import shutil
    src_tokenizer = os.path.join(args.model_dir, "tokenizer.json")
    dst_tokenizer = os.path.join(args.output_dir, "tokenizer.json")
    try:
        shutil.copy2(src_tokenizer, dst_tokenizer)
        print(f"[INFO] Copied tokenizer.json to {dst_tokenizer}")
    except Exception as e:
        print(f"[WARN] Could not copy tokenizer.json: {e}")

    print("[SUCCESS] Model conversion and (optional) deployment complete.\n\nNOTE: Engine built with safe batch/sequence settings. If you see 'KV cache reuse disabled' in logs, your engine or build script may not support paged context FMHA. Double-check build.py and your TRT-LLM version.")

if __name__ == "__main__":
    main()
