import os
import sys
import subprocess
import argparse

# Default paths (edit as needed)
TRTLLM_CONVERT_SCRIPT = "/mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py"
ENGINE_EXPORT_SCRIPT = "/mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/build.py"  # Example, adjust if needed


def run(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Auto-convert, export, and deploy a HuggingFace Llama model to TensorRT-LLM.")
    parser.add_argument("--model_dir", required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--output_dir", required=True, help="Path to output TensorRT-LLM directory")
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
    if args.autodeploy:
        convert_cmd.append("--autodeploy")
    convert_cmd.append("--load_model_on_cpu")
    run(convert_cmd)

    # Step 2: (Optional) Export engine
    if args.engine_export and os.path.exists(ENGINE_EXPORT_SCRIPT):
        export_cmd = [sys.executable, ENGINE_EXPORT_SCRIPT,
                      "--model_dir", args.output_dir]
        if args.autodeploy:
            export_cmd.append("--autodeploy")
        run(export_cmd)
    print("[SUCCESS] Model conversion and (optional) deployment complete.")

if __name__ == "__main__":
    main()
