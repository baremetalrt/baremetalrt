#!/usr/bin/env python3
"""
Best-practice TensorRT-LLM 0.19.0 engine build script.

- Place this file in your Windows project at scripts/build_trtllm_engine.py
- Edit parameters as needed for your model/checkpoint/output.
- Transfer to WSL2 if you want to run it there, or run from WSL2 if your scripts directory is mounted in your WSL home.
- Run with: python3 build_trtllm_engine.py

This script uses the official builder Python API, as recommended by NVIDIA for v0.19.0.
"""
#!/usr/bin/env python3
"""
Best-practice TensorRT-LLM 0.19.0 engine build script (pure Python API version).
- Uses LlamaForCausalLM, BuildConfig, and build() as recommended by NVIDIA for this version.
- Edit the checkpoint_dir and output_dir as needed.
- Run with: python3 build_trtllm_engine.py
"""
#!/usr/bin/env python3
"""
TensorRT-LLM 0.19.0 engine build script for Llama 3.1-8B INT4+INT8KV
(Code-backed, no guessing: imports LLaMAForCausalLM from correct submodule)
"""
import traceback
from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from tensorrt_llm.builder import build, BuildConfig

# NOTE: As of TensorRT-LLM 0.19.0, there is no official CLI engine build script. The recommended method is to use the Python API as shown here.

import argparse

checkpoint_dir_default = "/mnt/c/Github/baremetalrt/external/models/trtllm-Llama-3.1-8B-Instruct"
output_dir_default = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine"

parser = argparse.ArgumentParser(description="TensorRT-LLM Engine Build Script (configurable for OOM safety)")
parser.add_argument("--model_dir", default=checkpoint_dir_default, help="Path to checkpoint directory")
parser.add_argument("--output_dir", default=output_dir_default, help="Path to output engine directory")
parser.add_argument("--max_batch_size", type=int, default=1, help="Max batch size (default: 1)")
parser.add_argument("--max_input_len", type=int, default=2048, help="Max input length (default: 2048)")
parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length (default: 2048)")

parser.add_argument("--profiling_verbosity", default="layer_names_only", help="Profiling verbosity (default: layer_names_only)")
parser.add_argument("--use_paged_context_fmha", action="store_true", default=True, help="Enable paged context FMHA (default: True)")
args = parser.parse_args()

try:
    print(f"[INFO] Loading checkpoint from: {args.model_dir}")
    from tensorrt_llm.models.llama.model import LLaMAForCausalLM
    from tensorrt_llm.builder import build, BuildConfig
    model = LLaMAForCausalLM.from_checkpoint(args.model_dir)
    model.architecture = "llama3"

    build_config = BuildConfig(
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_seq_len=args.max_seq_len,
        strongly_typed=True,
        profiling_verbosity=args.profiling_verbosity,
        use_refit=False
    )
    print("[INFO] BuildConfig:")
    for k, v in build_config.__dict__.items():
        print(f"  {k}: {v}")

    engine = build(model, build_config)
    engine.save(args.output_dir)
    print(f"[SUCCESS] Engine built and saved to {args.output_dir}")
except Exception as e:
    print("Engine build failed:")
    import traceback
    traceback.print_exc()

