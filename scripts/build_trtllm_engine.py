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
from tensorrt_llm.models.llama.model import LLaMAForCausalLM
from tensorrt_llm.builder import build, BuildConfig

# NOTE: As of TensorRT-LLM 0.19.0, there is no official CLI engine build script. The recommended method is to use the Python API as shown here.

checkpoint_dir = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-int4-int8kv"
output_dir = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine"

try:
    # Load model from checkpoint (confirmed method)
    model = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)

    # Create BuildConfig with correct engine build parameters
    build_config = BuildConfig(
        max_batch_size=4,                # Explicitly set max batch size for 4070 Super
        strongly_typed=True,
        profiling_verbosity="layer_names_only",
        use_refit=False
        # Add other BuildConfig fields as needed (max_input_len, max_seq_len, etc.)
    )

    # Build engine using BuildConfig (dataclass)
    engine = build(model, build_config)
    engine.save(output_dir)
    print(f"Engine built and saved to {output_dir}")
except Exception as e:
    print(f"Engine build failed: {e}")
