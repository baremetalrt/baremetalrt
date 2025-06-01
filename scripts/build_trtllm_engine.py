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
from tensorrt_llm.builder import Builder, build

checkpoint_dir = "/home/brian/baremetalrt/models/Llama-3.1-8b-trtllm-int4-int8kv"
output_dir = "/home/brian/baremetalrt/models/Llama-3.1-8b-trtllm-engine"

# Load model from checkpoint (confirmed method)
model = LLaMAForCausalLM.from_checkpoint(checkpoint_dir)

# Create builder and build config (NVIDIA best practice, precision required as first positional argument)
builder = Builder()
config = builder.create_builder_config(
    "float16",            # precision (required positional argument)
    max_batch_size=4,
    max_input_len=2048,
    tp_size=1,
    parallel_build=True,
    workers=4,
    output_dir=output_dir
)

# Build engine
engine = build(model, config.to_dict())

# Save engine to output_dir
engine.save(output_dir)
print(f"Engine built and saved to {output_dir}")
