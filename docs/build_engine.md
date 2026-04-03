# Llama 3.1-8B TensorRT-LLM Engine Build Parameters

This engine was built from the converted TensorRT-LLM model using the following command and parameters:

```bash
# Example command (update with actual command used if different)
python3 /mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/llama/build.py \
  --model_dir /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-int4-int8kv \
  --output_dir /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine \
  --dtype bfloat16 \
  --tp_size 1 \
  --pp_size 1 \
  --max_batch_size 1 \
  --max_input_len 4096 \
  --max_output_len 512
```

- **TensorRT-LLM version:** 0.19.0
- **Build date:** 2025-06-02
- **Input model directory:** /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-int4-int8kv
- **Output engine directory:** /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine
- **Precision:** bfloat16
- **Tensor Parallelism (TP):** 1
- **Pipeline Parallelism (PP):** 1
- **Max batch size:** 1
- **Max input length:** 4096
- **Max output length:** 512
- **Notes:**
    - Update the command and parameters above if you use different values.
    - Document any hardware, environment, or build-specific details as needed.

---
This file documents the exact parameters used for engine build for reproducibility and future reference.
