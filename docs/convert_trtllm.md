# Llama 3.1-8B TensorRT-LLM Conversion Parameters

This model was converted to TensorRT-LLM format using the following command and parameters:

```bash
python3 /mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py \
  --model_dir /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-Instruct \
  --output_dir /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-streaming \
  --use_weight_only \
  --weight_only_precision int4 \
  --paged_kv_cache \
  --load_model_on_cpu
```

### Key Parameter Descriptions for Streaming Conversion

| Parameter                | Required | Description                                                                                      |
|--------------------------|----------|--------------------------------------------------------------------------------------------------|
| --use_weight_only        | Yes      | Enables weight-only quantization (INT4/INT8).                                                    |
| --weight_only_precision  | Yes      | Precision for quantization. Use `int4` for best streaming support.                               |
| --paged_kv_cache         | Yes      | Enables paged KV cache, which is required for streaming.                                         |
| --load_model_on_cpu      | No       | Loads model on CPU for conversion (safe default, not required for streaming).                    |

#### Optional/Advanced Parameters
| Parameter                | Required | Description                                                                                      |
|--------------------------|----------|--------------------------------------------------------------------------------------------------|
| --per_channel            | No       | More accurate quantization, slightly slower.                                                     |
| --per_token              | No       | More accurate activation quantization, slightly slower.                                          |
| --tp_size/--pp_size      | No       | For multi-GPU conversion. Usually 1 for single-GPU.                                              |
| --dtype                  | No       | Controls weight dtype for non-quantized layers (e.g., `bfloat16`, `float16`).                    |

- **TensorRT-LLM version:** 0.19.0
- **Conversion date:** 2025-06-02
- **Quantization:** INT4 weights, paged KV cache (streaming enabled)
- **Notes:**
    - Use the script `scripts/convert_to_trtllm_streaming.py` for streaming models.
    - Ensure engine build uses `--use_paged_context_fmha false` for streaming support.
    - See this file for parameter explanations and reproducibility.

- **TensorRT-LLM version:** 0.19.0
- **Conversion date:** 2025-06-02
- **Quantization:** INT4 weights, INT8 KV cache
- **Calibration dataset:** ccdv/cnn_dailymail (with trust_remote_code)
- **Conversion time:** ~4.9 hours on RTX 4070 SUPER (CPU calibration)
- **Notes:**
    - Used official script from /mnt/c/Github/baremetalrt/external/TensorRT-LLM-0.19.0/examples/models/core/llama/convert_checkpoint.py
    - Model directory: /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B
    - Output directory: /mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-int4-int8kv
    - Calibration and conversion completed successfully.

---
This file documents the exact parameters used for reproducibility and future reference.
