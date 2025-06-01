# Llama 3.1-8B TensorRT-LLM Conversion Parameters

This model was converted to TensorRT-LLM format using the following command and parameters:

```bash
python3 ~/baremetalrt/examples/llama/convert_checkpoint.py \
  --model_dir ~/baremetalrt/models/Llama-3.1-8b-hf \
  --output_dir ~/baremetalrt/models/Llama-3.1-8b-trtllm-int4-int8kv \
  --use_weight_only \
  --weight_only_precision int4 \
  --group_size 128 \
  --int8_kv_cache \
  --load_model_on_cpu
```

- **TensorRT-LLM version:** 0.19.0
- **Conversion date:** 2025-05-31
- **Quantization:** INT4 weights, INT8 KV cache
- **Calibration dataset:** ccdv/cnn_dailymail (with trust_remote_code)
- **Conversion time:** ~1.5 hours on RTX 4070 SUPER (CPU calibration)

---
This file documents the exact parameters used for reproducibility and future reference.
