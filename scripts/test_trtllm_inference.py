import argparse
import sys
import os
import torch
import time
from transformers import AutoTokenizer, PreTrainedTokenizer

def main():
    parser = argparse.ArgumentParser(description="Minimal TensorRT-LLM local inference test.")
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine file (.plan)')
    parser.add_argument('--max_tokens', type=int, default=32, help='Maximum number of tokens to generate')
    args = parser.parse_args()

    prompt = "hello world"
    engine_path = args.engine
    max_tokens = args.max_tokens

    # Try to infer tokenizer path from engine path
    engine_dir = os.path.dirname(engine_path)
    tokenizer_dir = engine_dir  # Assume tokenizer files are in the same directory as engine

    print(f"[INFO] Loading engine from: {engine_path}")
    try:
        from tensorrt_llm.runtime import ModelRunner
    except ImportError:
        print("[ERROR] tensorrt_llm is not installed or not available in your environment.")
        sys.exit(1)

    try:
        runner = ModelRunner(
            engine_path,
            max_batch_size=1,
            max_input_len=512,
            max_seq_len=2048,
            max_beam_width=1,
            kv_cache_type="fp16"
        )
    except Exception as e:
        print(f"[ERROR] Failed to load engine: {e}")
        sys.exit(1)

    print(f"[INFO] Loading Hugging Face tokenizer from: {tokenizer_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Failed to load tokenizer from {tokenizer_dir}: {e}")
        sys.exit(1)

    if not isinstance(tokenizer, PreTrainedTokenizer):
        print(f"[ERROR] Loaded object is not a valid tokenizer: {tokenizer}")
        sys.exit(1)

    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        print(f"[INFO] Tokenized prompt: {input_ids.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to tokenize prompt: {e}")
        sys.exit(1)

    print(f"[INFO] Running inference on prompt: {prompt}")
    try:
        start_time = time.time()
        output_ids = runner.generate([input_ids], max_new_tokens=max_tokens)
        ttft = time.time() - start_time
        if isinstance(output_ids, torch.Tensor):
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elif isinstance(output_ids, dict) and 'sequences' in output_ids:
            output_text = tokenizer.decode(output_ids['sequences'][0], skip_special_tokens=True)
        else:
            output_text = str(output_ids)
        print("[RESULT] Output:")
        print(output_text)
        print(f"[METRIC] Time to first token (TTFT): {ttft:.3f} seconds")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
