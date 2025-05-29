import argparse
import sys
import torch
from transformers import AutoTokenizer

try:
    from tensorrt_llm.runtime import ModelRunner
except ImportError:
    print("[ERROR] tensorrt_llm is not installed or not available in your environment.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test inference with a TensorRT-LLM engine.")
    parser.add_argument('--engine', type=str, required=True, help='Path to the TensorRT engine file (.plan)')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to run inference on')
    parser.add_argument('--max_tokens', type=int, default=128, help='Maximum number of tokens to generate')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum batch size for inference')
    parser.add_argument('--max_input_len', type=int, default=512, help='Maximum input length (tokens)')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Maximum sequence length (tokens)')
    parser.add_argument('--max_beam_width', type=int, default=1, help='Maximum beam width for beam search')
    args = parser.parse_args()

    engine_path = args.engine
    prompt = args.prompt
    max_tokens = args.max_tokens
    max_batch_size = args.max_batch_size
    max_input_len = args.max_input_len
    max_seq_len = args.max_seq_len
    max_beam_width = args.max_beam_width

    print(f"[INFO] Loading engine from: {engine_path}")
    try:
        runner = ModelRunner(
            engine_path,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_beam_width=max_beam_width
        )
    except Exception as e:
        print(f"[ERROR] Failed to load engine: {e}")
        sys.exit(1)

    # Use Hugging Face AutoTokenizer from local directory (NGC model)
    tokenizer_dir = "/mnt/c/Github/baremetalrt/external/models/Llama_7b_chat_4070"
    print(f"[INFO] Loading Hugging Face tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    # Patch remove_input_padding if missing or not boolean
    if not hasattr(tokenizer, "remove_input_padding") or not isinstance(tokenizer.remove_input_padding, bool):
        print("[WARN] Patching tokenizer.remove_input_padding = True")
        tokenizer.remove_input_padding = True
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    print(f"[INFO] Tokenized prompt: {input_ids.tolist()}")

    print(f"[INFO] Running inference on prompt: {prompt}")
    try:
        output_ids = runner.generate([input_ids], max_new_tokens=max_tokens)
        # output_ids is a tensor of shape (sequence_length + max_new_tokens,)
        if isinstance(output_ids, torch.Tensor):
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        elif isinstance(output_ids, dict) and 'sequences' in output_ids:
            output_text = tokenizer.decode(output_ids['sequences'][0], skip_special_tokens=True)
        else:
            output_text = str(output_ids)
        print("[RESULT] Output:")
        print(output_text)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
