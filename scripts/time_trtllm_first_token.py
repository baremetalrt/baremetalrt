import time
from tensorrt_llm import LLM, SamplingParams
import sys
import os

# Path to your built TensorRT-LLM engine directory
ENGINE_DIR = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine"

# Prompt (can be set via CLI)
prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, my name is"

# Sampling parameters (customize as needed)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=32,
    end_id=2  # EOS token for Llama models
)

def main():
    print(f"Loading TRT-LLM engine from: {ENGINE_DIR}")
    t0 = time.time()
    llm = LLM(model=ENGINE_DIR)
    t1 = time.time()
    print(f"Engine loaded in {t1 - t0:.2f} seconds")

    print(f"Prompt: {prompt}")
    print("Generating with timing...")
    t2 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    t3 = time.time()

    # Time to first token (TTFT): TRT-LLM returns all tokens at once, so TTFT ~= total time
    output_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(f"\nGenerated: {output_text}")
    print(f"Total generation time: {t3 - t2:.3f} seconds (TTFT)")

if __name__ == "__main__":
    main()
