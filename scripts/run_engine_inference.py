from tensorrt_llm import LLM, SamplingParams

# Path to your built TensorRT-LLM engine directory
ENGINE_DIR = "/mnt/c/Github/baremetalrt/external/models/Llama-3.1-8B-trtllm-engine"

# Example prompts (edit or add more as needed)
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "What is the meaning of life?",
]

# Sampling parameters (customize as needed)
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_new_tokens=64,
    end_id=2  # EOS token for Llama models
)

# Load engine and run inference
if __name__ == "__main__":
    llm = LLM(model=ENGINE_DIR)
    for output in llm.generate(prompts, sampling_params):
        print(f"Prompt: {output.prompt!r}\nGenerated: {output.outputs[0].text!r}\n")
