import sys
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

# Path to your HuggingFace model directory (AutoDeploy will handle engine loading)
MODEL_DIR = "/home/brian/baremetalrt/models/Llama-3.1-8b-trtllm-int4-int8kv"

# Example prompt (can be replaced by sys.argv[1] if provided)
prompt = sys.argv[1] if len(sys.argv) > 1 else "How big is the universe?"

print(f"Loading TensorRT-LLM engine from: {MODEL_DIR}")
model = LLaMAForCausalLM.from_checkpoint(MODEL_DIR)

print(f"Running inference on prompt: {prompt}")
output = model.generate(prompt, max_new_tokens=100)
print("\n=== Model Output ===\n")
print(output)
