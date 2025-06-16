import { LLM } from "@/types"
import { ANTHROPIC_LLM_LIST } from "./anthropic-llm-list"
import { GOOGLE_LLM_LIST } from "./google-llm-list"
import { MISTRAL_LLM_LIST } from "./mistral-llm-list"
import { GROQ_LLM_LIST } from "./groq-llm-list"
import { OPENAI_LLM_LIST } from "./openai-llm-list"
import { PERPLEXITY_LLM_LIST } from "./perplexity-llm-list"

export const LLM_LIST: LLM[] = [
  {
    modelId: "llama3.1_8b_trtllm_4int_streaming",
    modelName: "Llama 3.1 8B Streaming (TensorRT-LLM, 4INT)",
    provider: "custom",
    hostedId: "llama3.1_8b_trtllm_4int_streaming",
    platformLink: "https://github.com/NVIDIA/TensorRT-LLM",
    imageInput: false,
    description: "Llama 3.1 8B streaming mode (TensorRT-LLM, INT4+INT8KV)",
  },
  {
    modelId: "llama3.1_8b_trtllm_4int",
    modelName: "Llama 3.1 8B Non-Streaming (TensorRT-LLM, 4INT)",
    provider: "custom",
    hostedId: "llama3.1_8b_trtllm_4int",
    platformLink: "https://github.com/NVIDIA/TensorRT-LLM",
    imageInput: false,
    description: "Llama 3.1 8B non-streaming mode (TensorRT-LLM, INT4+INT8KV)",
  }
]

export const LLM_LIST_MAP: Record<string, LLM[]> = {
  openai: OPENAI_LLM_LIST,
  azure: OPENAI_LLM_LIST,
  google: GOOGLE_LLM_LIST,
  mistral: MISTRAL_LLM_LIST,
  groq: GROQ_LLM_LIST,
  perplexity: PERPLEXITY_LLM_LIST,
  anthropic: ANTHROPIC_LLM_LIST
}
