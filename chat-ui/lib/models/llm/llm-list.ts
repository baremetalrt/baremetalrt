import { LLM } from "@/types"
import { ANTHROPIC_LLM_LIST } from "./anthropic-llm-list"
import { GOOGLE_LLM_LIST } from "./google-llm-list"
import { MISTRAL_LLM_LIST } from "./mistral-llm-list"
import { GROQ_LLM_LIST } from "./groq-llm-list"
import { OPENAI_LLM_LIST } from "./openai-llm-list"
import { PERPLEXITY_LLM_LIST } from "./perplexity-llm-list"

export const LLM_LIST: LLM[] = [
  {
    modelId: "llama3.1_8b_trtllm_4int",
    modelName: "Llama 3.1 8B (TensorRT-LLM, 4INT)",
    provider: "custom",
    hostedId: "llama3.1_8b_trtllm_4int",
    platformLink: "https://github.com/NVIDIA/TensorRT-LLM",
    imageInput: false,
    description: "Ultra-fast local inference (TensorRT-LLM, INT4+INT8KV)",
  },
  {
    modelId: "llama2_8b_int8",
    modelName: "Llama 2 8B (INT8)",
    provider: "custom",
    hostedId: "llama2_8b_int8",
    platformLink: "https://huggingface.co/meta-llama/Llama-2-8b-hf",
    imageInput: false,
    description: "Offline: Local INT8 quantized model",
  },
  {
    modelId: "deepseek_7b",
    modelName: "Deepseek LLM 7B",
    provider: "custom",
    hostedId: "deepseek_7b",
    platformLink: "https://huggingface.co/deepseek-ai/deepseek-llm-7b-base",
    imageInput: false,
    description: "Offline: Local Deepseek 7B model",
  },
  {
    modelId: "mixtral_8x7b_instruct_4bit",
    modelName: "Mixtral 8x7B Instruct (4INT)",
    provider: "custom",
    hostedId: "mixtral_8x7b_instruct_4bit",
    platformLink: "https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1",
    imageInput: false,
    description: "Offline: Local Mixtral 8x7B Instruct 4INT",
  },
  {
    modelId: "llama3.1_405b_petals",
    modelName: "Llama 3.1 405B (Petals, API)",
    provider: "custom",
    hostedId: "llama3.1_405b_petals",
    platformLink: "https://petals.dev",
    imageInput: false,
    description: "Distributed API (teaser, coming soon)"
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
