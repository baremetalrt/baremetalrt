"""
Standalone TP checkpoint converter for Llama/Mistral family models.

No tensorrt_llm dependency — just torch + safetensors.
Splits HF weights into per-rank TRT-LLM checkpoint files.

Usage: python convert_tp.py --model_dir models/mistral-7b --output_dir engine_cache/checkpoint --tp_size 2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def convert(model_dir: str, output_dir: str, tp_size: int = 2):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HF config
    with open(model_dir / "config.json") as f:
        hf_config = json.load(f)

    num_heads = hf_config["num_attention_heads"]
    num_kv_heads = hf_config["num_key_value_heads"]
    hidden_size = hf_config["hidden_size"]
    intermediate_size = hf_config["intermediate_size"]
    num_layers = hf_config["num_hidden_layers"]
    vocab_size = hf_config["vocab_size"]
    head_dim = hidden_size // num_heads

    print(f"Model: {hf_config.get('_name_or_path', 'TinyLlama')}")
    print(f"  layers={num_layers}, heads={num_heads}, kv_heads={num_kv_heads}")
    print(f"  hidden={hidden_size}, intermediate={intermediate_size}, vocab={vocab_size}")
    print(f"  TP={tp_size}")

    # Load HF weights
    print("Loading weights...")
    if (model_dir / "model.safetensors").exists():
        state = load_file(str(model_dir / "model.safetensors"))
    else:
        state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")

    # Write TRT-LLM config
    trtllm_config = {
        "architecture": "LlamaForCausalLM",
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "max_position_embeddings": hf_config.get("max_position_embeddings", 2048),
        "hidden_act": "silu",
        "dtype": "float16",
        "mapping": {
            "world_size": tp_size,
            "tp_size": tp_size,
            "pp_size": 1,
        },
        "position_embedding_type": "rope_gpt_neox",
        "rotary_base": hf_config.get("rope_theta", 10000.0),
        "norm_epsilon": hf_config.get("rms_norm_eps", 1e-5),
    }

    # Convert for each rank
    for rank in range(tp_size):
        print(f"\nConverting rank {rank}...")
        rank_dir = output_dir
        weights = {}

        # Embedding — split vocab across ranks
        embed = state["model.embed_tokens.weight"].half()
        chunk = vocab_size // tp_size
        # Don't split embedding for TRT-LLM — each rank gets full embedding
        weights["transformer.vocab_embedding.weight"] = embed.numpy()

        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}"
            trt_prefix = f"transformer.layers.{layer_idx}"

            # Layer norms — not split
            weights[f"{trt_prefix}.input_layernorm.weight"] = \
                state[f"{prefix}.input_layernorm.weight"].half().numpy()
            weights[f"{trt_prefix}.post_layernorm.weight"] = \
                state[f"{prefix}.post_attention_layernorm.weight"].half().numpy()

            # QKV — split Q and K heads across ranks, V follows K
            q = state[f"{prefix}.self_attn.q_proj.weight"].half()
            k = state[f"{prefix}.self_attn.k_proj.weight"].half()
            v = state[f"{prefix}.self_attn.v_proj.weight"].half()

            # Split Q heads
            q_per_rank = num_heads // tp_size
            q_split = q.reshape(num_heads, head_dim, hidden_size)
            q_chunk = q_split[rank * q_per_rank:(rank + 1) * q_per_rank].reshape(-1, hidden_size)

            # Split KV heads
            kv_per_rank = num_kv_heads // tp_size
            k_split = k.reshape(num_kv_heads, head_dim, hidden_size)
            v_split = v.reshape(num_kv_heads, head_dim, hidden_size)
            k_chunk = k_split[rank * kv_per_rank:(rank + 1) * kv_per_rank].reshape(-1, hidden_size)
            v_chunk = v_split[rank * kv_per_rank:(rank + 1) * kv_per_rank].reshape(-1, hidden_size)

            # Concatenate QKV
            qkv = torch.cat([q_chunk, k_chunk, v_chunk], dim=0)
            weights[f"{trt_prefix}.attention.qkv.weight"] = qkv.numpy()

            # Dense (output projection) — split columns
            dense = state[f"{prefix}.self_attn.o_proj.weight"].half()
            dense_cols = hidden_size // tp_size
            weights[f"{trt_prefix}.attention.dense.weight"] = \
                dense[:, rank * dense_cols:(rank + 1) * dense_cols].contiguous().numpy()

            # MLP: TRT-LLM GatedMLP uses fc=up_proj, gate=gate_proj, proj=down_proj
            gate = state[f"{prefix}.mlp.gate_proj.weight"].half()
            up = state[f"{prefix}.mlp.up_proj.weight"].half()
            gate_rows = intermediate_size // tp_size
            gate_chunk = gate[rank * gate_rows:(rank + 1) * gate_rows]
            up_chunk = up[rank * gate_rows:(rank + 1) * gate_rows]
            weights[f"{trt_prefix}.mlp.fc.weight"] = up_chunk.numpy()       # up projection
            weights[f"{trt_prefix}.mlp.gate.weight"] = gate_chunk.numpy()   # gate projection
            weights[f"{trt_prefix}.mlp.proj.weight"] = \
                state[f"{prefix}.mlp.down_proj.weight"].half()[:, rank * gate_rows:(rank + 1) * gate_rows].contiguous().numpy()

        # Final layer norm
        weights["transformer.ln_f.weight"] = state["model.norm.weight"].half().numpy()

        # LM head — split across ranks along vocab dimension (tp_size shards)
        lm_head = state.get("lm_head.weight", state["model.embed_tokens.weight"]).half()
        lm_chunk = vocab_size // tp_size
        weights["lm_head.weight"] = lm_head[rank * lm_chunk:(rank + 1) * lm_chunk].numpy()

        # Save as safetensors
        out_file = rank_dir / f"rank{rank}.safetensors"
        # Convert numpy to torch for safetensors
        torch_weights = {k: torch.from_numpy(v) for k, v in weights.items()}
        save_file(torch_weights, str(out_file))
        print(f"Saved {out_file} ({out_file.stat().st_size // (1024*1024)}MB)")

    # Save config
    config_out = output_dir / "config.json"
    with open(config_out, "w") as f:
        json.dump(trtllm_config, f, indent=2)
    print(f"\nConfig saved to {config_out}")
    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tp_size", type=int, default=2)
    args = parser.parse_args()
    convert(args.model_dir, args.output_dir, args.tp_size)
