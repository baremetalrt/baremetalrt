import json
import numpy as np
import struct
import os

# Path to your Llama 3.1 config.json
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/Llama-3.1-8B-Instruct/config.json'))
# Output directory for rotary embedding files
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/Llama-3.1-8B-converted'))

# Helper to write numpy array as .bin (float32 LE)
def save_bin(fname, arr):
    arr = arr.astype(np.float32)
    arr.tofile(fname)
    print(f"[INFO] Saved {fname} with shape {arr.shape} and dtype {arr.dtype}")

def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Extract rotary config
    hidden_size = config.get('hidden_size', 4096)
    rope_scaling = config.get('rope_scaling', {})
    rope_theta = config.get('rope_theta', 500000.0)
    rope_type = rope_scaling.get('rope_type', config.get('rope_type', 'llama3'))
    base = rope_theta if rope_type == 'llama3' else 10000.0

    rotary_dim = hidden_size // config.get('num_attention_heads', 32)
    # Llama 3.1 typically uses rotary_dim = 128 for 8B
    rotary_dim = max(rotary_dim, 128)

    # rotary_inv_freq: 1/(base**(i/rotary_dim)) for i in 0..rotary_dim step 2
    inv_freq = 1.0 / (base ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    save_bin(os.path.join(OUTPUT_DIR, 'rotary_inv_freq.bin'), inv_freq)

    # embed_positions_for_gpt_attention: positions 0..max_position_embeddings-1
    max_pos = config.get('max_position_embeddings', 131072)  # Explicitly set to match your config
    embed_positions = np.arange(max_pos, dtype=np.int32)
    embed_positions.tofile(os.path.join(OUTPUT_DIR, 'embed_positions_for_gpt_attention.bin'))
    print(f"[INFO] Saved {os.path.join(OUTPUT_DIR, 'embed_positions_for_gpt_attention.bin')} with shape {embed_positions.shape} and dtype {embed_positions.dtype}")

if __name__ == '__main__':
    main()
